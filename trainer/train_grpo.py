"""【阅读路线 第 7 站（一）】GRPO/CISPO 在线策略优化入口：组内采样回答、评分并更新语言模型策略。

先读 rollout_engine.py 的文件头，了解“采样阶段 / 学习阶段”的两段式框架。

策略梯度的基本思想：让“好于预期”的回答中每个 token 的概率上升，“差于预期”的下降：
    ∇J ≈ E[ A · ∇log π(a_t|s_t) ]
其中 A（advantage，优势）= 这条回答的奖励 − 基线（baseline，“预期能得多少分”）。
减去基线不改变梯度的期望，但能大幅降低方差；关键问题是基线从哪里来。
- PPO（第 7 站（二））训练一个 Critic 网络来预测基线，需要额外一个与策略同规模的模型；
- GRPO（Group Relative Policy Optimization，组相对策略优化）对同一 prompt 采样一组（num_generations 条）
  回答，直接用组内奖励的均值当基线、标准差做归一化：A_i = (r_i − mean(r)) / std(r)。
  不需要 Critic，省显存、实现简单，是 DeepSeek-R1 等推理模型训练的主流做法。

本文件一个训练 step 的流程（对应 grpo_train_epoch 中的顺序）：
  tokenize prompt（左侧补齐）→ rollout 每个 prompt 采样 G 条回答 + old log-prob
  → calculate_rewards 规则分 + 奖励模型分 → policy / reference 分别前向，取回答 token 的 log-prob
  → 组内标准化得到 advantage → 按 EOS 截断得到有效 token mask
  → 组合 token 级损失（重要性比率 × 优势 + KL 惩罚）→ 反向、更新 → 定期把新权重同步给 rollout 引擎。

loss_type 可选：
- grpo：PPO 式的“裁剪概率比”目标；
- cispo（默认）：只对重要性权重做截断并停止其梯度，每个 token 都保留梯度（详见损失处注释）。
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import math
import re
import gc
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, LMForRewardModel
from trainer.rollout_engine import create_rollout_engine

warnings.filterwarnings('ignore')


def rep_penalty(text, n=3, cap=0.5):
    # 重复惩罚：统计 n-gram（默认 3 个词/符号为一组）中重复出现的比例，比例越高扣分越多，最多扣 cap。
    # RL 中模型很容易发现“重复某些句子能刷高奖励模型分数”这类漏洞（reward hacking），需要规则项约束。
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


def calculate_rewards(prompts, responses, reward_model):
    # 奖励由规则项和 reward model 分数组成，每条回答得到一个“序列级”标量（整条回答一个分数，不是每个 token 一个）。
    # 规则项：长度是否在合理区间、思考段 <think>...</think> 长度与闭合格式是否正确、重复惩罚；
    # 模型项：外部奖励模型对“对话上下文 + 去掉思考段后的最终回答”打分（见 LMForRewardModel）。
    # GRPO 的关键不是绝对分值，而是同一 prompt 下多条回答的相对排名（组内标准化优势）。
    # responses 的排列顺序是 [prompt0 的 G 条, prompt1 的 G 条, ...]，所以下标 = i*G + j。
    rewards = torch.zeros(len(responses), device=args.device)

    with torch.no_grad():
        reward_model_scores = []
        batch_size = len(prompts)

        for i in range(batch_size):
            for j in range(args.num_generations):
                response_idx = i * args.num_generations + j
                response = responses[response_idx]
                prompt = prompts[i]

                # 从模板渲染后的 prompt 字符串里反解出 messages 列表，供奖励模型重新组织输入。
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                answer = response
                rewards[response_idx] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
                if '</think>' in response:
                    thinking_content, answer_content = response.split('</think>', 1)
                    rewards[response_idx] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                    rewards[response_idx] += 0.25 if response.count('</think>') == 1 else -0.25
                    answer = answer_content.strip()
                rewards[response_idx] -= rep_penalty(answer)

                score = reward_model.get_score(messages, answer)
                reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def grpo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model, start_step=0, wandb=None, use_sglang=False):
    """执行采样、奖励归一化、token 级损失计算和策略更新。"""
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch['prompt']  # list[str], length B
        # 左侧补齐（padding_side="left"）：生成是在序列右端续写，左补齐让 batch 内所有 prompt 的
        # 最后一个真实 token 都落在同一列，新 token 能整齐地接在后面；右补齐则会让短 prompt 后面夹着 PAD。
        # 超长时保留右侧（最近的）max_seq_len 个 token。
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                                  padding_side="left", add_special_tokens=False).to(args.device)
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        # ===== 采样阶段 =====
        # 一轮更新分两段：先用当前策略无梯度采样并评分，再重算 log-prob 求梯度。
        # rollout 引擎返回生成 token、mask 和旧策略 log-prob，训练目标只覆盖 completion。
        # temperature=0.8 的随机采样保证同一 prompt 的 G 条回答各不相同；若 G 条完全一样，
        # 组内奖励方差为 0，优势全为 0，这个 prompt 就不提供任何学习信号。
        rollout_result = rollout_engine.rollout(
            prompt_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            num_generations=args.num_generations,
            max_new_tokens=args.max_gen_len,
            temperature=0.8,
        )
        outputs = rollout_result.output_ids
        completion_ids = rollout_result.completion_ids
        completions = rollout_result.completions
        # outputs=[B*num_gen, P+R]，completion_ids=[B*num_gen,R]；
        # old_per_token_logps 保存采样策略对 R 个回答 token 的 log π_old。
        old_per_token_logps = rollout_result.per_token_logps.to(args.device).detach()
        prompt_lens = rollout_result.prompt_lens.to(args.device)
        # attention_mask：先把所有 PAD 位置（主要是 prompt 左侧补齐）标为 0 ……
        full_mask = (outputs != tokenizer.pad_token_id).long()
        # logp_pos[i, j]：第 i 条序列中“预测回答第 j 个 token”的 logit 所在位置。
        # 回答第 j 个 token 位于 outputs 的 prompt_len+j；自回归中它由前一个位置预测，
        # 而 logits[:, :-1] 的第 k 列预测的是 outputs[:, k+1]，所以对应列号是 prompt_len−1+j。
        logp_pos = prompt_lens.unsqueeze(1) - 1 + torch.arange(completion_ids.size(1), device=args.device).unsqueeze(0)
        # …… 再把回答区间（位置 logp_pos+1）按 completion_mask 覆盖写入：
        # 回答里若恰好生成了与 PAD 同 id 的 token，也不会被误当成 padding 屏蔽掉。
        full_mask.scatter_(1, logp_pos + 1, rollout_result.completion_mask.to(args.device, dtype=full_mask.dtype))

        rewards = calculate_rewards(prompts, completions, reward_model).to(args.device)  # [B*num_gen]

        # ===== 学习阶段 =====
        with autocast_ctx:
            # 对完整序列做一次有梯度的前向（这就是“当前策略” π_θ）。
            res = model(outputs, attention_mask=full_mask)
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
            # 两次 gather：第一次在词表维取出每个位置“实际下一个 token”的 log-prob → [N, P+R-1]；
            # 第二次按 logp_pos 在序列维只取回答区间 → [N, R]。prompt 部分不是策略动作，不参与损失。
            per_token_logps = F.log_softmax(res.logits[:, :-1, :], dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)

        # reference（冻结的初始策略，通常是 SFT 模型）对同一批 token 的 log-prob，用于 KL 惩罚。
        with torch.no_grad():
            ref_per_token_logps = F.log_softmax(ref_model(outputs, attention_mask=full_mask).logits[:, :-1, :], dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)

        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(len(prompts)):
                Logger(f"[DEBUG] step={step}, sample[{i}]")
                Logger('-'*100)
                Logger(f"{'=' * 30} [DEBUG] sample[{i}] CONTEXT_BEGIN {'=' * 30}")
                Logger(prompts[i])
                Logger(f"{'=' * 31} [DEBUG] sample[{i}] CONTEXT_END {'=' * 31}")
                for j in range(args.num_generations):
                    idx = i * args.num_generations + j
                    Logger(f"{'=' * 28} [DEBUG] gen[{j}] RESPONSE_BEGIN {'=' * 28}")
                    Logger(completions[idx])
                    Logger(f"{'=' * 29} [DEBUG] gen[{j}] RESPONSE_END {'=' * 29}")
                    Logger(f"[DEBUG] gen[{j}] reward={rewards[idx].item():.4f}")
                Logger('='*100)

        # ===== 组内相对优势 =====
        # view(-1, G) 把同一 prompt 的 G 条回答放到同一行，按行求均值/标准差，再 repeat_interleave 展开回每条回答。
        grouped_rewards = rewards.view(-1, args.num_generations)  # [B, num_gen]
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)  # [B*num_gen]
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)  # [B*num_gen]
        # advantage（优势）估计回答比基线好/差多少：A_i = (r_i − 组均值) / (组标准差 + ε)。
        # 奖励高于组均值的回答为正优势（其 token 概率会被推高），低于均值的为负优势（被压低）；
        # 除以标准差让不同 prompt 的优势尺度可比——有的题目奖励普遍在 2 分附近，有的在 −1 附近，
        # 标准化后都只看“组内排名”。组内奖励全相同时 std=0，优势全为 0，只剩 KL 项产生梯度。
        # 注意优势是“序列级”的：同一条回答里的所有 token 共享同一个 A（下面 unsqueeze(1) 广播到 R 个 token），
        # GRPO 不区分回答中哪个 token 贡献更大，这与 PPO 用 GAE 给每个 token 单独估计优势不同。
        advantages = (rewards - mean_r) / (std_r + 1e-4)  # [B*num_gen]

        # ===== 有效 token mask =====
        completion_pad_mask = rollout_result.completion_mask.to(args.device).bool()
        # 回答的有效部分 = 从第一个生成 token 到第一个 EOS（含 EOS 本身，模型需要学会“何时停止”）。
        # generate 在序列结束后会继续填充 EOS，这些位置不是真正的动作，必须排除；SGLang 后端的右侧补齐也要排除。
        # 未生成 EOS（达到 max_gen_len 被截断）的回答整段有效。
        # 后续的 token 级策略损失与 KL 惩罚都乘这个 mask，避免无效 token 影响平均值。
        is_eos = (completion_ids == tokenizer.eos_token_id) & completion_pad_mask  # [B*num_gen, R]
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1) - 1, dtype=torch.long, device=args.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        completion_mask = ((torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)) & completion_pad_mask).int()  # [B*num_gen, R]

        # ===== 损失函数 =====
        # 三个 [N, R] 的 log-prob 各司其职：
        #   per_token_logps      当前策略 π_θ（有梯度）
        #   old_per_token_logps  采样时的策略 π_old（常数）→ 构造重要性比率 ratio
        #   ref_per_token_logps  冻结的参考策略 π_ref（常数）→ 构造 KL 惩罚
        #
        # ① KL 惩罚：防止策略为了刷高奖励而偏离 SFT 模型太远（语言退化、奖励黑客）。
        #    精确计算 KL 需要对整个词表求和，这里只用“被采样到的 token”做无偏估计（Schulman 的 k3 估计器）：
        #    设 x = log π_ref − log π_θ，则 KL(π_θ ‖ π_ref) ≈ exp(x) − x − 1。
        #    它恒 ≥ 0（因为 e^x ≥ 1+x），在 π_θ = π_ref 时为 0，比直接用 −x（k1）方差更小。
        #    这里把 KL 直接加进 loss（而不是像经典 RLHF-PPO 那样从奖励里扣除），是 GRPO 论文的做法。
        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1  # [B*num_gen, R]
        # ② 重要性采样比率 ratio = π_θ(a) / π_old(a) = exp(log π_θ − log π_old)。
        #    数据由 π_old 采样，却要估计 π_θ 下的期望，就要乘这个比率做修正。
        #    在 Torch 后端下每批数据只更新一次、且采样与学习之间参数没变，ratio 的数值≈1（仅有浮点误差），
        #    但它的梯度 ∇ratio = ratio·∇log π_θ ≈ ∇log π_θ 仍然正确，损失此时退化为标准策略梯度。
        #    只有在 SGLang 后端（权重同步有延迟）等 off-policy 情况下 ratio 才偏离 1，裁剪/截断才会生效。
        ratio = torch.exp(per_token_logps - old_per_token_logps)  # [B*num_gen, R]
        if args.loss_type == "cispo":
            # ③a CISPO：loss = −[ sg(min(ratio, ε_high)) · A · log π_θ − β·KL ]，sg 表示停止梯度（detach）。
            #    重要性权重只做上限截断并当作常数，梯度恒为 sg(权重)·A·∇log π_θ（REINFORCE 形式）。
            #    与 PPO clip 的区别：PPO 在比率超出区间时，该 token 的梯度会直接变成 0（被“丢弃”）；
            #    CISPO 只是限制权重大小，所有 token 都始终贡献梯度，对推理链中少数关键 token 更友好。
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
            per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
        else:
            # ③b PPO 式裁剪目标：loss = −[ min(ratio·A, clip(ratio, 1−ε, 1+ε)·A) − β·KL ]。
            #    取 min 是“悲观”估计：A>0 时 ratio 超过 1+ε 就不再给奖励（防止把好动作概率一次推得过高），
            #    A<0 时 ratio 低于 1−ε 就不再惩罚（防止一次压得过低）。被裁剪的一侧梯度为 0。
            clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            per_token_loss1 = ratio * advantages.unsqueeze(1)
            per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
            per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)
        # ④ 聚合：先在每条回答内部对有效 token 求平均，再对所有回答求平均（原始 GRPO 的“样本级”平均）。
        #    这意味着每条回答权重相同，长回答中单个 token 的权重被摊薄；
        #    另一种做法是对全部有效 token 统一求平均（token 级，如 DAPO），长回答的影响会更大。
        policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).mean()
        loss = (policy_loss + aux_loss) / args.accumulation_steps  # scalar
        # RL 脚本不使用 GradScaler：默认 bf16 不需要；若改用 fp16 则存在梯度下溢风险。
        loss.backward()

        if step % args.accumulation_steps == 0 or step == iters:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            # RL 脚本用 PyTorch 自带的 CosineAnnealingLR：每次 optimizer 更新后调用一次 scheduler.step()，
            # 学习率沿余弦曲线从 learning_rate 降到 learning_rate/10（与 get_lr 的手写方式效果类似）。
            scheduler.step()
            optimizer.zero_grad()

        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item() * args.accumulation_steps
            current_aux_loss = aux_loss.item()
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            kl_ref_val = ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item() / max(completion_mask.sum().item(), 1)
            advantages_mean_val = advantages.mean().item()
            advantages_std_val = advantages.std().item()
            current_lr = optimizer.param_groups[0]['lr']

            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'Reward: {avg_reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, '
                   f'Adv Std: {advantages_std_val:.4f}, Adv Mean: {advantages_mean_val:.4f}, '
                   f'Actor Loss: {policy_loss_val:.4f}, Avg Response Len: {avg_len_val:.2f}, Learning Rate: {current_lr:.8f}')

            if wandb and is_main_process():
                wandb.log({
                    "reward": avg_reward_val,
                    "kl_ref": kl_ref_val,
                    "advantages_std": advantages_std_val,
                    "advantages_mean": advantages_mean_val,
                    "policy_loss": policy_loss_val,
                    "avg_response_len": avg_len_val,
                    "learning_rate": current_lr
                })

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
            model.train()
            del state_dict

        # 把最新权重同步给 rollout 引擎（Torch 后端只是更新引用；SGLang 后端会写盘并通知服务端重载）。
        if step % args.save_interval == 0 or step == iters: rollout_engine.update_policy(model)

        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask, completion_pad_mask, prompt_lens, logp_pos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind GRPO (Group Relative Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='grpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=768, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif.jsonl", help="RLAIF数据路径")
    parser.add_argument("--num_generations", type=int, default=6, help="每个prompt生成的样本数")
    parser.add_argument("--beta", type=float, default=0.1, help="KL惩罚系数")
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"], help="loss类型")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO的PPO clip epsilon")
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="epsilon上界")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-GRPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug模式下每隔多少step打印一次采样")
    parser.add_argument("--thinking_ratio", type=float, default=0.9, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_grpo", help="SGLang共享存储路径")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               max_seq_len=args.max_seq_len + args.max_gen_len, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 初始化模型和数据 ==========
    # GRPO 同时需要三个模型：policy（被训练）、reference（冻结，KL 锚点）、reward model（冻结，打分）。
    # 与 PPO 相比少了 Critic，这正是 GRPO 省显存的来源。
    base_weight = args.from_weight
    # Policy模型
    model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    # Reference模型
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    # Reward模型
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    # Rollout引擎（可插拔替换，只负责 policy 推理）
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )
    # 数据和优化器
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len, thinking_ratio=args.thinking_ratio)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        # 同 train_ppo：RoPE buffer 各 rank 一致，每步广播纯属浪费
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    rollout_engine.update_policy(model)
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            grpo_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, reward_model, start_step, wandb, use_sglang = (args.rollout_engine == "sglang"))
        else:
            grpo_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, reward_model, 0, wandb, use_sglang = (args.rollout_engine == "sglang"))
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
