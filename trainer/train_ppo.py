"""【阅读路线 第 7 站（二）】PPO 在线强化学习入口：Actor 生成回答，Critic 估计价值并协同训练。

建议先读完 train_grpo.py，再对比阅读本文件。两者共享“采样 → 打分 → 重算 log-prob → 策略损失”的框架，
核心区别在于“优势 A 从哪里来”：
- GRPO：同一 prompt 采样多条回答，用组内均值当基线，整条回答共享一个 A；
- PPO：每个 prompt 只采样 1 条，训练一个 Critic（价值模型）预测每个位置的 V(s_t)
  （“从这里往后预计能拿多少奖励”），再用 GAE 给回答中的“每个 token”单独估计优势。
  代价是多一个与 Actor 同规模、同样要训练的模型。

本脚本中的四个模型，职责务必分清：
- Actor（actor_model）：被训练的策略，负责生成回答；
- Critic（critic_model）：被训练的价值模型，输出每个位置的标量价值，只用于构造优势，不生成文字；
- Reference（ref_model）：冻结的初始策略（SFT 模型），只用于 KL 惩罚，防止 Actor 跑偏；
- Reward model（reward_model）：冻结的打分模型，给整条回答一个序列级奖励。

PPO（Proximal Policy Optimization，近端策略优化）的“近端”：同一批采样数据会被重复用于多轮更新
（ppo_update_iters），为了防止在旧数据上把策略改得太多，用裁剪的概率比限制每次更新的幅度，
并用 approx_kl 早停。这也是本脚本里 ratio 真正偏离 1、裁剪真正生效的场景
（Actor 在同一批数据上完成第一次参数更新之后）。

ppo_train_epoch 的流程：采样与打分 → 构造 mask → 无梯度地计算 old value / ref log-prob
→ 奖励放到最后一个 token → GAE 反向递推得到逐 token 优势与回报 → 优势标准化
→ 多轮 × minibatch 更新 Actor（裁剪策略损失 + KL）与 Critic（裁剪价值损失）。
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import math
import re
import warnings
import torch
import torch.distributed as dist
import torch.nn.functional as F
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, LMForRewardModel
from trainer.rollout_engine import create_rollout_engine

warnings.filterwarnings('ignore')


def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


# 自定义的Critic模型，继承自MiniMindLM
class CriticModel(MiniMindForCausalLM):
    """Critic 在每个 token 位置估计状态价值 V(s)，不负责生成词表 logits。

    结构 = 语言模型主干（MiniMindModel）+ 一个 H→1 的 value_head。
    位置 t 的输出 V(s_t) 读作：“看完 prompt 和前 t 个回答 token 后，预计这条回答最终能拿多少奖励”。
    主干从 SFT 权重初始化（复用其语言理解能力），value_head 是新加的、随机初始化的层；
    因此训练初期 Critic 的估计很不准，常需要较高的学习率或预热（这里 critic lr 略高于 actor lr）。
    """
    def __init__(self, params):
        super().__init__(params)
        # lm_head 不参与 forward，仅靠 tie_word_embeddings 与 embed_tokens 共享权重，解绑后 DDP 会报未使用参数
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # 使用基础模型获取隐藏状态 [B, S, H]
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        # 注：MiniMindModel.forward 返回前已经做过一次最终 norm，这里又做了一次 RMSNorm
        # （RMSNorm 近似幂等，只多了一次逐维缩放），属于实现上的冗余，不影响理解。
        hidden_states = self.model.norm(outputs[0])
        # 使用value_head获取价值估计：[B, S, H] → [B, S, 1] → [B, S]，每个位置一个标量
        values = self.value_head(hidden_states).squeeze(-1)
        return values


def calculate_rewards(prompts, responses, reward_model):
    # 将可编程规则（长度、思考格式、重复惩罚）与偏好模型分数合成奖励；规则与 train_grpo.py 完全相同。
    # 每条回答得到一个序列级标量 [B]。先理解奖励的来源，再看 PPO 如何把序列级奖励分配到 token 级优势。
    rewards = torch.zeros(len(responses), device=args.device)

    with torch.no_grad():
        reward_model_scores = []
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]
            answer = response
            rewards[i] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
            if '</think>' in response:
                thinking_content, answer_content = response.split('</think>', 1)
                rewards[i] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                rewards[i] += 0.25 if response.count('</think>') == 1 else -0.25
                answer = answer_content.strip()
            rewards[i] -= rep_penalty(answer)

            score = reward_model.get_score(messages, answer)
            reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def ppo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, actor_scheduler, critic_scheduler, reward_model, start_step=0, wandb=None):
    """先采样并评分，再用 Actor/Critic 多轮拟合该批轨迹。"""
    actor_model.train()
    critic_model.train()
    grad_accum_step = 0

    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]  # list[str], length B
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_len,
                        padding_side="left").to(args.device)  # input_ids: [B, P], attention_mask: [B, P]

        # ===== 采样阶段 =====
        # rollout 是采样阶段，需保留旧策略概率；后续 actor 更新按 PPO 多轮优化，
        # critic 学习价值基线，reference 提供 KL 正则。num_generations=1：PPO 的基线来自 Critic，不需要组采样。
        rollout_result = rollout_engine.rollout(
            prompt_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            num_generations=1,
            max_new_tokens=args.max_gen_len,
            temperature=0.8,
        )
        gen_out = rollout_result.output_ids
        completion_ids = rollout_result.completion_ids
        prompt_lens = rollout_result.prompt_lens.to(args.device)
        responses_text = rollout_result.completions
        old_resp_logp = rollout_result.per_token_logps.to(args.device)
        rewards = calculate_rewards(prompts, responses_text, reward_model)  # [B]

        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(len(prompts)):
                Logger(f"[DEBUG] step={step}, sample[{i}]")
                Logger('-'*100)
                Logger(f"{'=' * 30} [DEBUG] sample[{i}] CONTEXT_BEGIN {'=' * 30}")
                Logger(prompts[i])
                Logger(f"{'=' * 31} [DEBUG] sample[{i}] CONTEXT_END {'=' * 31}")
                Logger(f"[DEBUG] prompt_len={prompt_lens[i].item()}, response_len={len(responses_text[i])}")
                Logger(f"{'=' * 28} [DEBUG] sample[{i}] RESPONSE_BEGIN {'=' * 28}")
                Logger(responses_text[i])
                Logger(f"{'=' * 29} [DEBUG] sample[{i}] RESPONSE_END {'=' * 29}")
                Logger(f"[DEBUG] reward={rewards[i].item():.4f}")
                Logger('='*100)

        # ===== 构造 mask 与位置索引（与 train_grpo.py 的做法相同）=====
        full_mask = (gen_out != tokenizer.pad_token_id).long()  # [B, P+R]
        # labels[k] = gen_out[k+1]：logits[:, :-1] 第 k 列应预测的 token。
        labels = gen_out[:, 1:].clone()  # [B, P+R-1]
        B = len(prompts)
        resp_labels = completion_ids
        resp_idx = torch.arange(resp_labels.size(1), device=gen_out.device).unsqueeze(0)
        # logp_pos[i, j]：预测回答第 j 个 token 的 logit 列号（= prompt_len − 1 + j），
        # 同时也是“状态 s_j”（看完 prompt 与前 j 个回答 token）对应的位置，Critic 的价值也从这里取。
        logp_pos = prompt_lens.unsqueeze(1) - 1 + resp_idx
        resp_pad_mask = rollout_result.completion_mask.to(args.device).bool()
        full_mask.scatter_(1, logp_pos + 1, resp_pad_mask.to(full_mask.dtype))
        # 有效回答长度 = 截至第一个 EOS（含）；没有 EOS 的回答整段有效。
        # resp_policy_mask / resp_value_mask：[B, R]，1 表示该 token 参与策略损失 / 价值损失。
        resp_lengths = resp_pad_mask.sum(dim=1); valid_resp = resp_lengths > 0; eos_mask = resp_labels.eq(tokenizer.eos_token_id) & resp_pad_mask
        has_eos = eos_mask.any(dim=1); eos_pos = torch.argmax(eos_mask.int(), dim=1)
        resp_lengths = torch.where(has_eos, eos_pos + 1, resp_lengths).long().clamp(min=1)
        resp_policy_mask = ((resp_idx < resp_lengths.unsqueeze(1)) & resp_pad_mask).float()
        resp_value_mask = resp_policy_mask.clone()

        # rollout 时冻结计算图得到行为策略概率与旧价值；trajectory（轨迹）是
        # prompt、模型生成 token、奖励和价值组成的一次交互记录。
        with torch.no_grad():  # Rollout阶段只需推理获取old_logp和old_values，切断梯度省显存
            # old value：更新前 Critic 对每个状态的估计，既用于 GAE，也作为价值裁剪的参照。
            # 无效位置乘 mask 置 0，GAE 递推到回答末尾之后时“下一状态价值”自然为 0（终止状态）。
            critic_for_rollout = critic_model.module if isinstance(critic_model, DistributedDataParallel) else critic_model
            values_seq = critic_for_rollout(input_ids=gen_out, attention_mask=full_mask)
            old_resp_values = values_seq.gather(1, logp_pos) * resp_value_mask

            ref_resp_logp = F.log_softmax(ref_model(input_ids=gen_out, attention_mask=full_mask).logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)
            # ===== 奖励分配 =====
            # reward model 给整条回答一个序列级分数；将它放在最后一个有效回答 token 上（其余 token 奖励为 0），
            # 这是“稀疏的终局奖励”：只有答完才知道好坏。让 GAE 递推把最终结果向前传播到较早动作。
            # （经典 RLHF-PPO 还会在每个 token 的奖励里扣除 β·KL；本实现改为把 KL 作为独立的损失项。）
            token_rewards = torch.zeros_like(old_resp_logp)
            last_idx = resp_lengths - 1  # [B]
            token_rewards[torch.arange(B, device=args.device)[valid_resp], last_idx[valid_resp]] += rewards[valid_resp]  # 末尾加外部奖励

            # ===== GAE（Generalized Advantage Estimation，广义优势估计）=====
            # TD 残差 δ_t = r_t + γ·V(s_{t+1}) − V(s_t)：“走了这一步之后，对结局的预估改善了多少”，
            #   是对单步优势的一个低方差、但依赖 Critic 准确度（有偏）的估计。
            # GAE 把未来各步的 δ 按 (γλ) 衰减加权求和：A_t = δ_t + γλ·δ_{t+1} + (γλ)²·δ_{t+2} + …
            #   从后往前递推即 A_t = δ_t + γλ·A_{t+1}（下面循环里的 lastgaelam）。
            # λ 在偏差与方差之间折中：λ=0 时 A_t = δ_t（只看一步，方差小、完全依赖 Critic）；
            #   λ=1 时 A_t = Σ 未来奖励 − V(s_t)（蒙特卡洛回报，无偏但方差大）。默认 λ=0.95。
            # γ 是未来奖励折扣，语言任务通常取 1.0（回答长度有限，末尾奖励不应被打折）。
            # 由于奖励只在最后一个 token，这个递推本质上就是把“终局奖励 − 各位置的价值预估”传播回每个 token。
            gen_len = old_resp_values.size(1); lastgaelam = torch.zeros(B, device=args.device); advs_rev = []
            for t in reversed(range(gen_len)):
                nv = old_resp_values[:, t + 1] if t < gen_len - 1 else 0.0
                delta = token_rewards[:, t] + args.gamma * nv - old_resp_values[:, t]
                lastgaelam = delta + args.gamma * args.lam * lastgaelam
                advs_rev.append(lastgaelam)
            advantages = torch.stack(advs_rev[::-1], dim=1)  # [B, R]
            # returns（回报）= A_t + V(s_t)，是 Critic 的回归目标：让 V(s_t) 去拟合“实际获得的（λ-）回报”。
            returns = advantages + old_resp_values  # [B, R]

            # 优势标准化（whitening）：在整个 batch 的有效 token 上减均值、除标准差。
            # 让优势量级稳定在 O(1)，不随奖励尺度漂移，学习率更好调；也让大约一半 token 被推高、一半被压低。
            adv_mean = (advantages * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
            adv_var = ((advantages - adv_mean) ** 2 * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
            advantages = (advantages - adv_mean) * torch.rsqrt(adv_var + 1e-8) * resp_policy_mask

        # ===== 学习阶段：同一批轨迹重复使用 ppo_update_iters 轮，每轮打乱后切成 minibatch =====
        # 重复利用昂贵的采样数据以提高样本效率；old_resp_logp / old_resp_values / advantages / returns
        # 在各轮之间保持不变（都是采样时的“快照”），只有 Actor/Critic 的当前输出随更新而变化。
        mb_size = max(1, min(args.mini_batch_size, B))
        stop_ppo = False
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        kl_sum = 0.0
        kl_ref_sum = 0.0
        clipfrac_sum = 0.0
        aux_loss_sum = 0.0
        log_count = 0
        for ppo_epoch in range(args.ppo_update_iters):
            if stop_ppo:
                break
            b_inds = torch.randperm(B, device=args.device)
            for i in range(0, B, mb_size):
                inds = b_inds[i:i + mb_size]

                # Critic 与 Actor 的当前（有梯度）前向：得到新价值与新 log-prob。
                mb_values_seq = critic_model(input_ids=gen_out[inds], attention_mask=full_mask[inds])
                mb_resp_values = mb_values_seq.gather(1, logp_pos[inds])

                with autocast_ctx:
                    res = actor_model(input_ids=gen_out[inds], attention_mask=full_mask[inds])
                    aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
                    mb_resp_logp = F.log_softmax(res.logits[:, :-1], dim=-1).gather(2, labels[inds].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos[inds])

                log_ratio = mb_resp_logp - old_resp_logp[inds]

                # 可开关的诊断：观察首轮首个 minibatch 的 mb 与 old logp 差异。
                if args.debug_log_ratio and ppo_epoch == 0 and i == 0 and is_main_process():
                    _lr = log_ratio.detach()
                    _m = resp_policy_mask[inds].bool()
                    if _m.any():
                        _lrv = _lr[_m]
                        Logger(f"[DBG log_ratio] step={step} max|lr|={_lrv.abs().max().item():.6e} "
                               f"mean|lr|={_lrv.abs().mean().item():.6e} "
                               f"ratio_max={torch.exp(_lrv).max().item():.6f} "
                               f"ratio_min={torch.exp(_lrv).min().item():.6f} "
                               f"dropout={getattr(lm_config, 'dropout', None)} "
                               f"training={actor_model.training}")
                # approx_kl：当前策略相对“采样时策略”偏离了多少，用 k2 估计器 ½·(log ratio)² 近似 KL(π_old ‖ π_θ)。
                # 它与下面的 kl_ref_penalty 是两个不同的 KL：前者衡量“这批数据上已经改了多少”（用于早停），
                # 后者衡量“离 SFT 模型有多远”（作为正则项进入损失）。
                # 早停：超过 early_stop_kl 说明在这批旧数据上已经更新得够多了，继续更新会使重要性修正失真。
                approx_kl = (0.5 * (log_ratio ** 2) * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)

                # 同步各卡的 approx_kl，防止某卡 break 而其它卡继续导致 DDP 死锁
                approx_kl_val = approx_kl.detach().clone()
                if dist.is_initialized():
                    dist.all_reduce(approx_kl_val, op=dist.ReduceOp.AVG)
                    
                if approx_kl_val > args.early_stop_kl:
                    stop_ppo = True
                
                ratio = torch.exp(log_ratio)
                # clipfrac：有多少比例的 token 的 ratio 落在 [1−ε, 1+ε] 之外。常用监控指标：
                # 长期接近 0 说明更新很保守；过高说明每轮更新幅度太大，可考虑减小学习率。
                clipfrac = ((((ratio - 1.0).abs() > args.clip_epsilon).float() * resp_policy_mask[inds]).sum()
                            / resp_policy_mask[inds].sum().clamp(min=1))
                # 相对 reference 的 KL 惩罚，k3 估计器 exp(x) − x − 1（x = log π_ref − log π_θ），与 GRPO 相同。
                kl_ref_penalty = ((torch.exp(ref_resp_logp[inds] - mb_resp_logp) - (ref_resp_logp[inds] - mb_resp_logp) - 1.0)
                                  * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
                # PPO clipped objective：同时计算未裁剪和裁剪后的策略目标，取更保守的一项。
                # 写成损失（取负）后 min 变成 max：policy_loss = max(−A·ratio, −A·clip(ratio)) 在有效 token 上的平均。
                # 与 GRPO 的区别：这里的 advantages 是 GAE 给出的逐 token 优势，而不是整条回答共享的一个值；
                # 并且这里对 batch 内全部有效 token 统一求平均（token 级），而 GRPO 是先对每条回答内部求平均。
                #
                # value_loss：Critic 的回归损失 ½·(V − returns)²，同样做裁剪——新价值相对 old value 的变化超过
                # ±cliprange_value 时，用裁剪后的值计算误差，并取两者较大者（悲观），防止 Critic 一步跳得太远。
                # vf_coef 平衡 Actor 与 Critic 损失。二者参数不共享，这里加在一起只是为了一次 backward。
                policy_loss = ((torch.max(-advantages[inds] * ratio,
                                          -advantages[inds] * torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon))
                               * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
                               + args.kl_coef * kl_ref_penalty)
                value_loss = 0.5 * (torch.max((mb_resp_values - returns[inds]) ** 2,
                                              (torch.clamp(mb_resp_values, old_resp_values[inds] - args.cliprange_value,
                                                           old_resp_values[inds] + args.cliprange_value) - returns[inds]) ** 2)
                                    * resp_value_mask[inds]).sum() / resp_value_mask[inds].sum().clamp(min=1)

                kl = approx_kl_val
                kl_ref = kl_ref_penalty.detach()

                # 早停时必须保证 forward-backward 闭环，故只截断 loss 不中断 DDP 通信
                # （DDP 在 forward 后期待一次 backward 来触发梯度 all-reduce；loss×0 让梯度为 0 但通信照常完成。）
                if stop_ppo:
                    loss = (policy_loss + args.vf_coef * value_loss + aux_loss) * 0.0
                else:
                    loss = (policy_loss + args.vf_coef * value_loss + aux_loss) / args.accumulation_steps
                
                loss.backward()

                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                kl_sum += kl.item()
                kl_ref_sum += kl_ref.item()
                clipfrac_sum += clipfrac.item()
                aux_loss_sum += aux_loss.item()
                log_count += 1

                grad_accum_step += 1

                # Actor 和 Critic 各有独立的优化器与学习率调度器，按各自的梯度分别裁剪、更新。
                if grad_accum_step % args.accumulation_steps == 0:
                    clip_grad_norm_(actor_model.parameters(), args.grad_clip)
                    clip_grad_norm_(critic_model.parameters(), args.grad_clip)
                    actor_optimizer.step()
                    critic_optimizer.step()
                    actor_scheduler.step()
                    critic_scheduler.step()
                    actor_optimizer.zero_grad()
                    critic_optimizer.zero_grad()

        if grad_accum_step % args.accumulation_steps != 0:
            clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            actor_optimizer.step()
            critic_optimizer.step()
            actor_scheduler.step()
            critic_scheduler.step()
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
        
        if step % args.save_interval == 0 or step == iters: rollout_engine.update_policy(actor_model)

        if is_main_process():
            critic_loss_val = value_loss_sum / max(log_count, 1)
            reward_val = rewards.mean().item()
            approx_kl_val = kl_sum / max(log_count, 1)
            kl_ref_val = kl_ref_sum / max(log_count, 1)
            clipfrac_val = clipfrac_sum / max(log_count, 1)
            avg_len_val = resp_lengths.float().mean().item()
            actor_lr, critic_lr = actor_optimizer.param_groups[0]['lr'], critic_optimizer.param_groups[0]['lr']

            if wandb is not None:
                wandb.log({
                    "reward": reward_val,
                    "kl_ref": kl_ref_val,
                    "approx_kl": approx_kl_val,
                    "clipfrac": clipfrac_val,
                    "critic_loss": critic_loss_val,
                    "avg_response_len": avg_len_val,
                    "actor_lr": actor_lr,
                    "critic_lr": critic_lr,
                })

            Logger(f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                   f"Reward: {reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, Approx KL: {approx_kl_val:.4f}, "
                   f"ClipFrac: {clipfrac_val:.4f}, Critic Loss: {critic_loss_val:.4f}, "
                   f"Avg Response Len: {avg_len_val:.2f}, Actor LR: {actor_lr:.8f}, Critic LR: {critic_lr:.8f}")

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            actor_model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_actor = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            raw_actor = getattr(raw_actor, '_orig_mod', raw_actor)
            actor_state = raw_actor.state_dict()
            torch.save({k: v.half().cpu() for k, v in actor_state.items()}, ckp)
            
            # 使用 lm_checkpoint 保存完整状态（包括 critic）
            lm_checkpoint(lm_config, weight=args.save_weight, model=actor_model, optimizer=actor_optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints',
                         scheduler=actor_scheduler, critic_model=critic_model, 
                         critic_optimizer=critic_optimizer, critic_scheduler=critic_scheduler)
            actor_model.train()
            del actor_state

        del enc, gen_out, completion_ids, responses_text, rewards, full_mask, values_seq, advantages
        del labels, resp_labels, resp_idx, resp_pad_mask, valid_resp, eos_mask, has_eos, eos_pos, resp_lengths, resp_policy_mask, resp_value_mask, old_resp_logp, ref_resp_logp
        del kl, kl_ref, policy_loss, value_loss, loss, token_rewards, returns, old_resp_values, prompt_lens, logp_pos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind PPO (Proximal Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='ppo_actor', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="Actor学习率")
    parser.add_argument("--critic_learning_rate", type=float, default=5e-7, help="Critic学习率")
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
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO裁剪参数")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function系数")
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL散度惩罚系数")
    parser.add_argument("--gamma", type=float, default=1.0, help="GAE折扣因子")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda参数")
    parser.add_argument("--cliprange_value", type=float, default=0.2, help="Value function裁剪范围")
    parser.add_argument("--ppo_update_iters", type=int, default=2, help="同一批rollout重复更新次数")
    parser.add_argument("--early_stop_kl", type=float, default=0.25, help="PPO early stop 的 KL 阈值")
    parser.add_argument("--mini_batch_size", type=int, default=2, help="PPO每次更新的minibatch大小")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-PPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug模式下每隔多少step打印一次采样")
    parser.add_argument("--debug_log_ratio", action="store_true", help="打印首轮首个minibatch的log_ratio差异量级，用于核查ratio≈1是否成立")
    parser.add_argument("--thinking_ratio", type=float, default=0.9, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_ppo", help="SGLang共享存储路径")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
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
        wandb_run_name = f"MiniMind-PPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 初始化模型和数据 ==========
    base_weight = args.from_weight
    # Actor模型
    actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    moe_suffix = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/{base_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    # Critic 主干加载与 Actor 相同的 SFT 权重；权重文件里没有 value_head，strict=False 让它保持随机初始化。
    critic_model = CriticModel(lm_config)
    critic_model.load_state_dict(state_dict, strict=False)
    critic_model = critic_model.to(args.device)
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    # Rollout引擎
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=actor_model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len), thinking_ratio=args.thinking_ratio)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    # 调度器的总步数 = 每个 batch 的 optimizer 更新次数（轮数 × minibatch 数 / 累积步数）× batch 数 × epoch 数。
    mb_factor = max(1, math.ceil(args.batch_size / args.mini_batch_size))
    total_optimizer_steps = math.ceil(iters * args.epochs * args.ppo_update_iters * mb_factor / args.accumulation_steps)
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_optimizer_steps, eta_min=args.critic_learning_rate / 10)

    start_epoch, start_step = 0, 0
    if ckp_data:
        actor_model.load_state_dict(ckp_data['model'])
        critic_model.load_state_dict(ckp_data['critic_model'])
        actor_optimizer.load_state_dict(ckp_data['optimizer'])
        critic_optimizer.load_state_dict(ckp_data['critic_optimizer'])
        actor_scheduler.load_state_dict(ckp_data['scheduler'])
        critic_scheduler.load_state_dict(ckp_data['critic_scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        actor_model = torch.compile(actor_model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        # freqs_cos/freqs_sin 各 rank 由 config 确定性算出，默认每步广播一次纯属浪费
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank], broadcast_buffers=False)
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank], broadcast_buffers=False)
    rollout_engine.update_policy(actor_model)
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            ppo_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, actor_scheduler, critic_scheduler, reward_model, start_step, wandb)
        else:
            ppo_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, actor_scheduler, critic_scheduler, reward_model, 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
