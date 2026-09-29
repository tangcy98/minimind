"""【阅读路线 第 7 站·入口】强化学习 rollout（轨迹采样）引擎，统一 PyTorch 与 SGLang 的生成接口。

在线 RL 的每一步都分成两个阶段，这是理解 PPO/GRPO/Agent 三个脚本的总框架：
1. 采样阶段（rollout，无梯度）：用当前策略为每个 prompt 生成回答，并记录
   生成的 token、有效位置 mask、以及“采样当时”策略给每个 token 的 log 概率（old log-prob）；
   然后用奖励函数/奖励模型给每条回答打分。
2. 学习阶段（有梯度）：把“prompt + 回答”重新送进策略模型前向一遍，得到“当前”策略的 log-prob，
   与 old log-prob、reference log-prob、奖励（优势）组合成策略损失，再反向更新。

为什么要在采样之后再前向一遍：generate 在 inference_mode 下逐 token 进行，没有保留计算图，
无法反向传播；训练时需要对整条序列做一次“teacher forcing”式的并行前向来建立计算图。

在强化学习的语言里：policy（策略）π 就是语言模型；state（状态）s_t 是 prompt + 已生成的前 t 个 token；
action（动作）a_t 是下一个生成的 token；一条回答就是一条 trajectory（轨迹）。

本文件把“采样”抽象成可替换的后端：TorchRolloutEngine 直接调用训练中的模型 generate；
SGLangRolloutEngine 通过 HTTP 调用独立的高吞吐推理服务（生成是 RL 中最耗时的部分）。

# 如果使用sglang加速，需通过以下命令首先启动（transformers格式）模型：
python -m sglang.launch_server --model-path ./minimind-3 --attention-backend triton --host 0.0.0.0 --port 8998
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests
import torch
import torch.distributed as dist
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Tuple
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer


# ===== 计算每个 token 的 logprob =====
def compute_per_token_logps(model, input_ids: Tensor, n_keep: int, attention_mask: Optional[Tensor] = None) -> Tensor:
    # 计算序列最后 n_keep 个 token 各自的 log π(a_t|s_t)：状态 s_t 下生成 token 动作 a_t 的对数概率。
    # 返回 [B, n_keep]。新旧策略的这个量之差（exp 后即概率比）衡量策略改变幅度。
    if n_keep <= 0:
        return input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    # generate 在 inference_mode 下产生的张量是“推理张量”，不能参与后续需要 autograd 的计算，clone 一份普通张量。
    input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
    # 错位取值：要得到最后 n_keep 个 token 的概率，需要它们“前一个位置”的 logits。
    # logits_to_keep=n_keep+1 只投影最后 n_keep+1 个位置到词表（省显存），再去掉最后一个位置
    # （它预测的是序列之外的 token），剩下的 n_keep 个位置恰好依次预测 input_ids[:, -n_keep:]。
    logits = unwrapped(input_ids, attention_mask=attention_mask, logits_to_keep=n_keep + 1).logits[:, :-1, :]
    per_token_logps = []
    # 逐行做 log_softmax + gather（而非整批一次做），避免同时物化整批 [B, n_keep, V] 的 log 概率张量，降低显存峰值。
    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
        ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
        per_token_logps.append(
            torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1)
        )
    return torch.stack(per_token_logps)


# ===== Rollout 结果 =====
@dataclass
class RolloutResult:
    """一批轨迹的标准结果：全序列、回答片段、采样概率、文本和有效长度信息。

    记 N = prompt 数 × num_generations（每个 prompt 被重复采样的次数），R = 最长回答长度。
    """
    output_ids: Tensor          # [N, P+R] prompt + 回答的完整 token 序列（学习阶段重新前向用它）
    completion_ids: Tensor      # [N, R]   只含回答部分
    per_token_logps: Tensor     # [N, R]   采样时策略对每个回答 token 的 log 概率（old log-prob）
    completions: List[str]      # N 条回答文本（去掉特殊 token），交给奖励函数打分
    prompt_lens: Tensor         # [N]      每条序列中 prompt 的长度，用来定位回答在 output_ids 中的起点
    completion_mask: Tensor     # [N, R]   1=有效回答 token，0=补齐（SGLang 后端各回答长度不同时才有 0）


# ===== Rollout 引擎抽象基类 =====
class RolloutEngine(ABC):
    """抽象采样后端；不同引擎必须返回兼容的 RolloutResult。"""
    tokenizer = None
    
    @abstractmethod
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        pass
    
    @abstractmethod
    def update_policy(self, model: torch.nn.Module):
        pass


# ===== PyTorch 原生推理引擎 =====
class TorchRolloutEngine(RolloutEngine):
    def __init__(self, policy_model: torch.nn.Module, tokenizer, device: str = "cuda", autocast_ctx=None):
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.device = device
        self.autocast_ctx = autocast_ctx
    
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        # completion 是模型针对 prompt 新生成的回答。repeat_interleave 把每个 prompt 连续复制 num_generations 份
        # （[p0,p0,p1,p1,...] 的顺序），对应 GRPO 的“组”；后续按 view(-1, num_generations) 还原分组。
        # 同时缓存行为策略（采样当时的 policy）的 log-prob，之后计算 importance ratio。
        # 取 .module 绕过 DDP 包装：generate 不需要梯度同步，直接调用底层模型。
        model = self.policy_model.module if isinstance(self.policy_model, DistributedDataParallel) else self.policy_model
        ctx = self.autocast_ctx if self.autocast_ctx else nullcontext()
        with torch.no_grad(), ctx:
            output_ids = model.generate(
                input_ids=prompt_ids.repeat_interleave(num_generations, dim=0),
                attention_mask=attention_mask.repeat_interleave(num_generations, dim=0),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            ).clone()  # [B*num_gen, P+R]
            # prompt 已左侧补齐到同一长度，所以回答统一从第 prompt_len 列开始。
            prompt_len = prompt_ids.size(1)
            completion_ids = output_ids[:, prompt_len:]  # [B*num_gen, R]
            full_mask = torch.cat([attention_mask.repeat_interleave(num_generations, dim=0), attention_mask.new_ones(output_ids.size(0), completion_ids.size(1))], dim=1)
            # old log-prob 用一次独立前向计算，且是“原始分布”（温度=1、不做 top-k/top-p 截断）下的概率，
            # 而不是 generate 实际采样时那个经过温度和截断的分布。这样它与学习阶段计算当前 log-prob 的
            # 口径完全一致，二者之比才是有意义的“策略变化量”（严格意义的行为策略概率在这里被近似了）。
            per_token_logps = compute_per_token_logps(self.policy_model, output_ids, completion_ids.size(1), attention_mask=full_mask)
        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        # completion_mask 全 1：generate 对已结束的序列持续填充 EOS 而不是 PAD，这里不区分，
        # 由训练脚本根据“第一个 EOS 的位置”自行截断有效回答长度。
        return RolloutResult(output_ids, completion_ids, per_token_logps, completions,
                             prompt_ids.new_full((output_ids.size(0),), prompt_len),
                             attention_mask.new_ones(output_ids.size(0), completion_ids.size(1)))

    def update_policy(self, model: torch.nn.Module):
        # Torch 后端直接持有训练中模型的引用，参数一更新，下次采样自然就用新策略，这里只是更新引用。
        # 因此在该后端下，采样策略永远等于“当前”策略（严格 on-policy）。
        self.policy_model = model


# ===== SGLang HTTP API 推理引擎 =====
class SGLangRolloutEngine(RolloutEngine):
    def __init__(self, base_url: str, model_path: str, shared_ckpt_path: str = "./sglang_ckpt", timeout: int = 120):
        self.base_url = base_url.rstrip('/')
        self.shared_ckpt_path = shared_ckpt_path
        self.timeout = timeout
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.http = requests
    
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        # 与 Torch 后端的区别：SGLang 服务端自己做连续批处理（continuous batching）和 KV cache 管理，
        # 输入只需要不带 padding 的 token 列表；old log-prob 直接由服务端在采样时返回（return_logprob），
        # 省去训练侧额外的一次前向。返回后再右侧补齐成张量，并用 prompt_lens / completion_mask 记录真实长度。
        # 去除左侧 padding tokens，只保留有效 token
        input_ids_list = []
        for ids, mask in zip(prompt_ids, attention_mask):
            valid_ids = ids[mask.bool()].tolist()
            input_ids_list.append(valid_ids)
        all_input_ids = [ids for ids in input_ids_list for _ in range(num_generations)]
        
        payload = {
            "input_ids": all_input_ids,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "stop_token_ids": [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else [],
            },
            "return_logprob": True,
        }
        
        resp = self.http.post(f"{self.base_url}/generate", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        
        results = resp.json()
        if not isinstance(results, list):
            results = [results]
        
        all_output_ids, all_completion_ids, all_logprobs = [], [], []
        completions = []
        
        for i, result in enumerate(results):
            meta = result.get("meta_info", {})
            completion_ids = meta.get("output_ids", result.get("output_ids", []))
            raw_logprobs = meta.get("output_token_logprobs", [])
            
            logprobs = []
            for item in raw_logprobs:
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    logprobs.append(item[0])
                elif isinstance(item, (int, float)):
                    logprobs.append(item)
            
            if len(logprobs) < len(completion_ids):
                logprobs = [0.0] * (len(completion_ids) - len(logprobs)) + logprobs
            elif len(logprobs) > len(completion_ids):
                logprobs = logprobs[-len(completion_ids):] if completion_ids else []
            prompt = all_input_ids[i]
            full_output = prompt + completion_ids
            all_output_ids.append(full_output)
            all_completion_ids.append(completion_ids)
            all_logprobs.append(logprobs)
            completions.append(self.tokenizer.decode(completion_ids, skip_special_tokens=True))
        
        device = prompt_ids.device
        max_comp_len = max(1, max(len(ids) for ids in all_completion_ids))
        max_out_len = max(len(ids) for ids in all_input_ids) + max_comp_len
        
        def pad_to_tensor(seqs, max_len, pad_val=0):
            return torch.tensor([s + [pad_val] * (max_len - len(s)) for s in seqs], device=device)
        
        pad_id = self.tokenizer.pad_token_id
        return RolloutResult(
            output_ids=pad_to_tensor(all_output_ids, max_out_len, pad_val=pad_id),
            completion_ids=pad_to_tensor(all_completion_ids, max_comp_len, pad_val=pad_id),
            per_token_logps=pad_to_tensor(all_logprobs, max_comp_len, pad_val=0.0),
            completions=completions,
            prompt_lens=torch.tensor([len(ids) for ids in all_input_ids], device=device),
            completion_mask=torch.tensor([[1] * len(ids) + [0] * (max_comp_len - len(ids)) for ids in all_completion_ids], device=device),
        )
    
    def update_policy(self, model: torch.nn.Module):
        # 外部推理服务持有独立权重副本；训练侧把最新 policy 写到共享目录，
        # 再通知 SGLang reload，保证下一轮 rollout 使用更新后的策略。
        # 训练脚本只在每 save_interval 步调用一次，两次同步之间 SGLang 用的是“旧”权重采样：
        # 这时采样策略与当前策略不再相同（off-policy 程度随间隔增大），old log-prob 与新 log-prob 的比值
        # 不再恒为 1，PPO clip / CISPO 截断这类“重要性采样修正”就真正发挥作用了。
        # 另外推理引擎与训练框架的数值实现（kernel、精度）不同，即便权重相同，两边的 log-prob 也会有微小差异。
        ok = True
        if not dist.is_initialized() or dist.get_rank() == 0:
            try:
                unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
                unwrapped = getattr(unwrapped, '_orig_mod', unwrapped)
                abs_path = os.path.abspath(self.shared_ckpt_path)
                state_dict = {k: v.detach().half().cpu() for k, v in unwrapped.state_dict().items()}
                unwrapped.save_pretrained(abs_path, state_dict=state_dict, safe_serialization=False)
                self.tokenizer.save_pretrained(abs_path)
                resp = self.http.post(f"{self.base_url}/update_weights_from_disk", json={"model_path": abs_path}, timeout=self.timeout)
                if resp.status_code != 200: print(f"[SGLANG WARNING] update_weights 失败: {resp.status_code}, {resp.text}")
                ok = resp.status_code == 200
            except Exception as e:
                print(f"[SGLANG WARNING] update_weights 异常: {e}"); ok = False
        # 只有 rank 0 写文件并通知服务端；把结果广播给所有 rank，并用 barrier 让其他 rank 等它完成，
        # 保证所有进程对“是否成功”的判断一致，避免部分进程报错退出、其余进程在集合通信中永久等待。
        if dist.is_initialized():
            ok_t = torch.tensor(int(ok), device=next(model.parameters()).device)
            dist.broadcast(ok_t, src=0); dist.barrier(); ok = bool(ok_t.item())
        if not ok: raise RuntimeError("SGLang update_policy failed")
        return ok
    
    def flush_cache(self) -> bool:
        resp = self.http.post(f"{self.base_url}/flush_cache", timeout=30)
        return resp.status_code == 200
    
    def health(self) -> bool:
        try:
            resp = self.http.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except:
            return False


# ===== 工厂函数 =====
def create_rollout_engine(
    engine_type: str = "torch",
    policy_model: torch.nn.Module = None,
    tokenizer = None,
    device: str = "cuda",
    autocast_ctx = None,
    sglang_base_url: str = None,
    sglang_model_path: str = None,
    sglang_shared_path: str = None,
) -> RolloutEngine:
    if engine_type == "torch":
        return TorchRolloutEngine(policy_model, tokenizer, device, autocast_ctx)
    elif engine_type == "sglang":
        return SGLangRolloutEngine(sglang_base_url, sglang_model_path, sglang_shared_path)
    else:
        raise ValueError(f"不支持的引擎类型: {engine_type}")
