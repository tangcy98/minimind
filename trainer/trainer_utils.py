"""【阅读路线 第 4 站·配套】训练公共工具：学习率调度、分布式初始化、模型加载和 checkpoint 管理。

建议在读 train_pretrain.py 的过程中按需跳转到这里，而不是从头读：
- get_lr：学习率调度（每个 step 的学习率怎么变）；
- init_distributed_mode / is_main_process：多卡 DDP 的进程组与“只让 0 号进程打印/保存”；
- setup_seed：可复现性；
- init_model：构造模型 + 加载上一阶段权重（预训练 → SFT → RL 的衔接点）；
- lm_checkpoint / SkipBatchSampler：断点续训（保存什么、如何跳过已训练的 batch）；
- LMForRewardModel：第 7 站 RL 用的奖励模型封装；safe_math_eval：Agent 工具的安全计算器。

Checkpoint（检查点）是训练状态快照：纯模型权重可以用于推理；resume checkpoint
还包含优化器、训练进度等状态，方便从中断处继续。
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ast
import operator
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from model.model_minimind import MiniMindForCausalLM

def get_model_params(model, config):
    # 打印总参数量；MoE 模型还会打印“激活参数量”（每个 token 实际参与计算的参数，A 后面的数字）：
    # 激活量 = 非专家部分 + 单个专家参数 × 每 token 选中的专家数。
    # 注意 model.parameters() 对绑定的 embedding/lm_head 只计一次。
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')


def is_main_process():
    # DDP 下每张卡一个进程，每个进程都跑同一份脚本。rank 是全局进程编号；
    # 打印日志、保存 checkpoint 这类“只需做一次”的事情只让 rank 0 做，避免重复输出和写文件冲突。
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    # 余弦退火学习率：lr_t = lr · (0.1 + 0.45·(1 + cos(π·t/T)))
    #   t=0 时 cos=1 → lr_t = lr（峰值）；t=T 时 cos=-1 → lr_t = 0.1·lr（下限，保留 10%）。
    # 为什么要衰减：训练早期用大步长快速下降，后期用小步长在低 loss 区域精细收敛，减少震荡。
    # 注意两点：
    # 1) 这里没有 warmup（常见做法是前几百步从 0 线性升到峰值，避免训练初期 Adam 二阶矩估计
    #    不准时步子过大）；MiniMind 模型小、从较小学习率起步，省略了这一段。
    # 2) 调用方按“micro-batch 步数”传入 current_step，而不是按 optimizer 更新次数；
    #    梯度累积时一个更新窗口内会多次改写 lr，真正生效的是窗口最后一个 micro-step 设置的值。
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    """根据 torchrun 注入的环境变量初始化 DDP 通信组，返回当前 GPU 序号。

    用 `torchrun --nproc_per_node N train_xxx.py` 启动时，torchrun 会拉起 N 个进程，并为每个进程
    设置环境变量：RANK（全局进程号）、LOCAL_RANK（本机内的进程号，即要用的 GPU 编号）、
    WORLD_SIZE（总进程数）以及 MASTER_ADDR/PORT（用于互相发现）。直接 `python train_xxx.py`
    启动时没有 RANK，走单卡模式。
    """
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    # NCCL 是 NVIDIA 的 GPU 间集合通信库，负责后面 DDP 反向传播时的梯度 all-reduce。
    # init_process_group 会阻塞直到所有进程都连上，之后才能进行集合通信。
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    # 固定 Python/NumPy/PyTorch(CPU+GPU) 的随机数种子，影响参数初始化、数据打乱、dropout、采样等。
    # cudnn.deterministic=True + benchmark=False 让 cuDNN 选择确定性算法，牺牲少量速度换取可复现。
    # （部分 CUDA 算子如 index_add_ 的原子加法仍可能有微小的非确定性。）
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    # 两用函数：传入 model 时是“保存模式”，不传 model 时是“加载模式”（返回 resume 数据或 None）。
    #
    # 保存两个文件：
    # - {weight}_{H}.pth：只有模型参数（fp16），体积小，用于推理或作为下一阶段的起点；
    # - {weight}_{H}_resume.pth：参数 + 优化器状态 + epoch/step + 其他（scaler、scheduler、critic 等），
    #   用于断点续训。为什么只存参数不够：AdamW 为每个参数维护一阶矩 m 与二阶矩 v（合计约为参数量的 2 倍，
    #   且是 fp32），丢了它们，恢复后的前若干步相当于“冷启动”优化器，训练轨迹会与原来不同。
    #
    # 写入先写 .tmp 再 os.replace：os.replace 是原子操作，保存中途被杀掉也不会留下半截损坏的 checkpoint。
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        # “剥壳”：DDP 包装后真实模型在 .module 下，torch.compile 包装后在 ._orig_mod 下。
        # 直接对包装器取 state_dict 会得到带 "module." / "_orig_mod." 前缀的键名，换个环境就加载不上。
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        # 转 fp16 存储以减半体积；训练中的参数本身仍是 fp32（混合精度只影响计算，不影响参数的主副本）。
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            # step 记录的是“每张卡处理了多少个 batch”。DDP 下每步全局消费 world_size 个 batch，
            # 若续训时卡数变了，按“已消费样本数不变”换算：新 step = 旧 step × 旧卡数 / 新卡数。
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):
    # 统一模型/Tokenizer 初始化入口，也是各训练阶段的“接力点”：
    # 预训练 from_weight='none'（随机初始化）→ 产出 pretrain_768.pth；
    # SFT from_weight='pretrain' → 产出 full_sft_768.pth；DPO/RL/LoRA 再从 full_sft 出发……
    # 文件名约定为 {save_dir}/{阶段名}_{hidden_size}[_moe].pth。
    #
    # strict=False：键名对不上时不报错，缺失的参数保持随机初始化、多余的键被忽略。
    # 这让结构有少量差异的权重也能加载（例如 PPO 的 Critic 多一个 value_head），
    # 但也意味着“权重文件放错/结构配错”时不会报错，只会悄悄训练一个部分随机的模型——
    # 排查“loss 异常高”时，先确认这里是否真的加载上了权重。
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    if from_weight!= 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    """跳过已完成的 batch，配合 resume checkpoint 恢复 epoch 内训练进度。

    sampler 决定样本下标的顺序（单卡时是 randperm 的结果，DDP 时是 DistributedSampler 分给本卡的子集），
    本类把下标按 batch_size 分组，并丢弃前 skip_batches 组。只要续训时用同一个随机种子得到
    同样的下标顺序，就能精确从中断的那个 batch 继续，而不会重复训练或漏掉数据。
    """
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


class LMForRewardModel:
    """奖励模型封装：把对话上下文和候选回答映射为一个标量偏好分数。"""
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def get_score(self, messages, response):
        # 奖励模型（Reward Model，RM）通常是用人类偏好对训练出来的打分模型：输入“对话 + 回答”，
        # 输出一个标量，分数越高代表越符合人类偏好。它本身不生成文字。
        # PPO/GRPO 用这个分数构造策略更新信号，而不是直接拟合 token 标签。
        # 这里把多轮历史压成单个 user 消息（以适配外部 RM 的输入格式），调用 RM 自带的 get_score，
        # 最后把分数裁剪到 [-3, 3]：防止个别极端分数主导优势估计，也限制“奖励黑客”带来的收益。
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
        last_query = messages[-1]['content'] if messages else ""
        message_context = f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}" if history_text else last_query
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response}
        ]
        score = self.model.get_score(self.tokenizer, eval_messages)
        return max(min(score, 3.0), -3.0)


# ===== 数学表达式安全求值：替代 eval，只放行算术运算与 math 白名单（长度上限 512） =====
def safe_math_eval(expression):
    """对模型生成的数学表达式求值：支持 + - * / // % **、math 白名单函数与 pi/e/tau 常量。"""
    def pow_guard(base, exp):  # 幂运算统一走这里，拦截 9**9**9 / 10**99999 这类把进程算死的输入
        if abs(exp) > 1e4 or (abs(base) > 1 and abs(exp) * math.log10(abs(base)) > 100): raise ValueError('幂运算结果过大')
        return base ** exp
    def resolve(node):  # 同时兼容 sqrt(4) 与 math.sqrt(4) 两种写法
        if isinstance(node, ast.Name): return node.id
        if isinstance(node, ast.Attribute) and getattr(node.value, 'id', '') == 'math': return node.attr
    def walk(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float): return node.value
        name = resolve(node)
        if name in consts: return consts[name]
        if isinstance(node, ast.UnaryOp): return unary_ops[type(node.op)](walk(node.operand))
        if isinstance(node, ast.BinOp): return bin_ops[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.Call) and not node.keywords: return funcs[resolve(node.func)](*map(walk, node.args))
        raise ValueError(f'不支持的表达式语法: {type(node).__name__}')
    consts = {'pi': math.pi, 'e': math.e, 'tau': math.tau}
    funcs = {'pow': pow_guard, **{n: getattr(math, n) for n in 'sqrt exp log log2 log10 sin cos tan asin acos atan atan2 sinh cosh tanh floor ceil trunc fabs fmod hypot gcd degrees radians'.split()}}
    bin_ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: pow_guard}
    unary_ops = {ast.UAdd: operator.pos, ast.USub: operator.neg}
    chars = str.maketrans({'^': '**', '×': '*', '÷': '/', '−': '-', '²': '**2', '³': '**3', '（': '(', '）': ')'})
    expr = str(expression).translate(chars).strip()
    if not expr or len(expr) > 512: raise ValueError('表达式为空或过长')
    try:
        return walk(ast.parse(expr, mode='eval').body)
    except (KeyError, SyntaxError, TypeError):
        raise ValueError('不支持的表达式语法') from None
