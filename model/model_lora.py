"""【阅读路线 第 5 站（下）】LoRA 适配器的实现与管理：插入、加载、保存以及将增量合并回基础权重。

先回答“为什么需要 LoRA”：全参数微调要为每个参数保存梯度和 AdamW 的 m、v 状态，
显存开销约为参数量的 4 倍以上（参数 + 梯度 + m + v），而且每个下游任务都要存一份完整模型。

LoRA（Low-Rank Adaptation，低秩适配）的假设：微调带来的权重变化 ΔW 是低秩的。
于是冻结原始权重 W（d_out×d_in），只训练两个小矩阵 B（d_out×r）和 A（r×d_in），r ≪ d：

    y = W·x + ΔW·x = W·x + B·(A·x)

可训练参数从 d_out·d_in 降到 r·(d_in+d_out)。冻结的 W 不需要梯度和优化器状态，
显存大幅下降；每个任务只需保存很小的 A、B（adapter，适配器），可按需挂载/切换。
合并（merge）则把 B·A 直接加进 W，导出成与原结构完全相同、推理无额外开销的普通模型。

阅读顺序：LoRA（模块本身）→ apply_lora（如何挂到模型上）→ trainer/train_lora.py（如何只训练它）
→ save_lora / load_lora / merge_lora（如何保存、加载与合并）。
"""
import torch
from torch import optim, nn


# 定义Lora网络结构
class LoRA(nn.Module):
    """可训练的低秩权重增量 ΔW = B×A；rank 控制增量子空间的维度。

    本实现是最简形式：没有原论文中的缩放系数 alpha/r，也没有 LoRA dropout；
    增量的有效幅度完全由学习率与训练过程决定。
    """
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A：降维 d_in → r
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B：升维 r → d_out
        # 初始化是 LoRA 的关键细节：A 随机、B 全零 ⇒ 训练开始时 ΔW = B·A = 0，
        # 挂上 LoRA 的模型输出与原模型完全一致，微调从“原模型”平滑起步。
        # 不能两者都为 0：那样 A、B 的梯度都恒为 0（互为对方梯度的乘数），永远学不动。
        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        # 先降维再升维：计算量 O(r·(d_in+d_out))，从不显式构造 d_out×d_in 的 ΔW。
        return self.B(self.A(x))


def apply_lora(model, rank=16):
    """在目标线性层上挂接 LoRA 分支。

    注意：这里只负责“挂接”，冻结基础参数是在 train_lora.py 中按参数名完成的。

    目标层的选择规则是 in_features == out_features（方阵线性层）。以默认配置
    （H=768、8 个 Q 头、4 个 KV 头、head_dim=96）为例，只有 q_proj（768→768）和
    o_proj（768→768）满足条件；k_proj/v_proj（768→384）、FFN 的 gate/up/down、lm_head 都不会挂 LoRA。
    主流实现（如 PEFT）通常按模块名显式指定 target_modules，也常覆盖 k/v 与 FFN。
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            lora = LoRA(module.in_features, module.out_features, rank=rank).to(model.device)
            # setattr 把 LoRA 注册为该 Linear 的子模块：参数名变成 “...q_proj.lora.A.weight”，
            # 因此后续可以用 'lora' in name 来筛选/冻结参数，state_dict 里也会带上这些键。
            setattr(module, "lora", lora)
            original_forward = module.forward

            # monkey-patch：用新函数替换该实例的 forward，实现 y = W·x + B·A·x。
            # 用默认参数 layer1/layer2 “显式绑定”当前循环中的对象；若直接在闭包里引用
            # original_forward/lora，Python 闭包按变量名延迟查找，所有层最终都会指向最后一次循环的对象。
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora


def load_lora(model, path):
    # 前提：已对 model 调用过 apply_lora（模块上存在 .lora）。
    # 去掉 DDP 保存时可能带上的 "module." 前缀，再把 “{层名}.lora.xxx” 的参数分发给各层的 LoRA 子模块。
    state_dict = torch.load(path, map_location=model.device)
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    # 只保存 LoRA 的 A/B 参数（fp16），文件通常只有完整模型的百分之几。
    # 推理时需要“基础权重 + 这个 adapter”一起加载（见 eval_llm.py 的 --lora_weight）。
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)


def merge_lora(model, lora_path, save_path):
    """把已训练增量折叠进基础模型权重并保存合并结果。

    因为 W·x + B·(A·x) = (W + B·A)·x，所以令 W' = W + B·A 后，普通 Linear 的输出与“基础层+LoRA 分支”完全相同。
    合并后的 state_dict 不含任何 .lora. 键，可以直接用原始 MiniMindForCausalLM 以 strict=True 加载。
    """
    load_lora(model, lora_path)
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if '.lora.' not in k}
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and '.lora.' not in name:
            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()
            if hasattr(module, 'lora'):
                # nn.Linear 的 weight 形状是 [out, in]：B.weight=[d_out, r]，A.weight=[r, d_in]，
                # 二者相乘恰为 [d_out, d_in]，与基础权重同形状，可以直接相加。
                state_dict[f'{name}.weight'] += (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()
    torch.save(state_dict, save_path)
