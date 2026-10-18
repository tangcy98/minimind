"""【阅读路线 第 2 站】MiniMind 的 Decoder-only 因果语言模型实现。

读这个文件之前，你应该已经知道：tokenizer 把字符串变成整数 id（第 1 站）。
读完这个文件，你应该能回答：一串 id 进入模型后，经过哪些计算，最终变成
“下一个 token 的概率分布”，以及训练 loss 是怎么从这个分布算出来的。

几个基本概念：
- Decoder-only：只用 Transformer 解码器堆栈、不含独立编码器（GPT/LLaMA/Qwen 都是这一类）。
- 因果（causal）/ 自回归（autoregressive）：位置 t 只能看到 0..t 的 token，
  模型的任务永远是“根据前文预测下一个 token”；生成时把预测出的 token 接回末尾再预测。

整体数据流与张量形状（B=batch 大小，S=序列长度，H=hidden_size，V=词表大小）：

    input_ids [B,S] ──Embedding 查表──▶ hidden [B,S,H]
        ──▶ N × MiniMindBlock（Attention 跨 token 混合信息 + FFN/MoE 逐 token 变换特征）
        ──▶ 最终 RMSNorm ──▶ lm_head 线性投影 ──▶ logits [B,S,V]
        ──▶ 训练：与“右边一位”的 labels 做交叉熵；生成：取最后一个位置采样下一个 token

全文代码注释中的形状符号约定（“A[..] → B[..]”表示该行输入 → 输出）：
    B      batch 大小                      S      本次送入模型的序列长度（训练=整句；增量解码=1）
    P      KV cache 已缓存的历史长度       S_k    注意力的 key 总长度 = P + S（无缓存时 P=0，S_k=S）
    H      hidden_size                     V      vocab_size
    nQ     num_attention_heads（Q 头数）   nKV    num_key_value_heads（K/V 头数）
    D      head_dim（每头维度）            r      n_rep = nQ / nKV（GQA 复制倍数）
    I      intermediate_size（FFN 中间宽） L      num_hidden_layers
    E      num_experts（MoE 专家数）       k      num_experts_per_tok（每 token 选几个专家）
    T      MoE 中摊平后的 token 数 = B·S   n_i    被路由到第 i 个专家的 token 数
    M      max_position_embeddings（RoPE 表长度）
    S'     lm_head 实际投影的位置数（logits_to_keep=0 时 S'=S，否则 S'=logits_to_keep）
    generate 中：B = 原 batch × num_return_sequences；S_cur = 当前已有序列总长；top_k 为采样截断参数
    "[]" 表示 0 维标量；nn.Linear(in,out) 的权重形状是 [out,in]。

建议阅读顺序：MiniMindConfig → RMSNorm → precompute_freqs_cis / apply_rotary_pos_emb
→ Attention → FeedForward / MOEFeedForward → MiniMindBlock → MiniMindModel
→ MiniMindForCausalLM.forward（训练用）→ MiniMindForCausalLM.generate（推理用）。
训练和生成共用同一套参数，区别只是 forward 是否传 labels、是否使用 KV cache。
"""
import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Config
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class MiniMindConfig(PretrainedConfig):
    """模型配置（configuration）：描述结构和行为的超参数，不是训练得到的权重。

    继承 transformers 的 PretrainedConfig 后，可以用 save_pretrained/from_pretrained
    读写 config.json；未显式声明的 kwargs 也会被父类保存为属性（例如 RL 脚本传入的
    max_seq_len），所以 config 也常被当作“杂项超参数容器”。

    结构超参数（决定参数张量形状，必须与 checkpoint 一致，否则权重加载不上）：
    - vocab_size=V：词表大小，决定 embedding 表 [V,H] 与 lm_head [H,V] 的尺寸，
      必须和 tokenizer 的词表完全对应。
    - hidden_size=H：每个 token 的向量宽度（残差流宽度）。
    - num_hidden_layers：Transformer block 堆叠层数，决定模型“深度”。
    - num_attention_heads：Query 头数；num_key_value_heads：Key/Value 头数。
      二者不相等时就是 GQA（见 Attention）。head_dim 默认 H / Q头数。
    - intermediate_size：FFN 中间层宽度。默认取 ≈π·H 并向上取整到 64 的倍数；
      SwiGLU 有三个矩阵，所以常用 ~8/3·H 到 ~4·H 之间的宽度，64 对齐利于 GPU 矩阵乘效率。
    - tie_word_embeddings：输入 embedding 与输出 lm_head 共用同一个 [V,H] 矩阵
      （权重绑定），小模型里词表参数占比高，这样能省下可观的参数量。

    行为超参数（不改变参数形状）：
    - rope_theta：RoPE 的频率基数 base，越大则低频维度波长越长，越适合长上下文。
    - max_position_embeddings：预计算 RoPE cos/sin 表的长度上限。
    - inference_rope_scaling：推理时开启 YaRN 长度外推（见 precompute_freqs_cis）。
    - rms_norm_eps：RMSNorm 分母上的小常数，防止除零。
    - flash_attn：条件允许时使用 PyTorch 融合注意力内核。
    - dropout：训练时随机置零比例；大模型预训练通常设为 0。

    MoE 超参数（仅 use_moe=True 时生效）：
    - num_experts：专家 FFN 的个数；num_experts_per_tok：每个 token 激活 top-k 个专家。
    - norm_topk_prob：是否把选中专家的权重重新归一化为和为 1。
    - router_aux_loss_coef：负载均衡辅助损失的系数。
    """
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Model
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class RMSNorm(torch.nn.Module):
    """RMSNorm（均方根归一化）：y = x / sqrt(mean(x²) + eps) * weight。

    为什么需要归一化：残差结构里每层都往 hidden state 上“加东西”，不做归一化时
    各层输入的数值尺度会随深度漂移，训练容易发散。RMSNorm 在最后一维（H 维）上
    把每个 token 的向量缩放到均方根≈1，再乘可学习的逐维缩放 weight（初始化为 1）。
    与 LayerNorm 相比，它不减均值、也没有 bias，计算更省，效果在 LLM 中基本相当。
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # weight: [dim]

    def norm(self, x):
        # rsqrt = 1/sqrt；mean(-1) 只在特征维上求均值，不同 token 之间互不影响。
        # shape: x[...,dim] → x.pow(2)[...,dim] → mean(-1,keepdim)[...,1] → rsqrt[...,1]
        #        x[...,dim] * rsqrt[...,1]（沿最后一维广播）→ [...,dim]
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 先转 FP32 计算平方均值：bf16/fp16 的尾数位很少，平方后求和容易损失精度或溢出。
        # 最后 type_as 转回输入 dtype，以便后续算子继续走混合精度。
        # shape: x[...,dim] → norm(x.float())[...,dim]；weight[dim] * [...,dim] 广播 → [...,dim]
        #        （dim=H 时 [...]=[B,S]；dim=D 时 [...]=[B,S,heads]，即 QK-Norm 按头内维度归一化）
        return (self.weight * self.norm(x.float())).type_as(x)

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    # 为什么需要位置编码：注意力本身对 token 顺序“无感”（打乱输入顺序，QKᵀ 的集合不变），
    # 必须额外注入位置信息。RoPE（Rotary Position Embedding，旋转位置编码）不把位置向量
    # 加到 hidden state 上，而是把 Q/K 的每两个维度看成一个二维平面，按“位置 m × 频率 θ_i”
    # 旋转对应角度。两个旋转后向量的点积只取决于角度差 (m-n)·θ_i，
    # 于是注意力分数天然只依赖相对位置 m-n。
    #
    # 频率 θ_i = base^(-2i/dim)，i=0..dim/2-1：i 小的维度转得快（高频，分辨近距离），
    # i 大的维度转得慢（低频，编码远距离）。rope_base 越大，低频越慢，越利于长上下文。
    # 本函数一次性预计算 [end, dim] 的 cos/sin 表，forward 时只按位置切片，避免每层重复计算。
    #
    # YaRN（rope_scaling 非空时）：模型只在 original_max_position_embeddings 长度内训练过，
    # 直接用于更长序列时，低频维度会出现训练中从未见过的旋转角度。YaRN 对不同频率分段处理：
    # 高频维度（波长远小于训练长度）保持不变，低频维度的频率除以 factor（相当于位置插值），
    # 中间用线性 ramp 平滑过渡。attention_factor 乘在 cos/sin 上，等效调整注意力分数的尺度
    # （YaRN 论文中的温度补偿），本仓库默认 1.0 即不补偿。
    #
    # 本函数中 dim=D（head_dim），end=M（位置表长度）。
    # shape: arange(0,dim,2)[:dim//2] [D/2] → ... → freqs（逐维频率 θ_i）[D/2]；attn_factor 标量
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            # shape: inv_dim(b) 标量 → 标量（维度下标）；low/high 标量
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            # shape: arange(D/2)[D/2] - low（标量）→ / 标量 → clamp → ramp[D/2]
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            # shape: freqs[D/2] * (1 - ramp + ramp/factor)[D/2] → freqs[D/2]
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end, device=freqs.device)  # shape: t[M]（位置 0..M-1）
    # outer 得到角度矩阵 [end, dim/2]：第 m 行第 i 列是 m·θ_i。
    # 再把 cos/sin 在最后一维复制一份拼成 [end, dim]，与下面 rotate_half 的“前半/后半配对”布局对应。
    # shape: outer(t[M], freqs[D/2]) → freqs[M,D/2]
    freqs = torch.outer(t, freqs).float()
    # shape: cos(freqs)[M,D/2] 两份在最后一维 cat → [M,D]；* 标量 attn_factor → freqs_cos[M,D]
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    # shape: sin(freqs)[M,D/2] 两份 cat → [M,D]；* 标量 → freqs_sin[M,D]
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin  # 两者形状均为 [M,D]

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    # 二维旋转公式：(x1, x2) → (x1·cosθ - x2·sinθ, x2·cosθ + x1·sinθ)。
    # 这里把向量前半 x[:d/2] 与后半 x[d/2:] 的对应维度配成一对（HF/LLaMA 的 “half” 布局，
    # 而非相邻两维交错配对），rotate_half(x) = [-x2, x1]，于是 x·cos + rotate_half(x)·sin
    # 就是逐对旋转。q/k 形状 [B,S,heads,D]，cos/sin 形状 [S,D]；unsqueeze_dim=1 在 S 之后
    # 插入 heads 维使其可广播（所有头共享同一套旋转）。只旋转 Q/K，不旋转 V。
    # shape: rotate_half: x[...,D] → 后半取负 [...,D/2] 与前半 [...,D/2] 在最后一维 cat → [...,D]
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    # shape: q[B,S,nQ,D]；cos/sin[S,D] --unsqueeze(1)--> [S,1,D]（广播到 [B,S,nQ,D]）
    #        q*cos [B,S,nQ,D] + rotate_half(q)[B,S,nQ,D] * sin [B,S,nQ,D] → q_embed[B,S,nQ,D]
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    # shape: k[B,S,nKV,D]；cos/sin 同上广播 → k_embed[B,S,nKV,D]
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed  # q_embed[B,S,nQ,D], k_embed[B,S,nKV,D]

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # GQA 的“共享”在计算上的实现：把每个 K/V 头复制 n_rep 份，使 K/V 头数与 Q 头数对齐。
    # [B,S,kv_heads,D] → [B,S,kv_heads*n_rep,D]。expand 只是视图不复制内存，reshape 时才物化。
    # 注意 KV cache 里存的是复制前的 K/V，所以显存节省发生在 cache 上，而不是这一步的计算上。
    bs, slen, num_key_value_heads, head_dim = x.shape  # shape: x[B,S_k,nKV,D]（这里的 slen 即 S_k，含缓存）
    if n_rep == 1: return x  # [B,S_k,nKV,D] → 原样返回（nKV=nQ）
    # shape: x[B,S_k,nKV,D] → x[:,:,:,None,:][B,S_k,nKV,1,D] → expand[B,S_k,nKV,r,D]
    #        → reshape[B,S_k,nKV·r,D] = [B,S_k,nQ,D]
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))

class Attention(nn.Module):
    """多头因果自注意力：让每个 token 从可见的前序 token 汇集上下文信息。

    核心公式：Attention(Q,K,V) = softmax(QKᵀ / sqrt(D) + mask) · V
    - Query (Q)：当前位置“想找什么”；Key (K)：每个位置“能被怎样匹配”；
      Value (V)：匹配上之后真正被汇总过去的内容。三者都是 hidden state 的线性投影。
    - QKᵀ 给出每个 query 对每个 key 的相似度分数；除以 sqrt(D) 防止维度大时分数方差过大、
      softmax 过于尖锐导致梯度消失。softmax 后每行是一组和为 1 的权重，再对 V 加权求和。
    - 多头（multi-head）：把 H 维切成若干个 head_dim 维的子空间并行做注意力，不同头可以
      关注不同的关系（语法、指代、位置等），最后拼接并经 o_proj 投影回 H 维。
    - 因果 mask：位置 t 只能看到 ≤t 的 key，这是“预测下一个 token 不能偷看答案”的保证，
      也是训练时能一次性并行计算所有位置 loss 的前提（teacher forcing）。
    - GQA（Grouped Query Attention）：num_key_value_heads < num_attention_heads，
      每 n_rep 个 Q 头共享一组 K/V 头。推理时 KV cache 的显存与 K/V 头数成正比，
      因此 GQA 主要节省的是长序列生成时的缓存显存与带宽。
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)  # weight [nQ·D, H]
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)  # weight [nKV·D, H]
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)  # weight [nKV·D, H]
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)  # weight [H, nQ·D]
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # weight [D]
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # weight [D]
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # 输入 x=[B,S,H]：B=batch 样本数，S=本次输入的序列长度（增量解码时通常为 1），H=隐藏维度。
        # position_embeddings 是本段 token 对应位置的 (cos, sin)；past_key_value 是这一层的历史 K/V。
        bsz, seq_len, _ = x.shape  # shape: x[B,S,H] → bsz=B, seq_len=S
        # ① 线性投影得到 Q/K/V，再 view 成 [B,S,头数,head_dim]，让每个头独立计算相似度。
        #    K/V 的头数更少（GQA），所以 k_proj/v_proj 的输出维度比 q_proj 小。
        # shape: x[B,S,H] → q_proj → xq[B,S,nQ·D]；k_proj → xk[B,S,nKV·D]；v_proj → xv[B,S,nKV·D]
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)  # [B,S,nQ·D] → [B,S,nQ,D]
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)  # [B,S,nKV·D] → [B,S,nKV,D]
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)  # [B,S,nKV·D] → [B,S,nKV,D]
        # ② QK-Norm：在每个头内部对 Q、K 各做一次 RMSNorm（Qwen3 等模型的做法），
        #    限制 QKᵀ 分数的数值范围，防止训练中注意力 logits 过大导致 softmax 饱和、loss 尖峰。
        # shape: xq[B,S,nQ,D] → q_norm（沿 D 归一化）→ [B,S,nQ,D]；xk[B,S,nKV,D] → k_norm → [B,S,nKV,D]
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        # ③ RoPE：按位置旋转 Q/K，把位置信息编码进点积（V 不需要位置信息，所以不旋转）。
        cos, sin = position_embeddings  # shape: cos[S,D], sin[S,D]（已按 start_pos 切片）
        # shape: xq[B,S,nQ,D], xk[B,S,nKV,D], cos/sin[S,D] → xq[B,S,nQ,D], xk[B,S,nKV,D]
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        if past_key_value is not None:
            # ④ KV cache（键值缓存）：自回归生成时，历史 token 的 K/V 不会变（因果 mask 保证
            #    旧 token 看不到新 token），因此可以缓存下来。每步只为新 token 计算 Q/K/V，
            #    再把新 K/V 拼到缓存后面，就能让新 token 的 Q 看到完整前缀。
            #    这把每步生成的计算量从“重算整个前缀”降到“只算 1 个 token”。
            #    cache 在 dim=1（序列维）上拼接，存的是 RoPE 之后、repeat_kv 之前的 K/V。
            # shape: past K[B,P,nKV,D] ⊕ xk[B,S,nKV,D]（dim=1 拼接）→ xk[B,S_k,nKV,D]，S_k=P+S
            xk = torch.cat([past_key_value[0], xk], dim=1)
            # shape: past V[B,P,nKV,D] ⊕ xv[B,S,nKV,D] → xv[B,S_k,nKV,D]
            xv = torch.cat([past_key_value[1], xv], dim=1)
        # shape: past_kv = (K[B,S_k,nKV,D], V[B,S_k,nKV,D])（无缓存时 S_k=S）；不用缓存则为 None
        past_kv = (xk, xv) if use_cache else None
        # ⑤ repeat_kv 把 K/V 头复制到与 Q 头数相同；transpose 成 [B,heads,S,D]，
        #    让 heads 成为批维度，后面的矩阵乘在每个头内部独立进行。
        # shape: xq[B,S,nQ,D] → transpose → [B,nQ,S,D]
        #        xk[B,S_k,nKV,D] → repeat_kv → [B,S_k,nQ,D] → transpose → [B,nQ,S_k,D]
        #        xv[B,S_k,nKV,D] → repeat_kv → [B,S_k,nQ,D] → transpose → [B,nQ,S_k,D]
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
        # ⑥ 计算注意力。两条路径数学上等价：
        #  - 快速路径 F.scaled_dot_product_attention（SDPA）：PyTorch 会自动选择 FlashAttention /
        #    memory-efficient 等融合内核，把 QKᵀ、缩放、mask、softmax、乘 V 融合成一个 kernel，
        #    分块计算、不在显存里物化完整的 [S,S] 分数矩阵，显存从 O(S²) 降到 O(S)，速度也更快。
        #    is_causal=True 让内核内部生成下三角因果 mask。
        #    只有在“无 KV cache、无 padding”时才走这里，因为 is_causal 假设 query 与 key 从同一位置起步、
        #    且这里没有额外传 padding mask。
        #  - 手写路径：带 KV cache 的增量解码、或有 padding（RL 中左侧补齐的 prompt）时使用。
        # 快速路径的条件判断：attention_mask[B,S_k] → (==1)[B,S_k] → all → 标量 bool
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            # shape: q[B,nQ,S,D], k[B,nQ,S_k,D], v[B,nQ,S_k,D]（此路径 P=0，故 S_k=S）→ output[B,nQ,S,D]
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            # scores=[B,heads,S_q,S_k]：S_q 是本次输入长度，S_k 是“缓存 + 本次”的总长度。转 float 保证 softmax 精度。
            # shape: xq[B,nQ,S,D] @ xk.transpose(-2,-1)[B,nQ,D,S_k] → [B,nQ,S,S_k]；/ 标量 √D → scores[B,nQ,S,S_k]
            scores = (xq @ xk.transpose(-2, -1)).float() / math.sqrt(self.head_dim)
            # 因果 mask：triu(1) 是严格上三角（对角线以上）全为 -inf 的矩阵。只加在最后 seq_len 列上，
            # 因为缓存中的历史 key 对所有新 query 都可见，只有本次输入内部才存在“未来位置”。
            # softmax 中 exp(-inf)=0，被遮住的位置权重恰好为 0。
            # shape: scores[:,:,:,-S:][B,nQ,S,S] += triu(1) 掩码 [S,S]（广播到 B、nQ）；scores 整体仍为 [B,nQ,S,S_k]
            if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            # padding mask：attention_mask 形状 [B,S_k]，1=真实 token，0=补齐 token。扩成 [B,1,1,S_k]
            # 后广播到所有头和所有 query，给 padding 列加上 -1e9，使任何 query 都不会从补齐位置取信息。
            # shape: attention_mask[B,S_k] → unsqueeze(1) [B,1,S_k] → unsqueeze(2) [B,1,1,S_k]
            #        (1.0 - mask) * -1e9 → [B,1,1,S_k]；scores[B,nQ,S,S_k] += 广播 → [B,nQ,S,S_k]
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            # shape: softmax(scores, dim=-1) [B,nQ,S,S_k]（沿 key 维，每行和为 1）→ type_as → dropout → [B,nQ,S,S_k]
            #        @ xv[B,nQ,S_k,D] → output[B,nQ,S,D]
            output = self.attn_dropout(F.softmax(scores, dim=-1).type_as(xq)) @ xv
        # ⑦ 合并多头：[B,heads,S,D] → [B,S,heads*D]，再经 o_proj 混合各头信息并投影回 H 维。
        # shape: output[B,nQ,S,D] → transpose(1,2) [B,S,nQ,D] → reshape → [B,S,nQ·D]
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        # shape: output[B,S,nQ·D] → o_proj → [B,S,H] → dropout → [B,S,H]
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv  # output[B,S,H]；past_kv=(K,V) 各 [B,S_k,nKV,D] 或 None

class FeedForward(nn.Module):
    """逐位置前馈网络（FFN / MLP）：独立变换每个位置的特征，不在 token 之间传递信息。

    分工：Attention 负责“从别的 token 搬运信息”，FFN 负责“在当前 token 内部加工信息”。
    模型的大部分参数量（以及一般认为的“知识存储”）集中在 FFN 中。
    结构是先升维到 intermediate_size、做非线性、再降回 H 维。
    """
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)  # weight [I, H]
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)  # weight [H, I]
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)  # weight [I, H]
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # SwiGLU：FFN(x) = down( SiLU(gate(x)) ⊙ up(x) )，其中 SiLU(z)=z·sigmoid(z)，⊙ 为逐元素乘。
        # gate 分支经激活后充当“软开关”，逐维决定 up 分支的哪些特征通过、放大或抑制；
        # 相比传统的 down(ReLU(up(x)))，门控结构在同等参数量下效果更好，是 LLaMA/Qwen 的标准配置。
        # shape: x[B,S,H]（作为 MoE 专家时为 [n_i,H]，下同只是前缀维度不同）
        #        gate_proj(x) [B,S,I] → act_fn（逐元素）→ [B,S,I]
        #        up_proj(x)   [B,S,I]
        #        逐元素乘 [B,S,I] → down_proj → [B,S,H]
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MOEFeedForward(nn.Module):
    """MoE（Mixture of Experts，专家混合）前馈层：用路由器为每个 token 选择少数专家参与计算。

    动机：想增加参数量（容量）又不想等比例增加每个 token 的计算量。做法是把一个 FFN
    换成 num_experts 个结构相同的 FFN（专家），每个 token 只经过其中 top-k 个。
    于是“总参数量”≈ 专家数 × 单个 FFN，而“激活参数量”（每个 token 实际参与计算的）只有 k 个。
    这就是日志里 Model Params: 总量M-A激活量M 的含义（见 trainer_utils.get_model_params）。
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)  # weight [E, H]
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # 路由不关心 token 属于哪条样本、哪个位置，所以先把 [B,S,H] 摊平成 T=B·S 个 token：x_flat=[T,H]。
        batch_size, seq_len, hidden_dim = x.shape  # shape: x[B,S,H] → B, S, H
        x_flat = x.view(-1, hidden_dim)  # shape: x[B,S,H] → x_flat[T,H]，T=B·S
        # ① router（路由器）self.gate 是 H→E 的线性层，softmax 后 scores=[T,E] 表示每个 token 对各专家的偏好概率。
        # shape: x_flat[T,H] → gate → [T,E] → softmax(dim=-1) → scores[T,E]
        scores = F.softmax(self.gate(x_flat), dim=-1)
        # ② 每个 token 取分数最高的 k 个专家：topk_idx=[T,k] 是专家编号，topk_weight=[T,k] 是对应权重。
        # shape: scores[T,E] → topk_weight[T,k], topk_idx[T,k]
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        if self.config.norm_topk_prob:
            if self.config.num_experts_per_tok > 1:
                # k>1：把选中的 k 个权重重新归一化为和为 1，输出是这 k 个专家输出的加权平均。
                # shape: topk_weight[T,k] / (sum(dim=-1,keepdim)[T,1] + 1e-20) → topk_weight[T,k]（每行和为 1）
                topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
            else:
                # k=1：归一化后权重恒为 1，前向相当于“只用被选专家的原始输出”，但这样 router 就收不到梯度了。
                # straight-through 技巧：top1 - top1.detach() 的数值恒为 0、梯度却等于 top1 的梯度，
                # 所以前向值是 1.0，反向时 router 仍能按“该专家输出对 loss 的贡献”学习打分。
                # shape: x_flat.detach()[T,H] → gate → [T,E] → softmax → [T,E] → topk(k=1)[0] → top1[T,1]
                top1 = torch.topk(F.softmax(self.gate(x_flat.detach()), dim=-1), k=1, dim=-1, sorted=False)[0]
                # shape: top1[T,1] - top1.detach()[T,1] + 标量 1.0 → topk_weight[T,1]（数值恒为 1.0）
                topk_weight = top1 - top1.detach() + 1.0 # k=1: 1.0 forward on purpose, gradient via straight-through
        # ③ 按专家分发计算：对每个专家 i，找出选中它的 token，只把这些 token 送进该专家，
        #    输出乘以路由权重后用 index_add_ 累加回各 token 的位置（k>1 时一个 token 会累加多个专家的输出）。
        y = torch.zeros_like(x_flat)  # shape: x_flat[T,H] → y[T,H]（全 0，累加器）
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)  # shape: topk_idx[T,k] → mask[T,k]（bool，该位置选中专家 i）
            if mask.any():
                # shape: mask[T,k] → any(dim=-1) [T] → nonzero() [n_i,1] → flatten → token_idx[n_i]
                token_idx = mask.any(dim=-1).nonzero().flatten()
                # shape: topk_weight[T,k][mask[T,k]] → [n_i]（按行优先取出）→ view(-1,1) → weight[n_i,1]
                weight = topk_weight[mask].view(-1, 1)
                # shape: x_flat[T,H][token_idx[n_i]] → [n_i,H] → expert → [n_i,H] * weight[n_i,1] → [n_i,H]
                #        y[T,H].index_add_(dim 0, token_idx[n_i], src[n_i,H])：把 src 的每行累加到 y 对应行 → y[T,H]
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                # 本 batch 没有 token 选中该专家时，它的参数不在计算图里、不会产生梯度。
                # DDP 要求每个参数在每步都有梯度才能完成 all-reduce，否则会报“unused parameters”或卡死。
                # 这里加一个数值为 0 的项，把这些参数“挂”进计算图，梯度为 0 但通信能正常进行。
                # shape: p（各权重矩阵，如 [I,H]/[H,I]）.sum() → []；累加后 * 0 → []；加到 y[0,0]（标量元素）上，y 仍为 [T,H]
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        # ④ 负载均衡辅助损失（Switch Transformer 形式）：aux = E · Σ_i f_i · P_i · coef。
        #    f_i = load：实际被分配给专家 i 的 token 比例（来自离散的 topk，不可导）；
        #    P_i = scores.mean(0)：router 给专家 i 的平均概率（可导）。
        #    当所有专家均匀分配时 aux 最小；若少数专家“赢者通吃”，f 与 P 同时偏大，aux 变大。
        #    梯度只通过 P 流回 router，推动它把概率分散开，避免少数专家过载、其余专家闲置（路由坍塌）。
        #    aux_loss 存在模块属性上，由 MiniMindModel 汇总各层后返回，训练脚本把它加到总 loss 上。
        if self.training and self.config.router_aux_loss_coef > 0:
            # shape: topk_idx[T,k] → one_hot → [T,k,E] → float → mean(0)（对 token 维求平均）→ load[k,E]
            #        （k 个“选择槽位”各自统计一行；k=1 时为 [1,E]，下面 sum() 会把 k 个槽位的比例加起来）
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            # shape: scores[T,E] → mean(0) → [E]；load[k,E] * [E]（广播）→ [k,E] → sum() → []；* 标量 E * 标量 coef → aux_loss[]
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()  # shape: new_zeros(1)[1] → squeeze → aux_loss[]
        return y.view(batch_size, seq_len, hidden_dim)  # shape: y[T,H] → [B,S,H]

class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # 一个 Transformer block = 两个“Pre-Norm + 残差”子层：
        #     h = h + Attention(RMSNorm(h))     # 跨 token 混合信息
        #     h = h + FFN(RMSNorm(h))           # 逐 token 加工特征
        # residual connection（残差连接）：子层只需学习“增量”，输入原样加回输出。这条恒等路径
        # 贯穿所有层（常称 residual stream，残差流），反向传播时梯度可以不经衰减地直达浅层，
        # 是深层网络能训得动的关键。
        # Pre-Norm：归一化放在子层“入口”，而不是残差相加之后（Post-Norm）。残差流本身不被归一化，
        # 训练更稳定、对学习率不那么敏感；代价是残差流数值会随层数增长，所以
        # MiniMindModel 在所有层之后还要再做一次最终 norm。
        residual = hidden_states  # shape: hidden_states[B,S,H] → residual[B,S,H]（同一张量的引用）
        # shape: hidden_states[B,S,H] → input_layernorm → [B,S,H] → self_attn(cos/sin[S,D], 历史 K/V[B,P,nKV,D]) →
        #        hidden_states[B,S,H]，present_key_value=(K[B,S_k,nKV,D], V[B,S_k,nKV,D]) 或 None
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual  # shape: hidden_states[B,S,H] += residual[B,S,H] → [B,S,H]
        # shape: hidden_states[B,S,H] → post_attention_layernorm → [B,S,H] → mlp → [B,S,H]；与 hidden_states[B,S,H] 相加 → [B,S,H]
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value  # [B,S,H], (K,V) 各 [B,S_k,nKV,D] 或 None

class MiniMindModel(nn.Module):
    """Transformer 主干：embedding → N 个 block → 最终 norm，输出每个位置的 hidden state。

    它只负责“理解上下文”，不负责映射到词表；映射到词表由外层 MiniMindForCausalLM 的 lm_head 完成。
    PPO 的 Critic 也复用这个主干，只是把输出头换成了标量价值头。
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)  # weight [V, H]
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        # register_buffer：随模型一起 .to(device)，但不是可训练参数、不接收梯度；
        # persistent=False 表示不写进 state_dict（它由 config 确定性算出，无需保存到 checkpoint）。
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        # input_ids=[B,S]；attention_mask=[B,S_total]（1=真实 token，0=padding）；
        # past_key_values 是长度为层数的列表，每项是该层的 (K, V) 缓存，训练时为 None。
        batch_size, seq_length = input_ids.shape  # shape: input_ids[B,S] → B, S
        # transformers 的通用 generate 会传入它自己的 Cache 对象（有 .layers 属性），
        # 与本实现的“元组列表”格式不兼容，这里丢弃，改为不用外部缓存。
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)  # 长度 L 的列表，每项 (K[B,P,nKV,D], V[B,P,nKV,D]) 或 None
        # start_pos：缓存里已有多少个 token。增量解码时新 token 的绝对位置从这里开始，
        # 用来从 RoPE 表中切出正确位置的 cos/sin。
        # shape: past_key_values[0][0]（第 0 层 K）[B,P,nKV,D] → shape[1] → start_pos = P（整数标量）
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        # embedding（词嵌入）：nn.Embedding 本质是一个 [V,H] 的可学习查找表，按 id 取出第 id 行，
        # 得到 [B,S,H]。它等价于 one-hot(id) 乘以权重矩阵，但查表比矩阵乘高效得多。
        # shape: input_ids[B,S] → embed_tokens（查表 weight[V,H]）→ [B,S,H] → dropout → hidden_states[B,S,H]
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        # （from_pretrained 可能先在 meta 设备上建空模型再填权重；非持久 buffer 不在权重文件里，
        #  会留下全 0 的 cos 表。cos(0)=1，所以 freqs_cos[0,0]==0 就说明表丢失，需要重算。）
        # shape: self.freqs_cos[M,D] → [0,0] → 标量（比较得到标量 bool）
        if self.freqs_cos[0, 0] == 0:
            # shape: 重算后 freqs_cos[M,D], freqs_sin[M,D]
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            # shape: [M,D] → .to(device) → [M,D]（形状不变，仅搬设备）
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        # 位置从 start_pos 开始连续编号，不会跳过 padding。RL 中 prompt 采用左侧补齐时，
        # 真实 token 的绝对位置会整体后移若干格；由于 RoPE 只让注意力依赖相对位置差，
        # 且 padding 列已被 attention_mask 屏蔽，这种整体平移基本不影响结果。
        # shape: freqs_cos[M,D][P:P+S] → [S,D]；freqs_sin[M,D][P:P+S] → [S,D]
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
        presents = []  # 将收集 L 个 (K[B,S_k,nKV,D], V[B,S_k,nKV,D])
        # 逐层前向：每层都接收相同的位置编码切片和“自己那一层”的历史 cache。
        # presents 按层收集更新后的 cache，供 generate 的下一步增量解码复用。
        for layer, past_key_value in zip(self.layers, past_key_values):
            # shape: hidden_states[B,S,H], position_embeddings=(cos[S,D], sin[S,D]), past_key_value=(K[B,P,nKV,D], V[B,P,nKV,D]) 或 None
            #        → hidden_states[B,S,H], present=(K[B,S_k,nKV,D], V[B,S_k,nKV,D]) 或 None
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        # 最终 RMSNorm：Pre-Norm 结构中残差流没有被归一化过，输出到 lm_head 之前统一尺度。
        hidden_states = self.norm(hidden_states)  # shape: hidden_states[B,S,H] → [B,S,H]
        # 汇总所有 MoE 层的负载均衡损失；dense 模型返回标量 0，训练脚本可统一写 loss + aux_loss。
        # shape: 每层 l.mlp.aux_loss[]；初始值 new_zeros(1)[1] → squeeze → []；求和后 aux_loss[]
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss  # hidden_states[B,S,H], presents（L 个 (K,V)）, aux_loss[]

class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """因果语言模型外壳：将隐状态投影到词表，并提供训练 loss 与生成协议。

    继承 PreTrainedModel 获得 save_pretrained/from_pretrained、权重初始化(post_init)等能力；
    继承 GenerationMixin 以兼容 transformers 的生成接口（本类重写了自己的 generate）。
    """
    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)  # weight [V, H]
        # 权重绑定：embedding 表 [V,H] 与 lm_head.weight [V,H] 指向同一个张量。
        # 直观理解：输入时“id → 向量”与输出时“向量 → 各 id 的相似度分数”共用一套词向量。
        # 两处梯度会累加到同一个参数上；state_dict 里仍会出现两个键名，但只占一份显存。
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight
        # post_init 调用 transformers 的 _init_weights 等逻辑，完成参数初始化与权重绑定的收尾工作。
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        # shape: input_ids[B,S], attention_mask[B,S_k] 或 None → hidden_states[B,S,H], past_key_values（L 个 (K,V)，各 [B,S_k,nKV,D]）, aux_loss[]
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        # logits_to_keep=0 时 slice(-0, None)=slice(0, None)，即保留全部位置（训练用）；
        # 取正整数 n 时只保留最后 n 个位置。生成时只需要最后一个位置的分布，RL 中只需要
        # 回答部分的分布，都可以借此跳过大量 [H→V] 投影，V 较大时省显存也省时间。
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep  # 切片对象/索引张量，无张量形状
        # shape: hidden_states[B,S,H] → [:, slice_indices, :] → [B,S',H] → lm_head → logits[B,S',V]
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        # logits（未归一化分数）形状 [B,S,V]：logits[b,t,:] 是“看完 0..t 之后，下一个 token 为各词”的打分。
        # softmax 把它转成概率分布 p(x_{t+1} | x_{≤t})，生成时再从分布中选/采样 token。
        loss = None
        if labels is not None:
            # 错位对齐：logits 丢掉最后一个位置（它预测的是序列之外的 token，没有答案），
            # labels 丢掉第一个位置（第 0 个 token 没有前文可以预测它）。
            # 对齐后第 t 对 = (看完 0..t 的预测, 真实的第 t+1 个 token)。
            # 因为因果 mask，一次前向就同时得到了 S-1 个“预测下一个 token”的训练样本（teacher forcing）。
            # shape（训练时 S'=S）：logits[B,S,V] → [..., :-1, :] → x[B,S-1,V]；labels[B,S] → [..., 1:] → y[B,S-1]
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            # F.cross_entropy 这一行内部做了三件事：
            #   1) 对 V 维做 log_softmax，得到每个位置对整个词表的对数概率；
            #   2) 取出正确 token 的那一项并取负：-log p(正确 token)（即负对数似然 NLL）；
            #   3) 丢弃 label==-100 的位置后，对剩余有效 token 求平均（默认 reduction='mean'）。
            # 所以 loss 是“batch 内所有有效 token 的平均 NLL”，exp(loss) 就是困惑度 perplexity。
            # 哪些位置是 -100 完全由数据集决定：预训练只忽略 padding，SFT 还会忽略 prompt 部分。
            # shape: x[B,S-1,V] → view(-1,V) → [B·(S-1),V]；y[B,S-1] → view(-1) → [B·(S-1)]
            #        cross_entropy([B·(S-1),V], [B·(S-1)]) → loss[]（有效 token 的平均 NLL，标量）
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        # aux_loss 单独返回，不加进 loss；是否、如何加由训练脚本决定。
        # 输出：loss[] 或 None；aux_loss[]；logits[B,S',V]；past_key_values=L 个 (K,V)；hidden_states[B,S,H]
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
    
    # https://github.com/jingyaogong/minimind/discussions/611
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        # 自回归解码循环：每轮只产生 1 个新 token，把它接到序列末尾，再继续预测下一个。
        # @torch.inference_mode 关闭梯度记录（比 no_grad 更彻底），推理不需要反向传播。
        # shape: inputs[B0,S0] → repeat(n,1) → input_ids[B,S0]，B=B0·n（n=num_return_sequences）
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        # shape: attention_mask[B0,S0] → repeat(n,1) → [B,S0]
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)  # None，或 L 个 (K[B,P,nKV,D], V[B,P,nKV,D])
        # finished[b]=True 表示第 b 条序列已经生成过 EOS；batch 内各序列长度不同，需要逐条追踪。
        # shape: input_ids.shape[0]=B → finished[B]（bool）
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())  # shape: input_ids[B,S0] → cpu → [B,S0]
        for _ in range(max_new_tokens):
            # prefill 与 decode 两个阶段：第一轮没有缓存，past_len=0，整个 prompt 一次性送入并建立 KV cache（prefill）；
            # 之后每轮只送入“缓存之后的新 token”（通常就是上一轮刚采样的 1 个），其余靠缓存（decode）。
            # shape: past_key_values[0][0]（第 0 层 K）[B,P,nKV,D] → shape[1] → past_len = P（整数标量）
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            # shape: input_ids[B,S_cur][:, P:] → [B,S_new]（首轮 S_new=S0，之后 S_new=1，且 P+S_new=S_cur）
            #        attention_mask[B,S_cur] → outputs.logits[B,S_new,V]；outputs.past_key_values: L 个 (K,V)，各 [B,S_cur,nKV,D]
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            # 新 token 一定是真实 token，attention_mask 末尾补 1，与下一轮的 K/V 总长度保持一致。
            # shape: attention_mask[B,S_cur] ⊕ new_ones[B,1]（dim=-1 拼接）→ [B,S_cur+1]
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            # 只取最后一个位置的 logits=[B,V]：它就是“下一个 token”的打分。
            # temperature（温度）：logits/T 后再 softmax。T<1 放大差距、分布更尖锐（更确定）；
            # T>1 缩小差距、分布更平坦（更随机）；T→0 等价于贪心 argmax。
            # shape: outputs.logits[B,S_new,V] → [:, -1, :] → [B,V]；/ 标量 temperature → logits[B,V]
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                # 重复惩罚：对已出现过的 token 降低分数（正分除以 penalty、负分乘以 penalty，保证都往“更不可能”方向走）。
                for i in range(input_ids.shape[0]):
                    # shape: input_ids[i][S_cur] → unique → seen[n_seen]（该序列出现过的不同 token id）
                    #        logits[i,seen] → score[n_seen]；where(score>0, score/p, score*p) → [n_seen]，写回 logits[i,seen]（logits 仍为 [B,V]）
                    seen = torch.unique(input_ids[i]); score = logits[i, seen]; logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            if top_k > 0:
                # top-k 截断：只保留分数最高的 k 个 token，其余设为 -inf（softmax 后概率为 0），砍掉长尾噪声词。
                # shape: logits[B,V] → topk(logits, top_k)[0] → [B,top_k] → [..., -1, None] → 第 top_k 大的值 [B,1]
                #        logits[B,V] < [B,1]（广播）→ bool [B,V] → 对应位置赋 -inf；logits 仍为 [B,V]
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                # top-p（nucleus）截断：按概率从高到低排序，保留累计概率刚好达到 top_p 的最小集合。
                # 与 top-k 相比，候选集大小随分布自适应：模型很确定时只剩几个词，不确定时保留更多。
                # shape: logits[B,V] → sort(descending) → sorted_logits[B,V], sorted_indices[B,V]（原词表下标）
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                # shape: sorted_logits[B,V] → softmax → [B,V] → cumsum → [B,V] → > top_p（标量）→ mask[B,V]（bool）
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                # mask 整体右移一位：让“使累计概率首次超过 top_p 的那个 token”也被保留，且至少保留第 1 名。
                # shape: mask[..., :-1][B,V-1] → clone → 赋给 mask[..., 1:][B,V-1]；mask[..., 0][B] = 0；mask 仍为 [B,V]
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                # scatter 把“排序后坐标”下的 mask 映射回原始词表坐标。
                # shape: mask[B,V].scatter(dim=1, index=sorted_indices[B,V], src=mask[B,V]) → [B,V]（原词表坐标的 bool）
                #        → logits[B,V][该 bool 为 True 的位置] = -inf；logits 仍为 [B,V]
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            # do_sample=True：按截断后的概率分布随机采样（multinomial）；False：贪心取最大值（argmax）。
            # shape: logits[B,V] → softmax → [B,V] → multinomial(num_samples=1) → next_token[B,1]
            #        或 logits[B,V] → argmax(dim=-1,keepdim=True) → next_token[B,1]
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            # 已结束的序列无论采到什么都强制填 EOS，保证 batch 对齐的同时不引入新内容。
            # 因此返回结果里 EOS 之后可能跟着一串 EOS，下游（如 RL 的 completion_mask）需要自行截断。
            # shape: finished[B] → unsqueeze(-1) → [B,1]；new_full((B,1), eos) → [B,1]；next_token[B,1]
            #        where(cond[B,1], eos[B,1], next_token[B,1]) → next_token[B,1]
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            # shape: input_ids[B,S_cur] ⊕ next_token[B,1]（dim=-1 拼接）→ input_ids[B,S_cur+1]
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            # shape: outputs.past_key_values → L 个 (K,V)，各 [B,S_cur,nKV,D]（S_cur 取本轮 forward 时的长度，
            #        即比刚拼接 next_token 后的 input_ids 少 1，新 token 下一轮才进缓存）；不用缓存则 None
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())  # shape: next_token[B,1] → cpu → [B,1]
            if eos_token_id is not None:
                # shape: next_token[B,1] → squeeze(-1) → [B] → eq(eos) → bool[B]；finished[B] |= bool[B] → [B]
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break  # shape: finished[B] → all() → 标量 bool
        if streamer: streamer.end()
        # past_kv：L 个 (K,V)，各 [B,S_final-1,nKV,D]（最后一个采样出的 token 尚未经过 forward，不在缓存里）
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids  # shape: [B,S_final]，S_final = S0 + 实际生成的 token 数
