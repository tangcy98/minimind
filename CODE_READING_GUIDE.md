# MiniMind 训练代码阅读路线与术语导览

这份导览把代码串成一条完整的数据流：原始文本 → token id → 模型预测 → 训练目标 → 参数更新 → 检查点 → 推理。
每个代码文件的开头都标有 **【阅读路线 第 N 站】**，与本文的章节编号一一对应。建议按章节顺序阅读：先读本文对应章节建立概念，再打开代码按文件头给出的顺序读，代码内的注释会解释“这几行在做什么、背后调用的库做了什么、这一步对应哪个训练概念”。

仓库中的 **token**（词元）是模型处理的最小离散符号，可以是字、词、标点或子词片段；它既不是固定等同于“一个汉字”，也不保证对应完整单词。模型最终处理的是 token 对应的整数 id。

## 全局地图

```
第1站 tokenizer         字符串 ⇄ id                            trainer/train_tokenizer.py, model/tokenizer*.json
第2站 模型              id → logits → 交叉熵 / 采样             model/model_minimind.py
第3站 数据              哪些 token 参与 loss（labels / mask）   dataset/lm_dataset.py
第4站 预训练            loss → 梯度 → 参数更新（训练骨架）       trainer/train_pretrain.py, trainer/trainer_utils.py
第5站 SFT / LoRA        换数据与 mask；只训练低秩增量             trainer/train_full_sft.py, model/model_lora.py, trainer/train_lora.py
第6站 蒸馏 / DPO        模仿教师分布；从偏好对学习               trainer/train_distillation.py, trainer/train_dpo.py
第7站 在线 RL           采样 → 打分 → 策略梯度                   trainer/rollout_engine.py → train_grpo.py → train_ppo.py → train_agent.py
第8站 推理与服务        加载权重、按模板生成                      eval_llm.py, scripts/*
```

贯穿全程的一条主线：**各阶段的模型结构几乎不变，loss 的核心也始终是“某些 token 的 log 概率”**。阶段之间真正变化的是：用什么数据、哪些 token 参与、这些 log 概率被乘上什么权重（常数 1、偏好差、优势 A……）。

## 0. 先了解项目阶段

先读 `README.md` 中的模型结构、主要训练和强化学习章节。**训练阶段**是按目标划分的一段模型优化过程；**强化学习（RL，Reinforcement Learning）**通过行动后的奖励反馈改进策略。各阶段要改变模型的能力：

- **预训练（pretraining）**：在大量文本上预测下一个 token，让模型获得语言模式和知识表征。
- **监督微调（SFT，Supervised Fine-Tuning）**：用示范对话继续训练，只要求模型生成助手回答。
- **偏好优化**：从同一问题下的优选/较差回答中学习偏好。**DPO（Direct Preference Optimization，直接偏好优化）**是直接从偏好对调整策略的方法。
- **在线强化学习**：让模型自己生成回答，按奖励反馈调整生成策略。**PPO（Proximal Policy Optimization，近端策略优化）**和 **GRPO（Group Relative Policy Optimization，组相对策略优化）**是本仓库实现的策略优化方法。

**策略（policy）**在强化学习中就是产生回答的模型。不同阶段通常使用同一个模型结构，但目标函数、数据组织和参与训练的参数可能不同。

权重的接力关系（`trainer_utils.init_model` 的 `from_weight` 参数）：`none → pretrain → full_sft → {dpo, grpo, ppo_actor, agent, lora_*, full_dist}`。

## 1. Tokenizer：字符串如何变成 id

读 `model/tokenizer.json`、`model/tokenizer_config.json`，再读 `trainer/train_tokenizer.py` 的 `get_texts`、`train_tokenizer`、`eval_tokenizer`。

**Tokenizer（分词器）**负责把字符串切成 token，并将每个 token 映射成整数 id；模型不直接读写字符串。**词表（vocabulary）**是 token 与 id 的固定映射表。**BPE（Byte Pair Encoding，字节对编码）**是一种学习子词词表的方法：从 256 个字节出发，反复把语料中最高频的相邻片段对合并成新 token，直到词表达到目标大小。因为基础单元是字节，任何字符串都能被编码。

**特殊 token**承担结构控制，MiniMind 中：

| token | id | 作用 |
|---|---|---|
| `<\|endoftext\|>` | 0 | PAD：补齐长度 |
| `<\|im_start\|>` | 1 | BOS：序列起点，同时是每条对话消息的开头 |
| `<\|im_end\|>` | 2 | EOS：序列结束，同时是每条消息的结尾；生成遇到它即停止 |
| `<think>` `<tool_call>` `<tool_response>` 等 | 21~26 | 思考与工具调用的结构标记（非 special，解码时不会被删掉） |

重点思考：

- `trainer/train_tokenizer.py` 负责“学习词表”，不是训练语言模型本身，也不需要 GPU。
- 词表中 id 的含义必须和模型 embedding 行保持一致；换了词表，即使模型张量尺寸相同，行对应的 token 也可能变了。
- 阅读 `model/tokenizer_config.json` 中的 chat template，观察 `system/user/assistant/tool` 如何变成模型输入序列。注意 `tool` 角色的内容会被包进 `<tool_response>`，并以 **`user`** 身份出现在 token 序列里。
- `eval_tokenizer` 中的流式解码测试说明：一个汉字的多个 UTF-8 字节可能分属不同 token，逐 token 解码会出现 `�`。

## 2. 模型：token id 如何变成下一个 token 的概率

按下面顺序读 `model/model_minimind.py`（文件头有整体数据流图）：

1. `MiniMindConfig`
2. `RMSNorm`、`precompute_freqs_cis`、`apply_rotary_pos_emb`
3. `repeat_kv`、`Attention`
4. `FeedForward`、`MOEFeedForward`
5. `MiniMindBlock`、`MiniMindModel`
6. `MiniMindForCausalLM.forward`（训练）与 `MiniMindForCausalLM.generate`（推理）

**Embedding（词嵌入）**是按 token id 查表得到的可训练向量。**隐状态（hidden state）**是 token 向量经过模型层后、带有上下文信息的表示。这里的主要形状是 `input_ids [B,S]` → `hidden_states [B,S,H]` → `logits [B,S,V]`，其中 `B` 是 batch 样本数、`S` 是序列长度、`H` 是隐藏维度、`V` 是词表大小。MiniMind 默认 **权重绑定**（`tie_word_embeddings`）：输入 embedding 与输出 `lm_head` 共用同一个 `[V,H]` 矩阵。

MiniMind 是 **Decoder-only Transformer（仅解码器 Transformer）**：它没有单独的编码器堆栈，依靠**因果遮罩（causal mask）**屏蔽当前位置之后的 token，确保预测时不偷看未来。因果遮罩也让训练时一次前向就能并行得到所有位置的预测（teacher forcing）。它完成**自回归生成**：每一步只预测下一个 token，生成后把它接到上下文末尾，再预测下一个。

**注意力（attention）**让当前位置按相关性从前文汇集信息：`softmax(QKᵀ/√D + mask)·V`。Query（Q，查询向量）表示当前位置要找什么，Key（K，键向量）表示一个位置如何被匹配，Value（V，值向量）是匹配后要汇总的内容。多个并行的子空间称为**注意力头（head）**。**GQA（Grouped Query Attention，分组查询注意力）**让多个 Q 头共享较少的 K/V 头，从而减少生成时 KV cache 的显存。**KV cache（键值缓存）**保存已生成前缀的 K/V，避免每生成一个 token 就重算整个前缀。本实现还在每个头内对 Q/K 做 RMSNorm（**QK-Norm**），稳定注意力分数的数值范围。满足条件时注意力走 PyTorch 的 `scaled_dot_product_attention` 融合内核（FlashAttention 等），不在显存中物化完整的 `[S,S]` 分数矩阵。

**RoPE（Rotary Positional Embedding，旋转位置编码）**把 Q/K 每两维视为一个平面按“位置 × 频率”旋转，使点积只依赖相对位置差；YaRN 是把 RoPE 扩展到更长上下文的一种分段频率缩放方法。**RMSNorm**用向量均方根归一化特征尺度。**残差连接（residual connection）**把子层输入直接加回输出，为信息和梯度提供短路径；Pre-Norm 表示先归一化再进入子层，因此最后还需要一次最终 norm。

**SwiGLU**是门控前馈网络：`down(SiLU(gate(x)) ⊙ up(x))`。它在每个 token 内变换特征，注意力则负责 token 之间的信息混合。

**MoE（Mixture of Experts，专家混合）**用路由器为每个 token 选择少量专家前馈网络计算。路由器输出选择分数，top-k 选出本 token 激活的专家；稀疏激活降低每个 token 的计算量，但模型仍保存所有专家参数（日志中 `总量M-A激活量M`）。**辅助负载均衡损失（auxiliary loss）**`E·Σ f_i·P_i` 鼓励 token 不要都挤到少数专家；k=1 时代码用 straight-through 技巧让路由器仍能获得梯度。

`MiniMindForCausalLM.forward` 最后把隐状态投影到词表大小 `V`，得到 **logits**（未归一化分数）。**softmax**通过指数归一化把 logits 转成总和为 1 的词表概率分布。训练时使用**交叉熵（cross entropy）**衡量预测分布与正确 token 的差异：`F.cross_entropy` 一行内部完成 log_softmax、取正确 token 的负对数概率、忽略 `-100`、对有效 token 求平均。位置 `t` 的 logits 预测位置 `t+1`，所以代码中是 `logits[:, :-1]` 对 `labels[:, 1:]`。

`generate` 展示了推理的全部要素：prefill/decode 两阶段与 KV cache、temperature、repetition penalty、top-k、top-p（nucleus）截断、随机采样与贪心、EOS 处理。

## 3. 数据：哪些 token 参与 loss

阅读 `dataset/lm_dataset.py`，顺序是 `PretrainDataset` → `SFTDataset` → `DPODataset` → `RLAIFDataset` → `AgentRLDataset`。

PyTorch 的 **Dataset** 定义如何读取一条样本；**DataLoader** 再负责多进程读取、取多条样本组成 batch。**label（标签）**在语言模型训练里是希望模型预测的 token id。**padding（补齐）**把不同长度序列补到相同长度；`-100` 是交叉熵的 ignore index（忽略标签），这种位置不计入 loss。

### 预训练与 SFT 数据

- `PretrainDataset.__getitem__`：读取一段普通文本，编码后加 BOS/EOS，再补齐；除 PAD 外的文本 token 都能产生 next-token 监督。
- `SFTDataset.create_chat_prompt`：用 **chat template（对话模板）**把角色、轮次和工具定义序列化成训练文本。
- `SFTDataset.generate_labels`：仅对 assistant 内容（含 `<think>` 与 `<tool_call>`）和结尾的 `<|im_end|>` 开启标签。user/system/tool 内容用于提供条件，不需要模型复述；监督结束符是为了让模型学会何时停止。
- 数据集中的 labels 与 input_ids 同位置，不做错位；模型 forward 负责错开一位，将“位置 t 的上下文”与“位置 t+1 的目标 token”配对。
- `pre_processing_chat` / `post_processing_chat` 是数据增强：随机插入 system prompt、随机去掉空的 think 壳。

**SFT（Supervised Fine-Tuning，监督微调）**是用示范输入/回答继续训练模型。它仍然用 next-token 交叉熵，变化的是对话格式和哪些标签参与 loss。训练模板与推理模板不匹配，会导致模型看到不同的分隔符和角色结构。

### 偏好与在线 RL 数据

- `DPODataset`：同一个对话上下文准备 `chosen`（偏好回答）和 `rejected`（较差回答）。它在数据侧直接完成错位：`x = ids[:-1]`、`y = ids[1:]`，`mask = loss_mask[1:]` 与 `y` 逐位对齐。
- `RLAIFDataset`：只准备 prompt（去掉最后的 assistant 回复，并追加生成前缀），答案会在线生成。
- `AgentRLDataset`：准备结构化的消息历史、可用工具和 ground truth（标准答案，用于自动验证任务结果）。

## 4. 预训练：一次参数更新从哪里来

先读 `trainer/train_pretrain.py::train_epoch`，再读文件底部的 `__main__` 初始化，随后按需跳到 `trainer/trainer_utils.py` 的 `get_lr`、`init_model`、`lm_checkpoint`、`SkipBatchSampler`。**这是后续所有训练脚本共用的骨架**，读懂它，后面只需关注“loss 换成了什么”。

**loss（损失）**是用于优化的标量目标，训练试图让它下降。**梯度（gradient）**是 loss 对参数的导数，告诉优化器每个参数应该向哪个方向调整。一个训练 step（一步）依次做：

1. 取 batch 并搬到训练设备，按全局进度设置学习率。
2. 前向传播（autocast 混合精度）：模型计算 logits 和交叉熵；MoE 还计算路由辅助损失。
3. loss 除以累积步数 N，再反向传播：梯度被**累加**到 `param.grad`；DDP 下反向过程中各卡梯度被 all-reduce 求平均。
4. 梯度累积：每 N 个 micro-batch 才更新一次，等价于 N 倍的有效 batch。
5. `unscale_` 还原梯度尺度后做梯度裁剪：总 L2 范数超过阈值时整体等比缩小。
6. AdamW 更新：一阶矩（动量）、二阶矩（自适应步长）、偏差修正、解耦权重衰减。
7. 清零梯度；定期打印日志和保存 checkpoint。

**优化器（optimizer）**负责把梯度转换为参数更新；**AdamW**为每个参数维护 m、v 两份状态（约为参数量 2 倍的显存）。**学习率（learning rate）**控制每次更新幅度；`get_lr` 是从峰值余弦衰减到 10% 的调度，**没有 warmup**，且按 micro-step 计算。**epoch**表示完整遍历一次训练集，step 表示一个 batch 的处理循环。

**混合精度（mixed precision）**：`autocast` 让矩阵乘等算子用 bf16/fp16 执行，softmax/归一化等保持 fp32，参数主副本始终为 fp32。fp16 范围窄，小梯度会下溢，因此需要 **GradScaler（梯度缩放器）**先放大 loss、更新前再还原，并在出现 inf/NaN 时跳过本次更新；默认的 bf16 指数范围与 fp32 相同，脚本中 scaler 处于关闭状态。**DDP（Distributed Data Parallel，分布式数据并行）**让多个进程通过 `DistributedSampler` 处理不同数据分片、各持一份模型副本并同步梯度。

区分两种保存物：推理 checkpoint（`out/*.pth`）只有 fp16 模型参数；resume checkpoint（`checkpoints/*_resume.pth`）还保存优化器、梯度缩放器/调度器和训练位置，才能较完整地恢复训练。

## 5. SFT 与 LoRA：从续写到指令回答

对比 `trainer/train_full_sft.py` 与 `trainer/train_pretrain.py`：训练循环几乎逐行相同，能力变化来自 SFT 对话数据、assistant-only label mask，以及从预训练权重出发、使用更小学习率。

再读 `model/model_lora.py` 与 `trainer/train_lora.py`。**LoRA（Low-Rank Adaptation，低秩适配）**将权重变化近似成两个窄矩阵的乘积 `ΔW = B·A`，冻结基础权重，只训练低秩增量。`rank` 控制增量子空间大小；`B` 零初始化保证训练起点与原模型完全一致。本实现的几个具体细节：

- `apply_lora` 只给 **in_features == out_features 的方阵线性层** 挂 LoRA；默认配置下即 `q_proj` 与 `o_proj`。
- 没有原论文的 `alpha/r` 缩放系数，也没有 LoRA dropout。
- 冻结省下的是被冻结参数的梯度与优化器状态；激活显存和反向计算仍然需要。
- 保存 adapter（适配器）只保存增量；merge（合并）则把 `B·A` 加回基础矩阵，导出单一权重。

## 6. 蒸馏与 DPO：模仿分布和学习偏好

### 蒸馏

读 `trainer/train_distillation.py::distillation_loss` 和 `train_epoch`。**知识蒸馏**让 student（学生模型）拟合 teacher（教师模型）的输出。教师的概率分布是**软目标**：不仅指出正确 token，也表达其他 token 的相对可能性。**温度**调节 softmax 分布的平滑程度；**KL 散度**度量学生和教师概率分布的差异，`F.kl_div(学生 log 概率, 教师概率)` 计算的是前向 KL(教师‖学生)，乘 `T²` 是为了抵消温度对梯度量级的影响。训练混合硬标签交叉熵与软目标蒸馏损失（`alpha` 调配）。教师与学生必须共用同一词表。

### DPO

读 `DPODataset`，再读 `trainer/train_dpo.py` 的 `logits_to_log_probs`、`dpo_loss`、`train_epoch`。

**DPO（Direct Preference Optimization，直接偏好优化）**直接从偏好对优化策略，不额外训练奖励模型，也不需要在线采样。其推导来自“最大化奖励 − β·KL(π‖π_ref)”问题的闭式解：隐式奖励 `r = β·log(π/π_ref)`，代入 Bradley-Terry 偏好模型即得 `L = −log σ(β·[(log π_w − log π_ref,w) − (log π_l − log π_ref,l)])`。

**log probability（对数概率）**把序列 token 概率的连乘写成对数概率求和。函数先分别计算 policy（待更新策略）和 reference（冻结参考策略）对 chosen/rejected 回答的序列对数概率，再比较两者的偏好差值。prompt 部分在 chosen/rejected 之间相同，其 log-prob 在差值中本就会抵消，mask 让分数只反映回答并去掉 padding 噪声。此实现把回答 token 的 log-prob **求和**而非平均，因此回答长度会影响序列分数。训练初始时 policy = reference，loss = log 2 ≈ 0.693。`beta` 越大，策略越被约束在 reference 附近。

## 7. 在线强化学习：PPO、GRPO 与 Agent

先读 `trainer/rollout_engine.py` 的文件头，再按 `train_grpo.py` → `train_ppo.py` → `train_agent.py` 阅读。

**强化学习（RL）**让模型从行动结果的奖励中学习，而非只拟合固定 token 标签。在语言模型中：状态 = prompt + 已生成的 token，动作 = 下一个 token，策略 = 语言模型本身。**Rollout（轨迹采样）**指策略实际生成回答并记录过程；**轨迹（trajectory）**包含 prompt、生成 token、采样概率和奖励。**Reward（奖励）**是对回答/任务结果的标量评分，本仓库中是“规则分 + 奖励模型分”，都是**序列级**的（一条回答一个分数）；训练算法再把它变成 token 级更新信号。

所有在线 RL 脚本的每一步都分两段：

1. **采样阶段（无梯度）**：`generate` 生成回答，记录 old log-prob，计算奖励。
2. **学习阶段（有梯度）**：把 prompt + 回答重新前向一次，得到当前策略的 log-prob，组合成损失。

策略梯度的核心形式是 `A · ∇log π(a_t|s_t)`：**Advantage（优势）** A 是回答奖励相对基线的高低，决定该 token 的概率被推高还是压低。各算法的区别主要在于 A 怎么算。

### GRPO

**GRPO（Group Relative Policy Optimization，组相对策略优化）**针对同一个 prompt 采样多条回答，以组内均值为基线、标准差归一化：`A_i = (r_i − mean) / std`。同一条回答的所有 token 共享同一个 A，不需要 Critic。

`train_grpo.py` 主线：左侧补齐 prompt → rollout 采样 → 奖励打分 → 重算 policy 和 reference 的 token log-prob → 组内标准化 advantage → 按首个 EOS 截断得到有效 token mask → 策略损失 → 反向更新。

- **重要性采样比（importance ratio）**是当前策略概率除以采样时策略概率，修正“样本由旧策略生成、现在由新策略优化”的差异。在 Torch 后端下每批只更新一次，ratio 数值≈1、梯度等价于标准策略梯度；使用 SGLang 后端（权重每 `save_interval` 步才同步）时 ratio 才明显偏离 1。
- **PPO clipping（概率比裁剪）**（`loss_type=grpo`）：`min(ratio·A, clip(ratio)·A)`，超出区间的 token 梯度为 0。
- **CISPO**（默认 `loss_type=cispo`）：把重要性权重截断到上限并停止梯度，`sg(min(ratio, ε_high))·A·log π`，所有 token 都保留梯度。
- **KL（Kullback–Leibler）散度**惩罚：用 k3 估计器 `exp(x) − x − 1`（`x = log π_ref − log π`）估计 KL(π‖π_ref)，直接加进 loss，把新策略约束在 reference 附近。
- 损失先在每条回答内部对有效 token 平均，再对回答平均。

### PPO

**PPO（Proximal Policy Optimization，近端策略优化）**显式训练 **Critic（价值模型）**估计每个状态的价值 V(s_t)，并用它给每个 token 单独估计优势。本脚本中有四个模型：**Actor**（被训练的生成策略）、**Critic**（被训练的价值模型，主干从 SFT 初始化、`value_head` 随机初始化）、**reference policy**（冻结，用于 KL）、**reward model**（冻结，用于打分），不要混淆 Critic 与 reference。

读 `train_ppo.py::ppo_train_epoch` 时重点看：

1. Actor rollout 生成回答（每个 prompt 1 条），奖励函数给整条回答评分。
2. Critic 在回答的每个 token 位置估计 value（状态价值）。
3. 序列级奖励只放在最后一个有效 token 上（稀疏终局奖励）；KL 不在奖励里扣除，而是作为独立损失项。
4. **GAE（Generalized Advantage Estimation，广义优势估计）**从后往前递推 `A_t = δ_t + γλ·A_{t+1}`，其中时间差分残差 `δ_t = r_t + γ·V(s_{t+1}) − V(s_t)`；`gamma` 是未来奖励折扣，`lam` 在偏差（依赖 Critic）与方差（依赖蒙特卡洛回报）之间折中。`returns = A + V` 是 Critic 的回归目标。优势再在 batch 内标准化。
5. 同一批轨迹重复更新 `ppo_update_iters` 轮（每轮打乱并切 minibatch）：Actor 用裁剪策略损失 + KL(ref)，Critic 用裁剪价值损失。`approx_kl`（相对采样时策略）超过阈值时早停——它与相对 reference 的 KL 是两个不同的量。

### Agentic RL

读 `train_agent.py::rollout_single` → `rollout_batch` → `calculate_rewards` → `rl_train_epoch`。**Agent（智能体）**通过“观察—行动—接收反馈”与环境多轮交互。本项目中，环境是一组用确定性假数据模拟的工具。

模型生成的工具请求是策略 action（动作）；工具执行结果是 observation（观察）。工具结果会进入后续上下文，但它不是模型生成的动作，不计入策略 token 概率损失（`response_mask = 0`）；此路径下 EOS 也被排除在损失之外。为保证训练时的 token 序列与模型实际采样的完全一致，`rollout_single` 在 token 层面追加（模型生成的 id 原样保留，模板新增的环境文本用 marker 切出后单独 tokenize），而不是每轮重新 tokenize 整段文本。奖励属于“可验证奖励”：根据工具调用是否合法、数量与参数是否匹配、ground truth 是否在最终回答中命中、是否在轮数上限内完成，以及无工具时的回答质量计算。损失部分与 GRPO 相同。

## 8. 推理、评估、服务与转换

- `eval_llm.py`：命令行加载 checkpoint 并交互生成，用于人工检查；预训练权重用 “BOS + 文本” 续写，对话权重用 chat template。
- `scripts/eval_toolcall.py`：验证工具请求能否解析、执行并得到合理回答。
- `scripts/convert_model.py`：在 MiniMind 原生与 Transformers/兼容格式之间转换权重和配置。
- `scripts/serve_openai_api.py`：把模型封装成兼容 OpenAI 格式的 HTTP API。
- `scripts/chat_api.py`：调用聊天 API 的最小客户端示例。
- `scripts/web_demo.py`：Streamlit 浏览器聊天界面。

推理时 **decoding（解码）**是从 logits 概率分布中选择下一个 token 的过程；**temperature（温度）**是采样前对 logits 缩放的参数，较低时分布更尖锐、输出更确定，较高时分布更平坦、随机性更大；**top-k / top-p** 截断掉低概率长尾后再采样。训练和推理需要使用相同 tokenizer 与兼容的 chat template。

## 文件职责速览

- `model/model_minimind.py`：模型配置、Transformer 主干、MoE、因果语言模型 forward 与 generate。
- `model/model_lora.py`：LoRA 模块以及挂载、加载、保存、合并。
- `model/__init__.py`、`dataset/__init__.py`：包入口说明，无训练逻辑。
- `dataset/lm_dataset.py`：各训练阶段的数据读取、文本模板和标签 mask。
- `trainer/train_tokenizer.py`：BPE tokenizer 训练示例。
- `trainer/train_pretrain.py`、`train_full_sft.py`、`train_lora.py`：预训练、全参 SFT、LoRA SFT。
- `trainer/train_distillation.py`、`train_dpo.py`：蒸馏与偏好优化。
- `trainer/rollout_engine.py`、`train_ppo.py`、`train_grpo.py`、`train_agent.py`：在线生成、策略优化和多轮工具任务。
- `trainer/trainer_utils.py`：训练公共组件（学习率、DDP、模型加载、checkpoint、奖励模型封装）。
- `eval_llm.py`、`scripts/*`：推理、评估、服务、Web UI 和模型转换。

## 阅读时的自检问题

- 对任一 batch，`input_ids`、`labels`、`attention_mask` 的形状分别是什么？
- `logits[:, t]` 对应预测哪个 token？哪些 label 被 `-100` 忽略？SFT 为什么要监督 `<|im_end|>`？
- `loss / accumulation_steps` 为什么等价于大 batch？`scaler.unscale_` 为什么必须在梯度裁剪之前？
- DPO 的初始 loss 为什么是 log 2？β 变大时策略会更靠近还是更远离 reference？
- 这段代码用的是采样时策略、当前策略，还是 reference？ratio 在什么情况下才会偏离 1？
- GRPO 和 PPO 的优势分别来自哪里？哪一个是“每个 token 不同”的？
- loss 是按 token 求和还是平均？是先在序列内平均，还是在全部 token 上平均？padding、prompt、工具返回是否被 mask？
- resume checkpoint 保存了哪些参数以外的状态？为什么只保存参数无法精确续训？

运行训练前还需按 README 下载/准备数据集；数据文件并未全部包含在仓库中。
