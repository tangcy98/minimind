# MiniMind 训练代码阅读路线与术语导览

这份导览把代码串成一条完整的数据流：原始文本 → 模型预测 → 训练目标 → 参数更新 → 检查点 → 推理。建议按章节顺序读；专业术语在阅读路线里首次出现时解释，代码文件头也写明了该文件负责的部分。

仓库中的 **token**（词元）是模型处理的最小离散符号，可以是字、词、标点或子词片段；它既不是固定等同于“一个汉字”，也不保证对应完整单词。模型最终处理的是 token 对应的整数 id。

## 0. 先了解项目阶段

先读 `README.md` 中的模型结构、主要训练和强化学习章节。**训练阶段**是按目标划分的一段模型优化过程；**强化学习（RL，Reinforcement Learning）**通过行动后的奖励反馈改进策略。各阶段要改变模型的能力：

- **预训练（pretraining）**：在大量文本上预测下一个 token，让模型获得语言模式和知识表征。
- **监督微调（SFT，Supervised Fine-Tuning）**：用示范对话继续训练，只要求模型生成助手回答。
- **偏好优化**：从同一问题下的优选/较差回答中学习偏好。**DPO（Direct Preference Optimization，直接偏好优化）**是直接从偏好对调整策略的方法。
- **在线强化学习**：让模型自己生成回答，按奖励反馈调整生成策略。**PPO（Proximal Policy Optimization，近端策略优化）**和 **GRPO（Group Relative Policy Optimization，组相对策略优化）**是本仓库实现的策略优化方法。

**策略（policy）**在强化学习中就是产生回答的模型。不同阶段通常使用同一个模型结构，但目标函数、数据组织和参与训练的参数可能不同。

## 1. Tokenizer：字符串如何变成 id

读 `model/tokenizer.json`、`model/tokenizer_config.json`，再读 `trainer/train_tokenizer.py` 的 `get_texts` 与 `train_tokenizer`。

**Tokenizer（分词器）**负责把字符串切成 token，并将每个 token 映射成整数 id；模型不直接读写字符串。**词表（vocabulary）**是 token 与 id 的固定映射表。**BPE（Byte Pair Encoding，字节对编码）**是一种学习子词词表的方法：从字节片段开始，反复合并语料里常见的相邻片段。**特殊 token**承担结构控制：BOS 表示序列起点，EOS 表示序列结束，PAD 是为 batch 对齐长度而补入的占位符，对话边界 token 标识消息开始/结束或角色。

重点思考：

- `trainer/train_tokenizer.py` 负责“学习词表”，不是训练语言模型本身。
- 词表中 id 的含义必须和模型 embedding 行保持一致；换了词表，即使模型张量尺寸相同，行对应的 token 也可能变了。
- 阅读 `model/tokenizer_config.json` 中的 chat template，观察 `system/user/assistant/tool` 如何变成模型输入序列。

## 2. 模型：token id 如何变成下一个 token 的概率

按下面顺序读 `model/model_minimind.py`：

1. `MiniMindConfig`
2. `RMSNorm`、`precompute_freqs_cis`、`apply_rotary_pos_emb`
3. `Attention`
4. `FeedForward`、`MOEFeedForward`
5. `MiniMindBlock`、`MiniMindModel`
6. `MiniMindForCausalLM.forward`

**Embedding（词嵌入）**是按 token id 查表得到的可训练向量。**隐状态（hidden state）**是 token 向量经过模型层后、带有上下文信息的表示。这里的主要形状是 `input_ids [B,S]` → `hidden_states [B,S,H]`，其中 `B` 是 batch 样本数、`S` 是序列长度、`H` 是隐藏维度。

MiniMind 是 **Decoder-only Transformer（仅解码器 Transformer）**：它没有单独的编码器堆栈，依靠**因果遮罩（causal mask）**屏蔽当前位置之后的 token，确保预测时不偷看未来。它完成**自回归生成**：每一步只预测下一个 token，生成后把它接到上下文末尾，再预测下一个。

**注意力（attention）**让当前位置按相关性从前文汇集信息。代码中的 Query（Q，查询向量）表示当前位置要找什么，Key（K，键向量）表示一个位置如何被匹配，Value（V，值向量）是匹配后要汇总的内容。多个并行的子空间称为**注意力头（head）**。**GQA（Grouped Query Attention，分组查询注意力）**让多个 Q 头共享较少的 K/V 头，从而减少生成时缓存占用。**KV cache（键值缓存）**保存已生成前缀的 K/V，避免每生成一个 token 就重算整个前缀。

**RoPE（Rotary Positional Embedding，旋转位置编码）**通过旋转 Q/K 向量编码位置信息，让注意力分数包含 token 间的位置关系；YaRN 是把 RoPE 扩展到更长上下文的一种频率缩放方法。**RMSNorm**用向量均方根归一化特征尺度。**残差连接（residual connection）**把子层输入直接加回输出，为信息和梯度提供短路径；Pre-Norm 表示先归一化再进入子层。

**SwiGLU**是门控前馈网络：一支经过 SiLU 激活后与另一支逐元素相乘，再投影回隐藏维度。它在每个 token 内变换特征，注意力则负责 token 之间的信息混合。

**MoE（Mixture of Experts，专家混合）**用路由器为每个 token 选择少量专家前馈网络计算。路由器输出选择分数，top-k 选出本 token 激活的专家；稀疏激活降低每个 token 的计算量，但模型仍保存所有专家参数。**辅助负载均衡损失（auxiliary loss）**鼓励 token 不要都挤到少数专家。

`MiniMindForCausalLM` 最后把隐状态投影到词表大小 `V`，得到 **logits**（未归一化分数），形状为 `[B,S,V]`。**softmax**通过指数归一化把 logits 转成总和为 1 的词表概率分布。训练时使用**交叉熵（cross entropy）**衡量预测分布与正确 token 的差异。位置 `t` 的 logits 预测位置 `t+1`，所以 logits 与 labels 必须错位一格。

## 3. 数据：哪些 token 参与 loss

阅读 `dataset/lm_dataset.py`，顺序是 `PretrainDataset` → `SFTDataset` → `DPODataset` → `RLAIFDataset` → `AgentRLDataset`。

PyTorch 的 **Dataset** 定义如何读取一条样本；**DataLoader** 再负责取多条样本组成 batch。**label（标签）**在语言模型训练里是希望模型预测的 token id。**padding（补齐）**把不同长度序列补到相同长度；`-100` 是本仓库交叉熵使用的 ignore index（忽略标签），这种位置不计入 loss。

### 预训练与 SFT 数据

- `PretrainDataset.__getitem__`：读取一段普通文本，编码后加 BOS/EOS，再补齐；除 PAD 外的文本 token 都能产生 next-token 监督。
- `SFTDataset.create_chat_prompt`：用 **chat template（对话模板）**把角色、轮次和工具定义序列化成训练文本。
- `SFTDataset.generate_labels`：仅对 assistant 内容和结束 token 开启标签。user/system/tool 内容用于提供条件，不需要模型复述。
- 输入标签先按原序列位置构造；模型 forward 负责右移一位，将“位置 t 的上下文”与“位置 t+1 的目标 token”配对。

**SFT（Supervised Fine-Tuning，监督微调）**是用示范输入/回答继续训练模型。它仍然用 next-token 交叉熵，变化的是对话格式和哪些标签参与 loss。训练模板与推理模板不匹配，会导致模型看到不同的分隔符和角色结构。

### 偏好与在线 RL 数据

- `DPODataset`：同一个对话上下文准备 `chosen`（偏好回答）和 `rejected`（较差回答），并为 assistant 回答生成 mask。
- `RLAIFDataset`：只准备 prompt，答案会在线生成。
- `AgentRLDataset`：准备消息历史、可用工具和 ground truth（标准答案，用于自动验证任务结果）。

## 4. 预训练：一次参数更新从哪里来

先读 `trainer/train_pretrain.py::train_epoch`，再读文件底部的 main 初始化，随后读 `trainer/trainer_utils.py` 的 `init_model`、`get_lr`、`lm_checkpoint`。

**loss（损失）**是用于优化的标量目标，训练试图让它下降。**梯度（gradient）**是 loss 对参数的导数，告诉优化器每个参数应该向哪个方向调整。一个训练 step（一步）依次做：

1. 取 batch 并搬到训练设备。
2. 前向传播：模型计算 logits 和交叉熵；MoE 还计算路由辅助损失。
3. 反向传播：根据 loss 计算每个参数的梯度。
4. 梯度累积：多个小 batch 的梯度相加，模拟更大的有效 batch。
5. 梯度裁剪：限制梯度整体大小，缓解数值突然变大的不稳定。
6. AdamW 更新：根据梯度和动量统计量修改模型参数。

**优化器（optimizer）**负责把梯度转换为参数更新；**AdamW**是常用的自适应梯度优化器，并将权重衰减与梯度更新解耦。**学习率（learning rate）**控制每次更新幅度；`get_lr` 中的余弦衰减让它随训练逐渐降低。**epoch**表示完整遍历一次训练集，step 表示一个 batch 的处理/更新循环。

**混合精度（mixed precision）**用低精度执行部分运算，以节省显存并提升吞吐；float16 可能产生过小梯度，因此使用 **GradScaler（梯度缩放器）**先放大梯度、更新前再还原。**DDP（Distributed Data Parallel，分布式数据并行）**让多个进程处理不同数据分片、持有模型副本并同步梯度。

区分两种保存物：推理 checkpoint 主要是模型参数；resume checkpoint 还保存优化器、梯度缩放器和训练位置，才能较完整地恢复训练。

## 5. SFT 与 LoRA：从续写到指令回答

对比 `trainer/train_full_sft.py` 与 `trainer/train_pretrain.py`。训练循环很相似，能力变化主要来自 SFT 对话数据和 assistant-only label mask。

再读 `model/model_lora.py` 与 `trainer/train_lora.py`。**LoRA（Low-Rank Adaptation，低秩适配）**将权重变化近似成两个窄矩阵的乘积，在训练时冻结大部分基础权重，只训练低秩增量。`rank` 控制增量子空间大小；保存 adapter（适配器）只保存增量，merge（合并）则把增量加回基础矩阵，导出单一权重。

## 6. 蒸馏与 DPO：模仿分布和学习偏好

### 蒸馏

读 `trainer/train_distillation.py::distillation_loss` 和 `train_epoch`。**知识蒸馏**让 student（学生模型）拟合 teacher（教师模型）的输出。教师的概率分布是**软目标**：不仅指出正确 token，也表达其他 token 的相对可能性。**温度**调节 softmax 分布的平滑程度；**KL 散度**度量学生和教师概率分布的差异。训练可以混合硬标签交叉熵与软目标蒸馏损失。

### DPO

读 `DPODataset`，再读 `trainer/train_dpo.py` 的 `logits_to_log_probs`、`dpo_loss`、`train_epoch`。

**DPO（Direct Preference Optimization，直接偏好优化）**直接从偏好对优化策略，不额外训练奖励模型。**log probability（对数概率）**把序列 token 概率的连乘写成对数概率求和。函数先分别计算 policy（待更新策略）和 reference（冻结参考策略）对 chosen/rejected 回答的序列对数概率，再比较两者的偏好差值。此实现把回答 token 的 log-prob 求和，因此回答长度也会影响序列分数；读代码时留意 mask 排除提示词但没有按回答长度归一化。policy 比 reference 更偏向 chosen 时损失降低；`beta` 控制偏好目标强度及相对参考策略的约束。

## 7. 在线强化学习：PPO、GRPO 与 Agent

先读 `trainer/rollout_engine.py`，再按 `train_grpo.py` → `train_ppo.py` → `train_agent.py` 阅读。

**强化学习（RL）**让模型从行动结果的奖励中学习，而非只拟合固定 token 标签。**Rollout（轨迹采样）**指策略实际生成回答并记录过程；**轨迹（trajectory）**包含 prompt、生成 token、采样概率和奖励。**Reward（奖励）**是对回答/任务结果的标量评分。奖励模型或规则可以对整条回答打分；训练算法再把该分数变成 token 级更新信号。

### GRPO

**GRPO（Group Relative Policy Optimization，组相对策略优化）**针对同一个 prompt 采样多条回答，在组内比较奖励。**Advantage（优势）**是回答奖励相对基线高低的估计；组内标准化后，高于组均值为正，低于组均值为负。

`train_grpo.py` 主线：rollout 采样 → 奖励打分 → 组内标准化 advantage → 重算 policy 和 reference 的 token log-prob → 策略损失 → 反向更新。**重要性采样比（importance ratio）**是新策略概率除以采样时旧策略概率，用于修正“样本由旧策略生成、现在由新策略优化”的差异。**PPO clipping（概率比裁剪）**限制比率变化。**KL（Kullback–Leibler）散度**衡量两个概率分布的差异；KL 正则把新策略约束在 reference 附近。此脚本还支持 CISPO，即裁剪 token 重要性权重的策略目标。

### PPO

**PPO（Proximal Policy Optimization，近端策略优化）**也限制新旧策略差异，但显式训练 **Critic（价值模型）**估计每个状态未来的回报。**Actor**是产生回答的策略，Critic 是价值基线；**reference policy**另作冻结参考模型，不能与 Critic 混为一谈。

读 `train_ppo.py::ppo_train_epoch` 时重点看：

1. Actor rollout 生成回答，奖励函数给整条回答评分。
2. Critic 为回答的 token 位置估计 value（状态价值）。
3. 最终奖励放在轨迹末端，再向前分配信用。
4. **GAE（Generalized Advantage Estimation，广义优势估计）**用时间差分残差的折扣和估计每个 token 的 advantage。时间差分残差是“即时奖励 + 下一状态价值 − 当前状态价值”；`gamma` 是未来奖励折扣，`lam` 控制估计的平滑程度。
5. PPO 多轮更新 Actor 与 Critic，并用 KL 监控/约束策略偏移。

### Agentic RL

读 `train_agent.py::rollout_single` → `rollout_batch` → `calculate_rewards` → `rl_train_epoch`。**Agent（智能体）**通过“观察—行动—接收反馈”与环境多轮交互。本项目中，环境包括工具调用和工具返回。

模型生成的工具请求是策略 action（动作）；工具执行结果是 observation（观察）。工具结果会进入后续上下文，但它不是模型生成的动作，不应计入策略 token 概率损失。奖励可根据工具调用是否合法、参数是否匹配、ground truth 是否命中，以及最终回答质量计算。

## 8. 推理、评估、服务与转换

- `eval_llm.py`：命令行加载 checkpoint 并交互生成，用于人工检查。
- `scripts/eval_toolcall.py`：验证工具请求能否解析、执行并得到合理回答。
- `scripts/convert_model.py`：在 MiniMind 原生与 Transformers/兼容格式之间转换权重和配置。
- `scripts/serve_openai_api.py`：把模型封装成兼容 OpenAI 格式的 HTTP API。
- `scripts/chat_api.py`：调用聊天 API 的最小客户端示例。
- `scripts/web_demo.py`：Streamlit 浏览器聊天界面。

推理时 **decoding（解码）**是从 logits 概率分布中选择下一个 token 的过程；**temperature（温度）**是采样前对 logits 缩放的参数，较低时分布更尖锐、输出更确定，较高时分布更平坦、随机性更大。训练和推理需要使用相同 tokenizer 与兼容的 chat template。

## 文件职责速览

- `model/model_minimind.py`：模型配置、Transformer 主干、MoE 和因果语言模型 forward。
- `model/model_lora.py`：LoRA 模块以及加载、保存、合并。
- `model/__init__.py`、`dataset/__init__.py`：包入口说明，无训练逻辑。
- `dataset/lm_dataset.py`：各训练阶段的数据读取、文本模板和标签 mask。
- `trainer/train_tokenizer.py`：BPE tokenizer 训练示例。
- `trainer/train_pretrain.py`、`train_full_sft.py`、`train_lora.py`：预训练、全参 SFT、LoRA SFT。
- `trainer/train_distillation.py`、`train_dpo.py`：蒸馏与偏好优化。
- `trainer/rollout_engine.py`、`train_ppo.py`、`train_grpo.py`、`train_agent.py`：在线生成、策略优化和多轮工具任务。
- `trainer/trainer_utils.py`：训练公共组件。
- `eval_llm.py`、`scripts/*`：推理、评估、服务、Web UI 和模型转换。

## 阅读时的自检问题

- 对任一 batch，`input_ids`、`labels`、`attention_mask` 的形状分别是什么？
- `logits[:, t]` 对应预测哪个 token？哪些 label 被 `-100` 忽略？
- 这段代码用的是采样时策略、新策略，还是 reference？
- loss 是按 token 求和还是平均？padding 和 prompt 是否被 mask？
- resume checkpoint 保存了哪些参数以外的状态？

运行训练前还需按 README 下载/准备数据集；数据文件并未全部包含在仓库中。
