"""【阅读路线 第 3 站】训练数据适配层：把文本或对话样本变成模型输入与监督标签。

上一站你看到：模型 forward 会把 logits 与右移一位的 labels 做交叉熵，并忽略 -100。
这一站要回答：“每个训练阶段到底让模型学习预测哪些 token？”——答案全部写在 labels / mask 里。
不同训练阶段的模型结构和 loss 公式几乎一样，真正决定“学到什么能力”的是这里的数据组织。

阅读顺序：PretrainDataset → SFTDataset → DPODataset → RLAIFDataset → AgentRLDataset。

PyTorch 约定：Dataset 负责“按下标取出一条样本”（__getitem__）；DataLoader 负责多进程读取、
把多条样本堆叠（collate）成 batch。这里每条样本都已补齐到固定长度，默认 collate 可直接堆叠。

MiniMind tokenizer 的特殊 token（见 model/tokenizer_config.json）：
- bos_token = <|im_start|>（id=1）：既当序列起点，也当对话中每条消息的开头；
- eos_token = <|im_end|>（id=2）：既当序列结束，也当每条消息的结尾，生成时遇到它就停；
- pad_token = <|endoftext|>（id=0）：仅用于补齐长度。
一段经 chat template 渲染后的对话大致如下（每条消息都是 <|im_start|>角色\\n内容<|im_end|>\\n）：

    <|im_start|>system\\n你是一个助手<|im_end|>\\n
    <|im_start|>user\\n你好<|im_end|>\\n
    <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n你好！有什么可以帮你？<|im_end|>\\n

工具返回（role=tool）会被模板渲染成 <|im_start|>user\\n<tool_response>...</tool_response><|im_end|>，
即在 token 层面以 user 身份出现。
"""
from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import load_dataset, Features, Sequence, Value
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pre_processing_chat(conversations, add_system_ratio=0.2):
    """数据增强：以 add_system_ratio 的概率在对话开头随机插入一条 system prompt。

    system prompt（系统提示）是约束助手身份/行为的开场消息。训练数据里有时带、有时不带、
    内容也不固定，模型才能在推理时对“有无 system、system 写了什么”都保持稳健，
    而不是只认得某一句固定开场白。
    """
    # tool use 数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations): return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    # 概率性添加system
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.2, remove_empty_think=None):
    """按概率删除空的 think 区段，让模型同时见过“有空 think 标签”和“完全没有 think 标签”两种格式。

    chat template 总会给 assistant 回复套上 <think>...</think>；对于没有思考内容的样本，
    渲染结果是固定的空壳 '<think>\\n\\n</think>\\n\\n'。若全部保留，模型会过度依赖这个空壳；
    若全部删除，又和推理时 open_thinking=0 注入空壳的格式不一致。这里默认 80% 删除、20% 保留。
    """
    # 以80%概率移除空思考标签
    if '<think>\n\n</think>\n\n' in prompt_content:
        if remove_empty_think is None:
            remove_empty_think = random.random() > empty_think_ratio
        if remove_empty_think:
            prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content

class PretrainDataset(Dataset):
    """自回归预训练数据：让模型根据左侧上下文预测文本中的下一个 token。

    每行 JSONL 是 {"text": "..."}。预训练没有“问题/答案”之分，文本里每个 token 都是监督信号，
    模型借此学习语言规律与世界知识。load_dataset('json') 使用 Apache Arrow 内存映射，
    大文件也不会一次性读进内存。
    """
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # 构造：[BOS] + 文本 token（截断到 max_length-2，给 BOS/EOS 留位置）+ [EOS] + [PAD]*n。
        # - BOS 给第一个文本 token 提供“前文”，EOS 让模型学会一段文本在哪里结束；
        # - 补齐到固定长度，方便 DataLoader 直接堆叠成 [B,S] 张量；
        # - labels 就是 input_ids 的拷贝（不在这里错位，错位由模型 forward 完成），
        #   只把 PAD 位置改成 -100，交叉熵就不会要求模型“预测补齐符”。
        # 注意：这里是一条样本一行、不足补齐，简单但会浪费一部分算力在 PAD 上；
        # 大规模预训练通常把多篇文本首尾相接“打包”（packing）成定长序列来避免浪费。
        sample = self.samples[index]
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels


class SFTDataset(Dataset):
    """把监督微调（SFT）对话格式化为模型输入，并仅监督 assistant 轮次。

    SFT 是 Supervised Fine-Tuning 的缩写，意思是用人工/合成的示范回答继续
    训练预训练模型。它仍然使用 next-token 预测，变化主要在对话模板和损失 mask。
    """
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 显式声明 schema：不同样本的字段可能缺失或类型不一（如 tools 有时是 list 有时是 str），
        # 统一声明成 string 可避免 Arrow 自动推断类型时报错；用到时再 json.loads。
        features = Features({'conversations': [{'role': Value('string'), 'content': Value('string'), 'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')}]})
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        # 用于在 token 序列中定位 assistant 回复的“起止标记”（token id 列表）：
        # bos_id = "<|im_start|>assistant\n" 的 token 序列 → assistant 回复从它之后开始；
        # eos_id = "<|im_end|>\n" 的 token 序列 → assistant 回复到它为止。
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        # 模型本身只认识一条扁平的 token 序列，并不知道什么是“多轮对话”。chat template 负责把
        # 结构化的 messages（role/content/tool_calls/tools）序列化成带角色标记的纯文本。
        # apply_chat_template 这一行内部：用 tokenizer_config.json 中的 Jinja2 模板渲染消息列表——
        #   - 若有 tools，把工具的 JSON schema 写进 system 消息，并附上“如何输出 <tool_call>”的说明；
        #   - assistant 消息统一套上 <think>{reasoning_content}</think>，tool_calls 渲染为 <tool_call>{json}</tool_call>；
        #   - role=tool 的结果包进 <tool_response>，并以 user 身份出现；
        #   - tokenize=False 只返回字符串；add_generation_prompt=False 表示不在末尾追加“<|im_start|>assistant\n”
        #     （训练数据已包含完整回复；推理/RL 时才需要追加，提示模型“轮到你说了”）。
        # 模板本身就是模型接口的一部分：训练与推理必须用同一模板，否则模型会看到陌生的分隔符。
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        # SFT 的核心：assistant-only loss mask。
        # 默认全部置为 -100（ignore_index），只在每段 assistant 回复内容
        # （含 <think> 思考段与 <tool_call> 工具调用）及其结尾 "<|im_end|>\n" 上开启损失。
        # 于是 system/user/tool 内容只作为条件（模型看得到，但不要求它预测/复述），
        # 模型学到的是“给定这样的上下文，应当怎样回答”，而不是“怎样模仿用户提问”。
        # 结尾的 <|im_end|> 也要监督：模型必须学会“回答完了就输出结束符”，否则推理时会停不下来。
        # "<|im_start|>assistant\n" 这个开头本身不监督——推理时它由模板(add_generation_prompt)提供。
        #
        # 算法：线性扫描，每遇到一次 bos_id 子序列，就向后找最近的 eos_id 子序列，
        # 把 [开头之后, 结尾标记末尾) 区间的 label 设为原 token id。多轮对话中每段 assistant 都会被标出。
        # 若回复被 max_length 截断、找不到结尾，则一直标到序列末尾。
        #
        # labels 在这里保持与 input_ids 同位置（labels[j]=input_ids[j]），不做错位；
        # 模型 forward 中 logits[:-1] 对 labels[1:]，所以 labels[j] 实际由位置 j-1 的输出来预测。
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        # 流水线：原始 messages →（随机 system）→ 模板渲染成文本 →（随机去掉空 think）
        # → tokenize 并截断 → PAD 补齐 → 生成 assistant-only labels。
        # PAD 位置不在任何 assistant 区间内，所以 label 自然是 -100。
        sample = self.samples[index]
        conversations = pre_processing_chat(sample['conversations'])
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)
        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class DPODataset(Dataset):
    """偏好数据适配器：把同一上下文下更好/较差的两条回答编码成配对样本（供第 6 站 DPO 使用）。

    每行数据形如 {"chosen": [messages...], "rejected": [messages...]}，两者的上下文相同、
    只有最后的 assistant 回复不同。与 SFT 不同，这里不再返回 -100 形式的 labels，
    而是直接返回错位好的 (x, y, mask)：DPO 需要自己从 logits 里取出每个 token 的 log-prob，
    不走模型内置的交叉熵。
    """
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # 每条偏好样本有 chosen/rejected 两段完整对话。两边采用同一模板、同一长度上限、
        # 同一种 assistant mask，且共用同一个“是否删除空 think”的随机结果，
        # 保证两者唯一的差别就是回答内容本身，这样比较概率才公平。
        sample = self.samples[index]
        chosen = sample['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = sample['rejected']  # 同上
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        remove_empty_think = random.random() > 0.2
        chosen_prompt = post_processing_chat(chosen_prompt, remove_empty_think=remove_empty_think)
        rejected_prompt = post_processing_chat(rejected_prompt, remove_empty_think=remove_empty_think)
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
        # 在数据侧提前完成错位（与模型 forward 里 logits[:-1]/labels[1:] 是同一件事）：
        # x = ids[:-1] 作为模型输入，y = ids[1:] 作为每个位置要预测的目标，二者长度都是 L-1；
        # mask 取 loss_mask[1:]，与 y 逐位对齐：mask[t]=1 表示“目标 y[t] 是 assistant 回复 token”。
        # 于是位置 t 的输出只在“下一个 token 属于回答”时才计入序列 log-prob。
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):
        # 与 SFTDataset.generate_labels 的扫描逻辑相同，只是输出 0/1 mask 而不是 -100/token id。
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):
    """在线强化学习（PPO/GRPO，第 7 站）的输入：只提供 prompt，回答由训练循环在线生成并评分。

    这是 RL 与 SFT/DPO 在数据上的根本区别：SFT/DPO 的回答是事先写好的静态数据（离线），
    RL 的回答由“当前正在训练的模型”现场生成（在线/on-policy），奖励也针对这些新回答实时计算。
    """
    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio  # 按概率开启 thinking
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        # conversations[:-1] 去掉数据里最后一条 assistant 回复（RL 不用它当标签）；
        # add_generation_prompt=True 在末尾追加 "<|im_start|>assistant\n" 及 think 前缀，
        # 让模型从“助手开始说话”的位置续写。open_thinking=True 时只追加 "<think>\n"，
        # 模型需要自己写出思考过程；否则追加空壳 "<think>\n\n</think>\n\n"，引导模型直接回答。
        # thinking_ratio 控制两种模式的混合比例。
        conversations = pre_processing_chat(conversations)
        use_thinking = random.random() < self.thinking_ratio
        return self.tokenizer.apply_chat_template(
            conversations[:-1],
            tokenize=False,
            open_thinking=use_thinking,
            add_generation_prompt=True
        )
    def __getitem__(self, index):
        # 返回的是字符串而非 token id：训练脚本会对整个 batch 统一做“左侧补齐”的 tokenize，
        # 因为生成是从序列右端续写，左侧补齐才能让所有样本的“续写起点”对齐在同一列。
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': ""
        }

class AgentRLDataset(Dataset):
    """Agentic RL（第 7 站 train_agent.py）的静态输入：历史消息、可用工具定义和 ground truth。

    这里保持 messages 为结构化 list 而不渲染成字符串：多轮 rollout 中每执行一次工具，
    都要把 assistant 输出和工具结果追加到 messages 里，再重新渲染模板。
    gt（ground truth）是期望在最终回答中出现的关键结果（如计算结果、汇率数值），用于规则化打分。
    """
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def parse_conversations(self, conversations):
        # 数据把最后一个 assistant 位置留空作为待续写槽位；去掉它后，
        # rollout 会根据前面的历史生成回答。ground truth 单独用于奖励验证。
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            messages.append(message)
        return messages[:-1], tools

    def __getitem__(self, index):
        sample = self.samples[index]
        messages, tools = self.parse_conversations(sample['conversations'])
        return {'messages': messages, 'tools': tools, 'gt': sample['gt']}


if __name__ == "__main__":
    pass
