"""【阅读路线 第 4 站】预训练入口：从文本数据训练模型预测下一个 token。

到这里，你已经知道模型怎样算出 loss（第 2 站），数据怎样决定监督哪些 token（第 3 站）。
这一站回答：“有了 loss 之后，参数究竟是怎样被一步步更新的？”
这是所有训练阶段共用的骨架，后面 SFT / LoRA / 蒸馏 / DPO 的训练循环几乎都是它的变体，
读懂这里，之后只需要关注“loss 怎么换了”。

建议阅读顺序：
1. 先读 train_epoch：一个 batch 的完整生命周期
   取数据 → 设学习率 → 混合精度前向算 loss → 反向传播求梯度 →（累积够了）裁剪梯度 → 优化器更新 → 清零梯度
   → 定期打印日志 / 保存 checkpoint。
2. 再读文件底部 __main__：训练所需的各个组件（分布式环境、模型、数据、优化器、混合精度、续训状态）
   是怎样被构造出来并交给 train_epoch 的。
3. 需要时跳到 trainer_utils.py 看 get_lr、init_model、lm_checkpoint 的细节。

预训练的“能力来源”：海量无标注文本 + 每个 token 都参与 next-token 预测。模型在这一阶段
学到语言规律与世界知识，但还不会“按对话格式回答问题”，那是第 5 站 SFT 的任务。
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """遍历一个 epoch 的 batch 并执行前向、反向和参数更新。

    术语：epoch 是完整遍历训练集一次；step（这里指 micro-step）是处理其中一个 batch；
    batch 是同时送入模型的一组样本；optimizer（优化器）根据梯度更新权重。
    iters 是本 epoch 的总 step 数；start_step 是续训时已跳过的 step 数。
    """
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # ① 取数据：input_ids / labels 都是 [B,S] 的 int64 张量（B=batch_size，S=max_seq_len）。
        #    DataLoader 在 CPU 上组 batch，这里搬到 GPU；pin_memory=True 让这次拷贝更快。
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        # ② 设学习率：全局进度 = 已完成的 epoch × 每 epoch 步数 + 当前步，按余弦曲线计算后
        #    写入优化器的每个参数组（param_group 可以让不同参数用不同学习率，这里只有一组）。
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ③ 前向传播（在混合精度上下文中）。
        # autocast：进入这个上下文后，PyTorch 会按算子类别自动选择精度——矩阵乘/卷积这类
        # 对精度不敏感且耗时的算子用 bf16/fp16 执行（更快、激活占显存更少），而 softmax、
        # 归一化、求和、loss 等对精度敏感的算子仍用 fp32。参数的主副本始终是 fp32，
        # autocast 只是在计算时临时转换，所以不用手动改模型的 dtype。
        with autocast_ctx:
            # forward 内部：logits 与错位一格的 labels 做交叉熵，得到所有有效 token 的平均 NLL（res.loss）；
            # MoE 模型还会返回负载均衡 aux_loss（dense 模型为 0），两者相加即训练目标。
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            # 梯度累积：每个 micro-batch 的 loss 先除以累积步数 N。因为梯度在 N 次 backward 中是
            # 逐次“相加”的，除以 N 后累加得到的正好是 N 个 micro-batch 的平均梯度，
            # 等价于一次用 N 倍 batch_size 训练（有效 batch = batch_size × N × 卡数），但显存只需一份。
            loss = loss / args.accumulation_steps

        # ④ 反向传播：loss.backward() 沿前向时记录的计算图，用链式法则算出 loss 对每个
        #    requires_grad 参数的梯度，并“累加”到 param.grad 上（不是覆盖，这正是梯度累积能成立的原因）。
        # scaler.scale(loss)：fp16 的可表示范围很窄（最小正规数约 6e-5），很多梯度小到会下溢成 0。
        #    GradScaler 先把 loss 乘一个大系数 S（如 65536），梯度随之被放大 S 倍而不下溢，更新前再除回去。
        #    bf16 的指数位与 fp32 相同、不会下溢，所以本脚本只在 --dtype float16 时启用 scaler，
        #    默认 bf16 下 scaler(enabled=False) 的各方法都是直通的空操作。
        # DDP 下：backward 过程中 DDP 注册的钩子会把各卡的梯度按桶（bucket）做 all-reduce 求平均，
        #    与反向计算重叠进行；backward 返回时每张卡上的梯度已经完全相同，因此各卡独立做 optimizer.step
        #    后参数仍保持一致。（这里每个 micro-step 都会通信；更省的写法是在累积期间用 model.no_sync()。）
        scaler.scale(loss).backward()

        # ⑤ 累积满 N 个 micro-batch（或到了 epoch 最后一个 batch，防止尾部梯度被丢弃）才真正更新一次参数。
        if step % args.accumulation_steps == 0 or step == iters:
            # unscale_：把梯度除以 S 还原成真实尺度。必须在裁剪之前做，否则裁剪阈值是在“放大后的梯度”上比较的。
            scaler.unscale_(optimizer)
            # 梯度裁剪：把所有参数的梯度看成一个大向量，计算其 L2 范数 ‖g‖；若 ‖g‖ > grad_clip，
            # 就整体乘以 grad_clip/‖g‖。方向不变、只缩短长度，防止偶发的坏 batch 产生巨大梯度把参数“推飞”
            # （表现为 loss 突然尖峰甚至 NaN）。返回值是裁剪前的范数，常被记录用于监控训练稳定性。
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # scaler.step(optimizer)：先检查梯度中是否有 inf/NaN（fp16 溢出的信号）；
            #   若有，跳过本次更新（参数不变）；若没有，调用 optimizer.step()。
            # optimizer.step() 即 AdamW 更新，对每个参数 θ、梯度 g：
            #   m = β1·m + (1-β1)·g            （一阶矩：梯度的滑动平均，起“动量”作用，默认 β1=0.9）
            #   v = β2·v + (1-β2)·g²           （二阶矩：梯度平方的滑动平均，默认 β2=0.999）
            #   m̂, v̂ = 偏差修正后的 m, v       （早期 m、v 从 0 起步偏小，除以 1-β^t 修正）
            #   θ = θ - lr·( m̂/(√v̂+ε) + wd·θ ）
            #   其中 m̂/√v̂ 让每个参数拥有“自适应步长”：梯度一直很大的参数步长被压小，稀疏的参数步长相对放大；
            #   wd·θ 是解耦的权重衰减（AdamW 与 Adam+L2 的区别），torch 默认 weight_decay=0.01。
            #   m、v 各占一份与参数同尺寸的 fp32 显存，这就是“优化器状态 ≈ 2 倍参数量”的来源。
            scaler.step(optimizer)
            # scaler.update()：动态调整缩放系数 S。出现 inf 则 S 减半；连续若干步无溢出则 S 翻倍。
            scaler.update()

            # 清空梯度，为下一个累积窗口做准备。set_to_none=True 直接释放 .grad 张量（设为 None）
            # 而不是填 0，省一次写显存操作，也降低显存峰值。
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            # 日志中的 loss 只是“当前这一个 micro-batch”的 loss（乘回 N 还原尺度），
            # 不是整个累积窗口或整个 log_interval 的平均值，因此曲线会有一定噪声。
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            # eta_min：按已用时间/已完成步数估算本 epoch 剩余时间（分钟），不是“已用时间”。
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # ⑥ 定期保存（只在主进程）：先写推理用的纯权重 .pth，再调用 lm_checkpoint 写续训用的完整状态。
        # model.eval()/model.train() 切换是为了保存期间关闭 dropout 等训练行为（保存本身不做前向，属于惯例写法）。
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        # 及时删除本步张量引用，让显存（尤其是 [B,S,V] 的 logits）尽早被缓存分配器回收。
        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--seed', default=42, type=int, help="随机种子（DDP下每个rank为seed+rank，每轮为seed+epoch）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # 阅读提示：这里按“运行环境 → 模型/数据/优化器 → 恢复状态 → 并行包装 → epoch”
    # 组织训练生命周期。要理解一个训练 step，可先看 train_epoch，再回到这里看其依赖如何构造。
    # 后续所有 train_*.py 的 __main__ 都沿用这 9 个步骤，差别只在第 5 步构造哪些模型/数据集。
    #
    # DDP（Distributed Data Parallel，分布式数据并行）：N 张卡各跑一个进程、各持一份完整模型副本；
    # 每个进程读取不同的数据分片，前向/反向各自独立，反向时把梯度 all-reduce 求平均，
    # 于是每步等效于用 N 倍的 batch 训练。单卡运行时不会启用通信组，下面的 DDP 相关分支都会跳过。
    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 每个 rank 用不同种子（seed+rank），避免各卡 dropout 等随机行为完全相同。
    # 模型初始权重不会因此不一致：DDP 构造时会把 rank 0 的参数广播给所有卡。
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # from_resume=1 时尝试读取 ../checkpoints 下的 *_resume.pth；不存在则返回 None，按全新训练处理。
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None

    # ========== 3. 设置混合精度 ==========
    # bf16 与 fp16 都是 16 位：fp16 = 5 位指数 + 10 位尾数（精度高、范围窄，需要 GradScaler 防下溢）；
    # bf16 = 8 位指数 + 7 位尾数（范围与 fp32 相同、精度较低，通常无需缩放）。
    # 支持 bf16 的 GPU（Ampere 及以后）上，LLM 训练基本都用 bf16。CPU 上不启用 autocast。
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    # 实验追踪（loss/lr 曲线可视化）。实际导入的是 SwanLab（API 与 wandb 兼容），只在主进程初始化；
    # 续训时用保存的 run id 接着画同一条曲线。
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    # 预训练默认 from_weight='none'：从随机初始化开始。
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # DistributedSampler：把数据集下标切成 world_size 份，每个 rank 只拿到自己那 1/N（互不重叠），
    # 这就是“数据并行”里“数据”被并行的方式。它内部用 (seed + epoch) 打乱，所以每个 epoch
    # 要调用 set_epoch(epoch)，否则每轮顺序都一样。
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # GradScaler 只在 fp16 时启用（原理见 train_epoch 中的注释）。
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # 优化器持有对模型参数的引用，并在内部为每个参数创建 m、v 两份状态（首次 step 时惰性创建）。
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    # 包装顺序很重要：必须先加载完权重/优化器状态，再包装。包装后原模型在 model.module（DDP）
    # 或 model._orig_mod（compile）里，这也是保存时要“剥壳”的原因。
    # torch.compile：首次运行时追踪计算图并用 Triton 生成融合 kernel，减少 Python 开销与显存读写，
    # 首个 step 会因编译明显变慢，之后提速。
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    # DDP 包装时：从 rank 0 广播参数与 buffer 保证各卡起点一致，并给每个参数注册梯度 all-reduce 钩子。
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # 数据顺序：DDP 时由 DistributedSampler 按 epoch 打乱并分片；单卡时用 randperm 生成打乱的下标。
        # 两者都以 epoch 为种子，因此续训时能重现同样的顺序，SkipBatchSampler 才能精确跳过已训练的 batch。
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(args.seed + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        # num_workers 个子进程并行执行 Dataset.__getitem__（tokenize 等 CPU 工作），与 GPU 计算重叠；
        # pin_memory 把 batch 放进锁页内存，使 .to(device) 的主机→显卡拷贝更快。
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
