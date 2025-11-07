import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import argparse
import time
from math import pi, cos
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import Transformer, make_std_masks
from utils import set_seed, save_checkpoint, compute_perplexity, generate_sample
from utils import create_src_mask, create_tgt_mask
import matplotlib.pyplot as plt

def train_one_epoch(model, dataloader, optimizer, device, global_step, total_steps,
                    pad_idx=0, clip_grad=1.0, bos_idx=2, label_smoothing=0.1):
    """
        训练一个 epoch 的函数
        支持标签平滑(label smoothing)，梯度裁剪，以及 token 级别准确率计算
        """
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=label_smoothing)
    total_tokens = 0
    correct_tokens = 0

    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)
        batch_size, seq_len = tgt.size()
        # 构造带有起始符号 BOS 的目标输入序列
        bos = torch.full((batch_size,1), fill_value=bos_idx, dtype=torch.long, device=device)
        tgt_input = torch.cat([bos, tgt[:, :-1]], dim=1)
        # 创建注意力 mask
        src_mask = create_src_mask(src, pad_idx)
        tgt_mask = create_tgt_mask(tgt_input, pad_idx)

        # 前向传播
        logits = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
        # 计算交叉熵损失
        loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))

        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

        optimizer.step()


        total_loss += float(loss.item()) * src.size(0)

        # 粗略的 token-level 准确率计算
        preds = logits.argmax(dim=-1)
        mask = (tgt != pad_idx)
        correct = (preds == tgt) & mask
        correct_tokens += correct.sum().item()
        total_tokens += mask.sum().item()

        global_step += 1

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    return avg_loss, accuracy, global_step

def evaluate(model, dataloader, device, pad_idx=0, bos_idx=2):
    """
       在验证集上评估模型
       不进行梯度更新，只计算损失和准确率
       """
    model.eval()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    total_tokens = 0
    correct_tokens = 0
    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)
            batch_size, seq_len = tgt.size()
            bos = torch.full((batch_size,1), fill_value=bos_idx, dtype=torch.long, device=device)
            tgt_input = torch.cat([bos, tgt[:, :-1]], dim=1)
            src_mask = create_src_mask(src, pad_idx)
            tgt_mask = create_tgt_mask(tgt_input, pad_idx)

            logits = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))
            total_loss += float(loss.item()) * src.size(0)

            preds = logits.argmax(dim=-1)
            mask = (tgt != pad_idx)
            correct = (preds == tgt) & mask
            correct_tokens += correct.sum().item()
            total_tokens += mask.sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    return avg_loss, accuracy

def set_lr_by_step(optimizer, base_lr, max_lr, global_step, warmup_steps, total_steps):
    """
    学习率调度策略：
    前 warmup_steps 线性升温，
    之后使用余弦退火下降到接近 0。
    """
    if global_step < warmup_steps and warmup_steps > 0:
        lr = base_lr + (max_lr - base_lr) * (global_step / warmup_steps)
    else:
        # progress in [0,1]
        if total_steps - warmup_steps <= 0:
            progress = 1.0
        else:
            progress = (global_step - warmup_steps) / (total_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
        # cosine decay to a small value (not exactly zero to avoid dead LR)
        lr = 0.5 * max_lr * (1 + cos(pi * progress))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def plot_and_save_curves(epochs_list, train_losses, val_losses, train_accs, val_accs, val_perplexities, lrs, save_dir):
    """
        绘制并保存训练曲线：
        - 学习率变化
        - 训练/验证损失
        - 准确率
        - 困惑度
        """

    os.makedirs(save_dir, exist_ok=True)

    # 学习率曲线
    plt.figure(figsize=(8,6))
    plt.plot(epochs_list, lrs, color='orange', label='Learning Rate')
    plt.title('LR (Warmup + Cosine)')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "learning_rate_curve.png"), dpi=200)
    plt.close()

    # 损失曲线
    plt.figure(figsize=(8,6))
    plt.plot(epochs_list, train_losses, color='orange', label='Train Loss')
    plt.plot(epochs_list, val_losses, color='skyblue', label='Val Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=200)
    plt.close()

    # 准确率曲线
    plt.figure(figsize=(8,6))
    plt.plot(epochs_list, train_accs, color='orange', label='Train Accuracy')
    plt.plot(epochs_list, val_accs, color='skyblue', label='Val Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"), dpi=200)
    plt.close()

    # 困惑度曲线
    plt.figure(figsize=(8,6))
    plt.plot(epochs_list, val_perplexities, color='orange', label='Val Perplexity')
    plt.title('Validation Perplexity')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "perplexity_curve.png"), dpi=200)
    plt.close()

"""
    主训练入口：
    - 加载数据
    - 构建模型
    - 定义优化器与学习率调度
    - 执行训练与验证循环
    """
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-archive', type=str,
                        default='/root/bigModel/data/wikitext-2.zip',
                        help='path to local wikitext-2 archive (zip or tar.gz)')
    parser.add_argument('--seq-len', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--enc-layers', type=int, default=3)
    parser.add_argument('--dec-layers', type=int, default=3)
    parser.add_argument('--d-ff', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=3e-4)  # base lr (start of warmup)
    parser.add_argument('--max-lr', type=float, default=3e-4)  # peak lr after warmup
    parser.add_argument('--warmup-epochs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-dir', type=str, default='results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--clip-grad', type=float, default=1.0)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--use-relative-pos', action='store_true',
                        help='use T5-style relative position bias in self-attention')
    parser.add_argument('--rel-pos-num-buckets', type=int, default=32)
    parser.add_argument('--rel-pos-max-distance', type=int, default=128)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    print("Loading data...")
    train_loader, val_loader, stoi, itos, vocab_size = get_dataloaders(
        args.data_archive, seq_len=args.seq_len, batch_size=args.batch_size, seed=args.seed
    )

    device = torch.device(args.device)
    print("Using device:", device)

    # 构建 Transformer 模型
    model = Transformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        d_ff=args.d_ff,
        use_relative_pos=args.use_relative_pos,
        rel_pos_num_buckets=args.rel_pos_num_buckets,
        rel_pos_max_distance=args.rel_pos_max_distance
    ).to(device)

    # AdamW 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # 计算总步数和 warmup 步数
    num_batches = len(train_loader)
    total_steps = args.epochs * num_batches
    warmup_steps = args.warmup_epochs * num_batches

    # 日志记录
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    val_perplexities = []
    lrs = []
    epochs_list = []

    best_val = float('inf')
    global_step = 0

    print("Start training for {} epochs, {} steps total ({} warmup steps)".format(args.epochs, total_steps, warmup_steps))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        epoch_correct = 0

        criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=args.label_smoothing)

        # 每个 batch 手动调节学习率
        for batch_idx, (src, tgt) in enumerate(train_loader):
            lr_now = set_lr_by_step(optimizer, base_lr=args.lr, max_lr=args.max_lr,
                                    global_step=global_step, warmup_steps=warmup_steps, total_steps=total_steps)

            src = src.to(device)
            tgt = tgt.to(device)
            batch_size, seq_len = tgt.size()
            bos = torch.full((batch_size,1), fill_value=2, dtype=torch.long, device=device)
            tgt_input = torch.cat([bos, tgt[:, :-1]], dim=1)
            src_mask = create_src_mask(src, 0)
            tgt_mask = create_tgt_mask(tgt_input, 0)

            logits = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)

            optimizer.step()

            epoch_loss += float(loss.item()) * src.size(0)

            preds = logits.argmax(dim=-1)
            mask = (tgt != 0)
            correct = (preds == tgt) & mask
            epoch_correct += correct.sum().item()
            epoch_tokens += mask.sum().item()

            global_step += 1

        train_loss = epoch_loss / len(train_loader.dataset)
        train_acc = epoch_correct / epoch_tokens if epoch_tokens > 0 else 0.0

        val_loss, val_acc = evaluate(model, val_loader, device, pad_idx=0, bos_idx=2)
        val_ppl = compute_perplexity(val_loss)

        current_lr = optimizer.param_groups[0]['lr']

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        val_perplexities.append(val_ppl)
        lrs.append(current_lr)
        epochs_list.append(epoch)

        t1 = time.time()

        print(f"Epoch {epoch} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | val ppl: {val_ppl:.2f} | "
              f"train acc: {train_acc:.4f} | val acc: {val_acc:.4f} | lr: {current_lr:.6g} | time: {t1-t0:.1f}s")

        ckpt_path = os.path.join(args.save_dir, f"model_epoch{epoch}.pt")
        torch.save({'model': model.state_dict(),
                    'optim': optimizer.state_dict(),
                    'stoi': stoi,
                    'itos': itos,
                    'args': vars(args),
                    'epoch': epoch}, ckpt_path)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({'model': model.state_dict(),
                        'optim': optimizer.state_dict(),
                        'stoi': stoi,
                        'itos': itos,
                        'args': vars(args),
                        'epoch': epoch}, os.path.join(args.save_dir, "best_model.pt"))


        try:
            val_iter = iter(val_loader)
            src_vis, tgt_vis = next(val_iter)
            src_vis = src_vis.to(device)[:1]
            tgt_vis = tgt_vis.to(device)[:1]
            bos_idx = 2
            pad_idx = 0
            bos = torch.full((1, 1), bos_idx, dtype=torch.long, device=device)
            tgt_in = torch.cat([bos, tgt_vis[:, :-1]], dim=1)
            src_mask_vis = create_src_mask(src_vis, pad_idx)
            tgt_mask_vis = create_tgt_mask(tgt_in, pad_idx)
            with torch.no_grad():
                _ = model(src_vis, tgt_in, src_mask=src_mask_vis, tgt_mask=tgt_mask_vis)
            attn = model.decoder.layers[-1].self_attn.last_attn  # [B,H,T,S]
            if attn is not None:
                attn_2d = attn[0, 0]
                from utils import plot_attention
                attn_path = os.path.join(args.save_dir,
                                         f"attn_{'rel' if args.use_relative_pos else 'abs'}_epoch{epoch}.png")
                plot_attention(attn_2d, attn_path,
                               title=f"Decoder self-attn (head 0) - {'Relative' if args.use_relative_pos else 'Absolute'} - epoch {epoch}")
                print('Saved attention visualization to', attn_path)
        except Exception as e:
            print('Attention viz failed:', e)

        sample = generate_sample(model, stoi, itos, device, seed_text=None, max_len=40)
        print("Sample generation (greedy):", sample[:200])

    plot_and_save_curves(epochs_list, train_losses, val_losses, train_accs, val_accs, val_perplexities, lrs, args.save_dir)

    print("Training finished. Best val loss:", best_val)

if __name__ == "__main__":
    main()