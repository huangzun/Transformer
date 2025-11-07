# utils.py
import random
import math
import torch
import matplotlib.pyplot as plt


def set_seed(seed=42):#固定随机种子，保证实验可复现
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(state, path):#保存训练检查点
    torch.save(state, path)

def plot_losses(train_losses, val_losses, out_path):#绘制并保存训练/验证损失曲线
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def compute_perplexity(loss):#根据平均交叉熵损失计算困惑度
    return math.exp(loss)

def create_src_mask(src, pad_idx):
    """
        构造 Encoder 的 padding mask
        输入:
            src: [B, S]
        返回:
            [B,1,1,S]，pad 位置为 0，其余为 1
        """
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)  # [B,1,1,S]

def create_tgt_mask(tgt, pad_idx):
    """
        构造 Decoder 的综合 mask（padding + 因果 no-peak）
        输入:
            tgt: [B, T]
        返回:
            [B,1,T,T]，未来位置与 pad 位置屏蔽
        """
    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # [B,1,1,T]
    T = tgt.size(1)
    # 下三角为 True：只允许看到当前及之前的 token
    nopeak = torch.tril(torch.ones((1, 1, T, T), device=tgt.device)).bool()
    tgt_mask = pad_mask & nopeak
    return tgt_mask  # [B,1,T,T]

def plot_attention(attn, out_path, title=None):
    """
    可视化单头注意力矩阵
    参数:
        attn: [T_q, T_k] 的注意力矩阵（或 tensor/ndarray）
    """

    if torch.is_tensor(attn):
        attn = attn.detach().cpu().numpy()

    plt.figure()
    plt.imshow(attn, aspect="auto")
    plt.xlabel("Key positions")
    plt.ylabel("Query positions")
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def generate_sample(model, stoi, itos, device, seed_text=None, max_len=40):
    """
        基于当前模型做一个贪心解码示例（便于训练过程观察效果）
        说明：
            - 使用 encoder-decoder 结构：src 来自 seed（或随机词），decoder 以 <bos> 开始逐步生成
            - 贪心策略：每步选择概率最大的 token
        """
    model.eval()
    bos = stoi.get("<bos>", 2)
    pad = stoi.get("<pad>", 0)
    vocab_size = len(stoi)
    if seed_text is None:
        # 若未提供种子文本，则随机挑一个词作为起始
        seed_idx = random.randint(0, vocab_size - 1)
        seed_tokens = [seed_idx]
    else:
        # 简单空格分词并查词表，未知词映射到 <unk>
        seed_tokens = [stoi.get(t, stoi.get("<unk>", 1)) for t in seed_text.split()]
    src = torch.tensor([seed_tokens + [pad] * (max_len - len(seed_tokens))], dtype=torch.long, device=device)
    # greedy decode on decoder-only style: here we do encoder-decoder with src=src, dec input starts with bos
    with torch.no_grad():
        memory_mask = create_src_mask(src, pad)
        memory = model.encode(src, src_mask=memory_mask)
        out_ids = []
        prev = torch.tensor([[bos]], dtype=torch.long, device=device)
        for _ in range(max_len):
            tgt_mask = create_tgt_mask(prev, pad)
            logits = model.decode(prev, memory, src_mask=memory_mask, tgt_mask=tgt_mask)
            logits = model.out(logits)  # [1, T, V]
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            if next_token == stoi.get("<eos>", None):
                break
            out_ids.append(next_token)
            prev = torch.cat([prev, torch.tensor([[next_token]], device=device)], dim=1)
        return " ".join([itos.get(i, "<unk>") for i in out_ids])