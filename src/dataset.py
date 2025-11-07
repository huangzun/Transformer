# dataset.py
import os
import zipfile
import random
from typing import Tuple, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader

# <pad>: 填充符
# <unk>: 未登录词
# <bos>: 句子开始（decoder 输入用）
# <eos>: 句子结束
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

def read_archive_text(archive_path: str, member_ext=".tokens"):
    """
    从 zip 压缩包中读取文本（适用于 wikitext-2.zip 格式）
    如果不是 zip，则尝试按普通文本文件读取。

    参数:
        archive_path: 数据文件路径
        member_ext: 需要读取的文件后缀名

    返回:
        拼接后的完整文本字符串
    """
    if not os.path.exists(archive_path):
        raise FileNotFoundError(f"{archive_path} not found")
    text_parts = []
    # 若是 zip 文件
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as z:
            for name in z.namelist():
                if name.endswith(member_ext):# 只读取 .tokens 文件
                    with z.open(name) as f:
                        try:
                            s = f.read().decode("utf-8")
                        except:
                            s = f.read().decode("latin1")
                        text_parts.append(s)
    else:
        # 按普通文件读取
        with open(archive_path, "r", encoding="utf-8") as f:
            text_parts.append(f.read())
    return "\n".join(text_parts)

def build_tokenizer_and_vocab(text: str, min_freq: int = 1):
    """
    简单基于空格的分词，并构建词表（按词频过滤）。

    参数:
        text: 原始文本
        min_freq: 最小词频（低于该频率的词会被过滤）

    返回:
        tokens: 分词结果列表
        stoi: 词 → ID 的字典
        itos: ID → 词 的字典
    """
    tokens = text.split()
    freq = {}
    for t in tokens:# 统计词频
        freq[t] = freq.get(t, 0) + 1

    # 过滤低频词
    vocab_tokens = [t for t, c in freq.items() if c >= min_freq]
    # 按「词频降序 → 字典序」排序，保证确定性
    vocab_tokens.sort(key=lambda x: (-freq[x], x))

    # 构建 stoi，特殊符号优先
    stoi = {}
    idx = 0
    for s in SPECIAL_TOKENS:
        stoi[s] = idx
        idx += 1
    for t in vocab_tokens:
        if t in stoi:
            continue
        stoi[t] = idx
        idx += 1
    itos = {i: s for s, i in stoi.items()}
    return tokens, stoi, itos

def tokens_to_ids(tokens: List[str], stoi: Dict[str,int]):
    """
        将 tokens 转换为 ID 序列，未登录词映射到 <unk>。
        """
    unk = stoi["<unk>"]
    return [stoi.get(t, unk) for t in tokens]

class LMSequenceDataset(Dataset):
    """
       将连续 token 序列切分成固定长度的多个样本。
       这里使用「不重叠切分」，一步步按 seq_len 切。
       """
    def __init__(self, ids: List[int], seq_len: int = 128):

        self.seq_len = seq_len

        self.sequences = []
        n = len(ids)
        i = 0
        while i + seq_len <= n:# 逐段切片
            self.sequences.append(torch.tensor(ids[i:i+seq_len], dtype=torch.long))
            i += seq_len# 不重叠前移

    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return self.sequences[idx]

def collate_lm(batch):
    """
    DataLoader 的 collate 函数。
    输入 batch: List[Tensor(seq_len)]
    输出:
        src: 原序列
        tgt: 目标序列（语言模型中即为预测下一个 token，shift 在训练时完成）
    """
    data = torch.stack(batch, dim=0)  # [B, seq_len]
    return data, data

def get_dataloaders(archive_path: str, seq_len: int = 128, batch_size: int = 32, seed: int = 42, min_freq: int = 1, split_ratio=0.9):
    """
       构建训练/验证数据加载器。

       - 从压缩文件中读取文本
       - 建立词表
       - 将文本转换为 token ID
       - 按比例切分数据
       - 构建 Dataset & DataLoader

       返回:
           train_loader, val_loader, stoi, itos, vocab_size
       """
    random.seed(seed)

    text = read_archive_text(archive_path)
    tokens, stoi, itos = build_tokenizer_and_vocab(text, min_freq=min_freq)
    token_ids = tokens_to_ids(tokens, stoi)


    n = len(token_ids)
    split_idx = int(n * split_ratio)
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]

    train_ds = LMSequenceDataset(train_ids, seq_len=seq_len)
    val_ds = LMSequenceDataset(val_ids, seq_len=seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_lm, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_lm, drop_last=True)

    vocab_size = len(stoi)


    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("Empty dataset: adjust seq_len or archive content")

    # 基本安全检查：确保不会出现超出词表范围的 token id
    for i, batch in enumerate(train_loader):
        if batch[0].max().item() >= vocab_size:
            raise AssertionError(f"Token id overflow in training data: found {int(batch[0].max().item())} >= vocab_size {vocab_size}")
        if i >= 10:# 检查前 10 个即可，避免额外开销
            break

    return train_loader, val_loader, stoi, itos, vocab_size