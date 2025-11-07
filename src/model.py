# model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 绝对位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # [d_model/2]频率呈几何级递减

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # 奇数维度: cos 部分少一个维度
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)  # 不参与梯度

    def forward(self, x):
        """
        参数:
            x: [B, T, d_model]
        返回:
            加上位置编码后的张量，形状同 x
        """
        return x + self.pe[:, : x.size(1), :]


# 相对位置偏置：T5-style RelativePositionBias
# 只负责生成一个 [1, heads, Q, K] 的 bias，加到注意力 logit 上
class RelativePositionBiasT5(nn.Module):
    """
    T5-style relative position bias.
    """

    def __init__(
        self,
        num_heads: int,
        num_buckets: int = 32,
        max_distance: int = 128,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional

        # 可学习表：每个 bucket、每个 head 对应一个标量偏置
        # 形状: [num_buckets, num_heads]
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        """
        将相对位置（pos_k - pos_q）映射到 bucket 索引。
        输入:
            relative_position: [Q, K]
        输出:
            bucket 索引: [Q, K]
        """
        num_buckets = self.num_buckets
        max_distance = self.max_distance

        # 约定：n = -relative_position；n<0 表示 key 在 query 右侧
        ret = 0
        n = -relative_position
        if self.bidirectional:
            half = num_buckets // 2
            # 负向相对距离使用前半部分 buckets
            ret = (n < 0).to(torch.long) * half
            n = n.abs()
            num_buckets = half
        else:
            n = torch.clamp(n, min=0) # 单向时，负数（未来位置）裁成 0

        # 小距离用线性划分，大距离用对数划分，能在较少 bucket 下覆盖更大范围
        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact + 1e-6)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        )
        val_if_large = val_if_large.to(torch.long)
        val_if_large = torch.clamp(val_if_large, max=num_buckets - 1)
        buckets = torch.where(is_small, n.to(torch.long), val_if_large)
        return ret + buckets

    def forward(self, q_len: int, k_len: int, device=None) -> torch.Tensor:
        """
        返回偏置张量，形状 [1, heads, Q, K]，可直接加到注意力分数矩阵上。
        """
        context_position = torch.arange(q_len, dtype=torch.long, device=device)[:, None]  # [Q,1]
        memory_position = torch.arange(k_len, dtype=torch.long, device=device)[None, :]  # [1,K]
        relative_position = memory_position - context_position  # [Q, K]

        rp_bucket = self._relative_position_bucket(relative_position)  # [Q,K]
        # 先查表得到 [Q, K, heads]
        values = self.relative_attention_bias(rp_bucket)
        # 维度换位得到 [1, heads, Q, K]
        values = values.permute(2, 0, 1).unsqueeze(0)
        return values


# 缩放点积注意力
def scaled_dot_product_attention(
    q, k, v, mask=None, dropout=None, rel_pos_bias: torch.Tensor = None
):
    """
    参数:
        q, k, v: 形状分别为
            q: [B, heads, T_q, d_k]
            k: [B, heads, T_k, d_k]
            v: [B, heads, T_k, d_k]
        mask: 可广播到 [B, 1, T_q, T_k] 或 [B, heads, T_q, T_k] 的 0/1 掩码
        rel_pos_bias: 相对位置偏置，形状 [1, heads, T_q, T_k] 或 [heads, T_q, T_k]
    返回:
        output: [B, heads, T_q, d_k]
        attn:   [B, heads, T_q, T_k]（注意力权重）
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # [B,H,T_q,T_k]

    if rel_pos_bias is not None:
        # [H,T_q,T_k] -> [1,H,T_q,T_k]
        if rel_pos_bias.dim() == 3:
            rel_pos_bias = rel_pos_bias.unsqueeze(0)
        scores = scores + rel_pos_bias  # broadcast 到 [B,H,T_q,T_k]

    if mask is not None:
        # 支持 [B,1,1,S] 或 [B,1,T,S] 或 [B,T,S]
        if mask.dim() == 3:  # [B,T,S]
            mask = mask.unsqueeze(1)  # -> [B,1,T,S]
        # 与 scores 形状兼容：[B,H,T,S]
        scores = scores.masked_fill(mask == 0, float("-1e9"))

    attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    output = torch.matmul(attn, v)  # [B,H,T_q,d_k]
    return output, attn


# 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        dropout=0.1,
        use_rel_pos: bool = False,
        rel_pos_num_buckets: int = 32,
        rel_pos_max_distance: int = 128,
        rel_pos_bidirectional: bool = True,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.h = num_heads

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.use_rel_pos = use_rel_pos
        self.rel_pos_bias = (
            RelativePositionBiasT5(
                num_heads,
                num_buckets=rel_pos_num_buckets,
                max_distance=rel_pos_max_distance,
                bidirectional=rel_pos_bidirectional,
            )
            if use_rel_pos
            else None
        )

        # 方便训练后可视化
        self.last_attn = None  # [B,H,T,S]

    def forward(self, query, key, value, mask=None):
        """
        参数:
            query, key, value: [B, T, d_model]
            mask: 同 scaled_dot_product_attention
        返回:
            线性映射后的输出，形状 [B, T, d_model]
        """
        B = query.size(0)

        def shape(x):
            # [B, T, d_model] -> [B, H, T, d_k]
            return x.view(B, -1, self.h, self.d_k).transpose(1, 2)

        q = shape(self.linear_q(query))
        k = shape(self.linear_k(key))
        v = shape(self.linear_v(value))

        rel_bias = None
        # 这里只在自注意力（Q,K 长度相等）时加相对位置编码
        if self.use_rel_pos and query.size(1) == key.size(1):
            rel_bias = self.rel_pos_bias(
                q_len=query.size(1), k_len=key.size(1), device=query.device
            )

        x, attn = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.dropout, rel_pos_bias=rel_bias
        )
        self.last_attn = attn.detach()  # [B,H,T,S]

        # [B,H,T,d_k] -> [B,T,H*d_k]
        x = x.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.linear_out(x)


# 前馈网络
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


# 编码器 / 解码器层
class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        heads,
        d_ff,
        dropout=0.1,
        use_rel_pos: bool = False,
        rel_pos_num_buckets: int = 32,
        rel_pos_max_distance: int = 128,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model,
            heads,
            dropout=dropout,
            use_rel_pos=use_rel_pos,
            rel_pos_num_buckets=rel_pos_num_buckets,
            rel_pos_max_distance=rel_pos_max_distance,
        )
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # 自注意力
        attn_out = self.self_attn(x, x, x, mask=src_mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        # 前馈网络
        ffn_out = self.ff(x)
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=layer.self_attn.linear_q.in_features,
                heads=layer.self_attn.h,
                d_ff=layer.ff.w1.out_features,
                dropout=layer.dropout.p,
                use_rel_pos=(layer.self_attn.rel_pos_bias is not None),
                rel_pos_num_buckets=layer.self_attn.rel_pos_bias.num_buckets if layer.self_attn.rel_pos_bias else 32,
                rel_pos_max_distance=layer.self_attn.rel_pos_bias.max_distance if layer.self_attn.rel_pos_bias else 128,
            ) for _ in range(N)
        ])

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask=src_mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        heads,
        d_ff,
        dropout=0.1,
        use_rel_pos: bool = False,
        rel_pos_num_buckets: int = 32,
        rel_pos_max_distance: int = 128,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model,
            heads,
            dropout=dropout,
            use_rel_pos=use_rel_pos,
            rel_pos_num_buckets=rel_pos_num_buckets,
            rel_pos_max_distance=rel_pos_max_distance,
        )
        self.src_attn = MultiHeadAttention(d_model, heads, dropout=dropout, use_rel_pos=False)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        # 解码器自注意力
        x2 = self.self_attn(x, x, x, mask=tgt_mask)
        x = x + self.dropout(x2)
        x = self.norm1(x)
        # 编码器-解码器交叉注意力
        x2 = self.src_attn(x, memory, memory, mask=src_mask)
        x = x + self.dropout(x2)
        x = self.norm2(x)
        # 前馈网络
        x2 = self.ff(x)
        x = x + self.dropout(x2)
        x = self.norm3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N: int):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=layer.self_attn.linear_q.in_features,
                heads=layer.self_attn.h,
                d_ff=layer.ff.w1.out_features,
                dropout=layer.dropout.p,
                use_rel_pos=(layer.self_attn.rel_pos_bias is not None),
                rel_pos_num_buckets=layer.self_attn.rel_pos_bias.num_buckets if layer.self_attn.rel_pos_bias else 32,
                rel_pos_max_distance=layer.self_attn.rel_pos_bias.max_distance if layer.self_attn.rel_pos_bias else 128,
            ) for _ in range(N)
        ])

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, memory, src_mask=src_mask, tgt_mask=tgt_mask)
        return x



class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        d_ff=1024,
        max_len=512,
        pad_idx=0,
        use_relative_pos: bool = False,
        rel_pos_num_buckets: int = 32,
        rel_pos_max_distance: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        # 词嵌入 + 绝对位置编码
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

        # 构建单层模板，再复制 N 次
        enc_layer = EncoderLayer(
            d_model,
            nhead,
            d_ff,
            use_rel_pos=use_relative_pos,
            rel_pos_num_buckets=rel_pos_num_buckets,
            rel_pos_max_distance=rel_pos_max_distance,
        )
        dec_layer = DecoderLayer(
            d_model,
            nhead,
            d_ff,
            use_rel_pos=use_relative_pos,
            rel_pos_num_buckets=rel_pos_num_buckets,
            rel_pos_max_distance=rel_pos_max_distance,
        )

        self.encoder = Encoder(enc_layer, num_encoder_layers)
        self.decoder = Decoder(dec_layer, num_decoder_layers)
        self.out = nn.Linear(d_model, vocab_size)

        self.use_relative_pos = use_relative_pos

    def encode(self, src, src_mask=None):
        # src: [B,S]
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        return self.encoder(x, src_mask=src_mask)

    def decode(self, tgt, memory, src_mask=None, tgt_mask=None):
        # tgt: [B,T]
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        return self.decoder(x, memory, src_mask=src_mask, tgt_mask=tgt_mask)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
               编码-解码整体前向：
                   1) 编码 src 得到 memory
                   2) 解码 tgt（带因果 mask）并与 memory 做 cross-attn
                   3) 映射到词表维度得到 logits
               """
        memory = self.encode(src, src_mask=src_mask)
        out = self.decode(tgt, memory, src_mask=src_mask, tgt_mask=tgt_mask)
        logits = self.out(out)  # [B,T,V]
        return logits


def make_std_masks(src, tgt, pad_idx):
    """
    兼容原来从 model 导入的接口：
    返回:
        src_mask: [B,1,1,S]  —— padding 位置为 0，其余为 1
        tgt_mask: [B,1,T,T]  —— 同时包含 padding mask 和 下三角 no-peak 因果 mask
    说明：
        - 因果 mask 确保解码器只看见当前及之前的 token
        - 该函数不做 dtype/设备强约束，保持与输入一致
    """
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)  # [B,1,1,S]

    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # [B,1,1,T]
    T = tgt.size(1)
    nopeak = torch.tril(
        torch.ones((T, T), dtype=torch.uint8, device=tgt.device)
    ).unsqueeze(0).unsqueeze(1)  # [1,1,T,T]
    tgt_mask = tgt_pad_mask & nopeak  # [B,1,T,T]

    return src_mask, tgt_mask
