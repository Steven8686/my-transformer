import numpy as np
import torch.nn as nn
import torch


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, num_heads, p=0.):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p)

        # linear projections
        self.W_Q = nn.Linear(d_model, d_k * num_heads)
        self.W_K = nn.Linear(d_model, d_k * num_heads)
        self.W_V = nn.Linear(d_model, d_v * num_heads)
        self.W_out = nn.Linear(d_v * num_heads, d_model)

        # Normalization
        # References: <<Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification>>
        nn.init.normal_(self.W_Q.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.W_K.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.W_V.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        nn.init.normal_(self.W_out.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

    def forward(self, Q, K, V, attn_mask, **kwargs):
        N = Q.size(0)
        q_len, k_len = Q.size(1), K.size(1)
        d_k, d_v = self.d_k, self.d_v
        num_heads = self.num_heads

        # multi_head split
        Q = self.W_Q(Q).view(N, -1, num_heads, d_k).transpose(1, 2)
        K = self.W_K(K).view(N, -1, num_heads, d_k).transpose(1, 2)
        V = self.W_V(V).view(N, -1, num_heads, d_v).transpose(1, 2)

        # pre-process mask
        if attn_mask is not None:
            assert attn_mask.size() == (N, q_len, k_len)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)  # broadcast
            attn_mask = attn_mask.bool()

        # calculate attention weight
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e4)
        attns = torch.softmax(scores, dim=-1)  # attention weights
        attns = self.dropout(attns)

        # calculate output
        output = torch.matmul(attns, V)

        # multi_head merge
        output = output.transpose(1, 2).contiguous().reshape(N, -1, d_v * num_heads)
        output = self.W_out(output)

        return output


def get_len_mask(b: int, max_len: int, feat_lens: torch.Tensor, device: torch.device) -> torch.Tensor:
    attn_mask = torch.ones((b, max_len, max_len), device=device)
    for i in range(b):
        attn_mask[i, :, :feat_lens[i]] = 0
    return attn_mask.to(torch.bool)


def pos_sinusoid_embedding(seq_len, d_model):
    embeddings = torch.zeros((seq_len, d_model))
    for i in range(d_model):
        f = torch.sin if i % 2 == 0 else torch.cos
        embeddings[:, i] = f(torch.arange(0, seq_len) / np.power(1e4, 2 * (i // 2) / d_model))
    return embeddings.float()


def get_subsequent_mask(b: int, max_len: int, device: torch.device) -> torch.Tensor:
    """
    Args:
        b: batch-size.
        max_len: the length of the whole seqeunce.
        device: cuda or cpu.
    """
    return torch.triu(torch.ones((b, max_len, max_len), device=device), diagonal=1).to(torch.bool)     # or .to(torch.uint8)


def get_enc_dec_mask(
    b: int, max_feat_len: int, feat_lens: torch.Tensor, max_label_len: int, device: torch.device) -> torch.Tensor:
    attn_mask = torch.zeros((b, max_label_len, max_feat_len), device=device)       # (b, seq_q, seq_k)
    for i in range(b):
        attn_mask[i, :, feat_lens[i]:] = 1
    return attn_mask.to(torch.bool)


class PoswiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, p=0.):
        super(PoswiseFFN, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.conv1 = nn.Conv1d(d_model, d_ff, 1, 1, 0)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=p)

    def forward(self, X):
        out = self.conv1(X.transpose(1, 2))     # (N, d_model, seq_len) -> (N, d_ff, seq_len)
        out = self.relu(out)
        out = self.conv2(out).transpose(1, 2)   # (N, d_ff, seq_len) -> (N, d_model, seq_len)
        out = self.dropout(out)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, dim, n, dff, dropout_posffn, dropout_attn):
        """
        Args:
            dim: input dimension
            n: number of attention heads
            dff: dimention of PosFFN (Positional FeedForward)
            dropout_posffn: dropout ratio of PosFFN
            dropout_attn: dropout ratio of attention module
        """
        assert dim % n == 0
        hdim = dim // n     # dimension of each attention head
        super(EncoderLayer, self).__init__()
        # LayerNorm
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        # MultiHeadAttention
        self.multi_head_attn = MultiHeadAttention(hdim, hdim, dim, n, dropout_attn)
        # Position-wise Feedforward Neural Network
        self.poswise_ffn = PoswiseFFN(dim, dff, p=dropout_posffn)

    def forward(self, enc_in, attn_mask):
        # reserve original input for later residual connections
        residual = enc_in
        # MultiHeadAttention forward
        context = self.multi_head_attn(enc_in, enc_in, enc_in, attn_mask)
        # residual connection and norm
        out = self.norm1(residual + context)
        residual = out
        # position-wise feedforward
        out = self.poswise_ffn(out)
        # residual connection and norm
        out = self.norm2(residual + out)

        return out


class Encoder(nn.Module):
    def __init__(
            self, dropout_emb, dropout_posffn, dropout_attn,
            num_layers, enc_dim, num_heads, dff, tgt_len,
    ):
        """
        Args:
            dropout_emb: dropout ratio of Position Embeddings.
            dropout_posffn: dropout ratio of PosFFN.
            dropout_attn: dropout ratio of attention module.
            num_layers: number of encoder layers
            enc_dim: input dimension of encoder
            num_heads: number of attention heads
            dff: dimensionf of PosFFN
            tgt_len: the maximum length of sequences
        """
        super(Encoder, self).__init__()
        # The maximum length of input sequence
        self.tgt_len = tgt_len
        self.pos_emb = nn.Embedding.from_pretrained(pos_sinusoid_embedding(tgt_len, enc_dim), freeze=True)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.layers = nn.ModuleList(
            [EncoderLayer(enc_dim, num_heads, dff, dropout_posffn, dropout_attn) for _ in range(num_layers)]
        )

    def forward(self, X, X_lens, mask=None):
        # add position embedding
        batch_size, seq_len, d_model = X.shape
        out = X + self.pos_emb(torch.arange(seq_len, device=X.device))  # (batch_size, seq_len, d_model)
        out = self.emb_dropout(out)
        # encoder layers
        for layer in self.layers:
            out = layer(out, mask)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, dim, n, dff, dropout_posffn, dropout_attn):
        """
        Args:
            dim: input dimension
            n: number of attention heads
            dff: dimention of PosFFN (Positional FeedForward)
            dropout_posffn: dropout ratio of PosFFN
            dropout_attn: dropout ratio of attention module
        """
        super(DecoderLayer, self).__init__()
        assert dim % n == 0
        hdim = dim // n
        # LayerNorms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        # Position-wise Feed-Forward Networks
        self.poswise_ffn = PoswiseFFN(dim, dff, p=dropout_posffn)
        # MultiHeadAttention, both self-attention and encoder-decoder cross attention)
        self.dec_attn = MultiHeadAttention(hdim, hdim, dim, n, dropout_attn)
        self.enc_dec_attn = MultiHeadAttention(hdim, hdim, dim, n, dropout_attn)

    def forward(self, dec_in, enc_out, dec_mask, dec_enc_mask, cache=None, freqs_cis=None):
        # decoder's self-attention
        residual = dec_in
        context = self.dec_attn(dec_in, dec_in, dec_in, dec_mask)
        dec_out = self.norm1(residual + context)
        # encoder-decoder cross attention
        residual = dec_out
        context = self.enc_dec_attn(dec_out, enc_out, enc_out, dec_enc_mask)
        dec_out = self.norm2(residual + context)
        # position-wise feed-forward networks
        residual = dec_out
        out = self.poswise_ffn(dec_out)
        dec_out = self.norm3(residual + out)
        return dec_out


class Decoder(nn.Module):
    def __init__(
            self, dropout_emb, dropout_posffn, dropout_attn,
            num_layers, dec_dim, num_heads, dff, tgt_len, tgt_vocab_size,
    ):
        """
        Args:
            dropout_emb: dropout ratio of Position Embeddings.
            dropout_posffn: dropout ratio of PosFFN.
            dropout_attn: dropout ratio of attention module.
            num_layers: number of encoder layers
            dec_dim: input dimension of decoder
            num_heads: number of attention heads
            dff: dimensionf of PosFFN
            tgt_len: the target length to be embedded.
            tgt_vocab_size: the target vocabulary size.
        """
        super(Decoder, self).__init__()

        # output embedding
        self.tgt_emb = nn.Embedding(tgt_vocab_size, dec_dim)
        self.dropout_emb = nn.Dropout(p=dropout_emb)                            # embedding dropout
        # position embedding
        self.pos_emb = nn.Embedding.from_pretrained(pos_sinusoid_embedding(tgt_len, dec_dim), freeze=True)
        # decoder layers
        self.layers = nn.ModuleList(
            [
                DecoderLayer(dec_dim, num_heads, dff, dropout_posffn, dropout_attn) for _ in
                range(num_layers)
            ]
        )

    def forward(self, labels, enc_out, dec_mask, dec_enc_mask, cache=None):
        # output embedding and position embedding
        tgt_emb = self.tgt_emb(labels)
        pos_emb = self.pos_emb(torch.arange(labels.size(1), device=labels.device))
        dec_out = self.dropout_emb(tgt_emb + pos_emb)
        # decoder layers
        for layer in self.layers:
                dec_out = layer(dec_out, enc_out, dec_mask, dec_enc_mask)
        return dec_out


class Transformer(nn.Module):
    def __init__(
            self, frontend: nn.Module, encoder: nn.Module, decoder: nn.Module,
            dec_out_dim: int, vocab: int,
    ) -> None:
        super().__init__()
        self.frontend = frontend     # feature extractor
        self.encoder = encoder
        self.decoder = decoder
        self.linear = nn.Linear(dec_out_dim, vocab)

    def forward(self, X: torch.Tensor, X_lens: torch.Tensor, labels: torch.Tensor):
        X_lens, labels = X_lens.long(), labels.long()
        b = X.size(0)
        device = X.device
        # frontend
        out = self.frontend(X)
        max_feat_len = out.size(1)                            # compute after frontend because of optional subsampling
        max_label_len = labels.size(1)
        # encoder
        enc_mask = get_len_mask(b, max_feat_len, X_lens, device)
        enc_out = self.encoder(out, X_lens, enc_mask)
        # decoder
        dec_mask = get_subsequent_mask(b, max_label_len, device)
        dec_enc_mask = get_enc_dec_mask(b, max_feat_len, X_lens, max_label_len, device)
        dec_out = self.decoder(labels, enc_out, dec_mask, dec_enc_mask)
        logits = self.linear(dec_out)

        return logits


class TransformerDecoderModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers):
        super(TransformerDecoderModel, self).__init__()  # 调用基类的初始化函数
        # 创建嵌入层，将词索引转换为嵌入向量
        self.embed = nn.Embedding(vocab_size, embed_size)
        # 初始化位置编码，是一个可学习的参数
        self.positional_encoding = nn.Parameter(torch.randn(embed_size).unsqueeze(0))
        # 定义一个Transformer解码器层
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_dim)
        # 堆叠多个解码器层构成完整的解码器
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # 定义输出层，将解码器输出转换回词汇空间
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, src):
        # 嵌入输入并添加位置编码
        src = self.embed(src) + self.positional_encoding
        # 生成源序列的掩码，用于屏蔽未来的信息
        src_mask = self.generate_square_subsequent_mask(src.size(0))
        # 通过解码器传递源数据和掩码
        output = self.transformer_decoder(src, src, src_mask)
        # 应用线性层输出最终的预测结果
        output = self.fc(output)
        return output

    def generate_square_subsequent_mask(self, sz):
        # 生成一个上三角矩阵，用于序列生成中遮蔽未来位置的信息
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        # 将掩码的非零位置设为无穷大，零位置设为0
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
