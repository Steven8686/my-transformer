import copy

import torch.nn as nn
import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = np.float16


class MultiHeadAttention(nn.Module):
    """
        Implements multi-head attention mechanism from "Attention Is All You Need".
        Handles input projections, splits into multiple heads, computes attention scores,
        applies masks, and produces final output through linear projection.

        Parameters:
            embed_size (int): Embedding dimension of model (d_model)
            num_head (int): Number of parallel attention heads
            dim_k (int): Dimension of key vectors (default: embed_size)
            dim_v (int): Dimension of value vectors (default: embed_size)
            drop_rate (float): Dropout probability for attention weights

        Forward Args:
            q (Tensor): [N, len_q, dim_k] query vectors
            k (Tensor): [N, len_k, dim_k] key vectors
            v (Tensor): [N, len_v, dim_v] value vectors
            padding_mask (Tensor): [N, len_q] boolean mask for variable-length sequences
            sequence_mask (Tensor): [len_q, len_k] causal mask for autoregressive tasks

        Returns:
            out (Tensor): [N, len_q, embed_size] context-aware representations
    """
    def __init__(self, embed_size, num_head, dim_k, dim_v, drop_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        # basic parameters
        self.embed_size = embed_size
        self.dim_k = dim_k if dim_k is not None else embed_size
        self.dim_v = dim_v if dim_v is not None else embed_size
        self.num_head = num_head
        self.dim_head = embed_size // num_head
        self.dropout = nn.Dropout(drop_rate)
        # The block below illustrates the situation when QKV input dimension doesn't equal to embed_size. At this time we
        # need projections.
        # Be aware that dim_k is not len_k
        self.is_QKV_same_dim = True
        if self.dim_k != embed_size and self.dim_v != embed_size:
            self.is_QKV_same_dim = False
        self.Q = nn.Linear(dim_k, embed_size)
        self.K = nn.Linear(dim_k, embed_size)
        self.V = nn.Linear(dim_v, embed_size)
        self.Out = nn.Linear(embed_size, embed_size)

    def forward(self, q, k, v, padding_mask, sequence_mask):
        # batch size N
        N = q.shape[0]
        # QKV input max length
        len_k, len_v, len_q = k.shape[1], v.shape[1], q.shape[1]

        # projection [N, len_qkv, dim_qkv] -> [N, len_qkv, embed_size]
        q = self.Q(q)
        v = self.V(v)
        k = self.K(k)
        # split 3rd dimension for multi-head attention
        q = q.reshape(N, len_q, self.num_head, self.dim_head)
        v = v.reshape(N, len_v, self.num_head, self.dim_head)
        k = k.reshape(N, len_k, self.num_head, self.dim_head)

        # attention score
        energy = torch.einsum("nqhd,nkhd->nhqk", [q, k])
        attention = torch.softmax(energy/(self.embed_size**0.5), dim=3)
        # applying masks
        # padding_mask [N, len_q] sequence_mask [len_q, len_k]
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
        attention.masked_fill(~padding_mask, float('-inf'))
        sequence_mask = sequence_mask.unsqueeze(0).unsqueeze(0)
        sequence_mask = sequence_mask.expand(N, self.num_head, -1, -1)
        attention.masked_fill(sequence_mask, float('-inf'))
        # output projection
        out = (torch.einsum("nhqk,nkhd->nqhd", [attention, v]))
        out = out.reshape(out.shape[0], out.shape[1], -1)
        out = self.Out(out)
        return out


class DecoderLayer(nn.Module):
    """
        Implements a Transformer decoder layer with two multi-head attention mechanisms and position-wise feed-forward network.
        Follows the standard architecture with residual connections and layer normalization. Processes inputs through:
        1. Masked self-attention (preventing positions from attending to subsequent positions)
        2. Encoder-decoder attention (attending to encoder outputs)
        3. Feed-forward network
    """
    def __init__(self, embed_size, num_heads, dim_k, dim_v, drop_rate=0.1):
        """
            Parameters:
                embed_size (int): Dimensionality of input and output representations
                num_heads (int): Number of parallel attention heads
                dim_k (int): Dimension of key vectors (optional, defaults to embed_size)
                dim_v (int): Dimension of value vectors (optional, defaults to embed_size)
                drop_rate (float): Dropout probability (default: 0.1)
        """
        super(DecoderLayer, self).__init__()
        # Two layers of multi-head attention for a Decoder Layer
        self.multi_head_self_attention = MultiHeadAttention(embed_size, num_heads, dim_k, dim_v, drop_rate)
        self.multi_head_encoder_decoder_attention = MultiHeadAttention(embed_size, num_heads, dim_k, dim_v, drop_rate)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size),
            nn.Dropout(drop_rate)
        )

        self.norm1 = nn.LayerNorm(embed_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_size, eps=1e-6)
        self.norm3 = nn.LayerNorm(embed_size, eps=1e-6)

        self.dropout1 = nn.Dropout(drop_rate)
        self.dropout2 = nn.Dropout(drop_rate)
        self.dropout3 = nn.Dropout(drop_rate)

    def forward(self, x, padding_mask, look_ahead_mask):
        """
            Forward Parameters:
                x (Tensor): Input tensor of shape (batch_size, seq_len, embed_size)
                padding_mask (Tensor): Mask for padding tokens [batch_size, seq_len]
                look_ahead_mask (Tensor): Causal mask for decoder self-attention [seq_len, seq_len]
            Returns:
                Tensor: Output of same shape as input (batch_size, seq_len, embed_size)
        """
        attn1 = self.multi_head_self_attention(x, x, x, padding_mask, look_ahead_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.norm1(attn1 + x)

        attn2 = self.multi_head_encoder_decoder_attention(out1, out1, out1, padding_mask, look_ahead_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.norm2(attn2 + out1)

        ffn_output = self.feed_forward(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.norm3(ffn_output + out2)

        return out3


class Decoder(nn.Module):
    """Implements Transformer decoder with learnable positional encoding """
    def __init__(self, num_layers, vocab_size, embed_size, num_heads, seq_length):
        """
            Parameters:
                num_layers: Number of decoder layers
                vocab_size: Size of vocabulary
                embed_size: Dimension of embeddings
                num_heads: Number of attention heads
        """
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.layers = nn.ModuleList([copy.deepcopy(DecoderLayer(embed_size, num_heads, embed_size, embed_size)) for i in range(num_layers)])
        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(seq_length, embed_size).unsqueeze(0))
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x, padding_mask):
        """Transformer decoder forward pass.
            Args:
                x: (Tensor) Input token indices of target sequence, shape [batch_size, tgt_seq_len]
                padding_mask: (Tensor) Boolean mask for padding tokens in target sequence, shape [batch_size, tgt_seq_len]
            Returns:
                (Tensor) Output logits for vocabulary prediction, shape [batch_size, tgt_seq_len, vocab_size]
        """
        x = self.embed(x)
        x = x + self.positional_encoding
        look_ahead_mask = self.create_sequence_mask(x.size(1))
        for layer in self.layers:
            x = layer(x, padding_mask, look_ahead_mask)
        x = self.fc(x)
        return x

    def create_sequence_mask(self, sz):
        # Generate an upper triangular matrix to represent sequence mask
        mask = (torch.triu(torch.ones(sz, sz)) == 1)
        return mask.to(device)

