import torch
from torch import nn, matmul, softmax
from torch.nn import functional as F
import math


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    attn = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k)
    if mask is not None:
        attn = attn.masked_fill(mask == 0, -1e9)
    attn_weights = F.softmax(attn, dim=-1)
    if dropout is not None:
        attn_weights = dropout(attn_weights)
    return torch.matmul(attn_weights, value), attn_weights




class Parametric_attention(nn.Module):
    def __init__(self, input_dim, model_dim, dropout=None):
        super(Parametric_attention, self).__init__()
        self.W_k = nn.Linear(input_dim, model_dim)
        self.W_q = nn.Linear(input_dim, model_dim)
        self.W_v = nn.Linear(input_dim, model_dim)
        self.dropout = dropout


    def forward(self, x):
        key = self.W_k(x)
        value = self.W_v(x)
        query = self.W_q(x)
        d_k = query.size(-1)
        attn = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        attn_weights = F.softmax(attn, dim=-1)
        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)
        return torch.matmul(attn_weights, value), attn_weights




class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = heads
        self.head_dim = int(d_model/heads)
        self.to_query = nn.Linear(d_model, heads * self.head_dim, bias=True)
        self.to_key = nn.Linear(d_model, heads * self.head_dim, bias=True)
        self.to_value = nn.Linear(d_model, heads * self.head_dim, bias=True)
        self.unify_heads = nn.Linear(heads * self.head_dim, d_model, bias=True)

    def forward(self, inputs, mask=None, kv=None):
        # Create Q, K, and V using input vectors
        #inputs = inputs.unsqueeze(0)
        bs, seq, emb_dim = inputs.shape

        if kv is not None:
            kv = kv
        else:
            kv = inputs

        kv_bs, kv_seq_len, _ = kv.size()

        # Transpose: bs x seq-length x num-heads x heads_dim -> bs x num-heads x seq-length x heads_dim
        q = self.to_query(inputs).view(bs, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_key(kv).view(kv_bs, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_value(kv).view(kv_bs, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute Attention scores
        attn_scores = matmul(q, k.transpose(-1, -2))
        # Scale after Dot-product : attn_scores/root_square(head_dim)
        attn_scores = attn_scores/(self.head_dim ** 1/float(2))

        # Apply masking
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 1, value=-1e9)

        # Convert attention scores into probability distributions
        softmax_attn_scores = softmax(attn_scores, dim=-1)

        # Compute Weighted Values
        output = matmul(softmax_attn_scores, v)

        # Reshape the weighted values
        # Transpose: bs x seq-length x num-heads x heads_dim -> bs x seq-length x num-heads x heads_dim)
        output = output.transpose(1, 2).contiguous().view(bs, seq, self.num_heads * self.head_dim)
        output_final = self.unify_heads(output)
        return output_final


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ff_size=2096, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, heads=num_heads)
        self.attn_norm = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, d_model)
        )

        self.final_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings, mask=None):
        # Attn with Pre-Normalization
        embeddings = embeddings + self.dropout(self.self_attn(self.attn_norm(embeddings), mask=mask))
        # FeedForward with Pre-Normalization
        embeddings = embeddings + self.dropout(self.ff(self.final_norm(embeddings)))

        return embeddings


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads=2, num_layers=2):
        super(TransformerEncoder, self).__init__()

        self.enc_layers = nn.ModuleList()
        for i in range(num_layers):
            self.enc_layers.append(TransformerEncoderLayer(d_model, num_heads=num_heads))

        self.norm = nn.LayerNorm(d_model)

    def forward(self, embeddings, mask=None):

        hidden_states = []
        for layer in self.enc_layers:
            embeddings = layer(embeddings, mask)
            hidden_states.append(embeddings)

        return hidden_states, self.norm(hidden_states[-1])


if __name__ == '__main__':
    #d_model = 300
    #input = torch.rand(1, 13, d_model)
    x = torch.tensor([[
        [1, 0, 1, 0],  # input 1
        [0, 2, 2, 2],  # input 2
        [1, 1, 1, 1],  # input 3
    ],
        [
            [1, 0, 1, 0],  # input 1
            [0, 2, 2, 2],  # input 2
            [1, 1, 1, 1],  # input 3
        ]],
        dtype=torch.float32)

    model = Parametric_attention(4,3)
    output, weights = model.forward(x)
    #model_encoder = TransformerEncoder(d_model=d_model, num_heads=8, num_layers=6)
    #hidden, output = model_encoder(input)
    print(output)
