import torch
import torch.nn as nn
import math
import torch.nn.functional as F

#TODO Create Attention embedding model to create context embeddings from downsampled embeddings


def calculate_attention(query: torch.tensor, keys: torch.tensor, values: torch.tensor):
    attention_scores = torch.matmul(query, keys.transpose(-2, -1))
    attention_scores = attention_scores/math.sqrt(keys.shape[-1])
    attention_scores = F.softmax(attention_scores, dim=1)
    attention = torch.matmul(attention_scores, values)
    return attention, attention_scores


class FeedForward(nn.Module):
    def __init__(self, embed_size: int):
        super().__init__()
        self.layer1 = nn.Linear(embed_size, embed_size)
        self.layer2 = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.layer2(x)
        return x
    

class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_size: int):
        super().__init__()
        self.embed_size = embed_size
        self.query_dense = nn.Linear(embed_size, embed_size)
        self.key_dense = nn.Linear(embed_size, embed_size)
        self.value_dense = nn.Linear(embed_size, embed_size)

    def forward(self, embeddings: torch.Tensor):
        query = self.query_dense(embeddings)
        key = self.key_dense(embeddings)
        value = self.value_dense(embeddings)
        attention, _ = calculate_attention(value, key, query)
        return attention
    

class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, embed_size: int, max_seq_length: int):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size)
        )
        pe = torch.zeros(max_seq_length, embed_size)
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.sin(position*div_term)
        self.register_buffer("positional_embedding", pe)
    def forward(self, x:torch.tensor):
        return x + self.positional_embedding[: x.size(1), :]
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_size: int):
        super().__init__()
        self.attention_layer = SelfAttentionLayer(embed_size)
        self.feed_forward = FeedForward(embed_size)
        self.layer_norm1 = nn.LayerNorm(embed_size)

    def forward(self, x: torch.tensor):
        context = self.attention_layer(x)
        context = self.layer_norm1(x)
        context = self.feed_forward(x)
        context = F.gelu(context)
        output = context + x
        return output



class Transformer(nn.Module):
    def __init__(self, embed_size: int, num_layers: int, max_sequence_len: int):
        super().__init__()
        self.positional_encoding = SinusoidalPositionEncoding(embed_size, max_sequence_len)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_size) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor):
        x = self.positional_encoding(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        return x
        

if __name__ == "__main__":
    embed_size = 16
    num_layers=3
    max_seq_len = 15

    model = Transformer(embed_size, num_layers, max_seq_len)
    x = torch.randn(64, 2042, embed_size)
    print(x.shape)
    print(model(x).shape)


