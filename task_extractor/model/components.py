import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    def _init_(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super()._init_()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TokenEmbedding(nn.Module):
    def _init_(self, vocab_size: int, d_model: int):
        super()._init_()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * math.sqrt(self.d_model)

class MultiHeadAttention(nn.Module):
    def _init_(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super()._init_()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.scale = math.sqrt(self.d_k)
    
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        Q, K, V = self.W_q(query), self.W_k(key), self.W_v(value)
        Q, K, V = self.split_heads(Q), self.split_heads(K), self.split_heads(V)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None: scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = self.dropout(F.softmax(scores, dim=-1))
        output = self.W_o(self.combine_heads(torch.matmul(attn_weights, V)))
        return output, attn_weights

class PositionWiseFeedForward(nn.Module):
    def _init_(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super()._init_()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.GELU()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class TransformerEncoderLayer(nn.Module):
    def _init_(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, pre_norm: bool = True):
        super()._init_()
        self.pre_norm = pre_norm
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pre_norm:
            normed = self.norm1(x)
            attn_output, attn_weights = self.self_attention(normed, normed, normed, mask)
            x = x + self.dropout1(attn_output)
            normed = self.norm2(x)
            x = x + self.dropout2(self.feed_forward(normed))
        else:
            attn_output, attn_weights = self.self_attention(x, x, x, mask)
            x = self.norm1(x + self.dropout1(attn_output))
            x = self.norm2(x + self.dropout2(self.feed_forward(x)))
        return x, attn_weights

class TransformerEncoder(nn.Module):
    def _init_(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, pre_norm: bool = True):
        super()._init_()
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, pre_norm) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model) if pre_norm else nn.Identity()
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[list]]:
        attention_weights = []
        for layer in self.layers:
            x, attn = layer(x, mask)
            if return_attention: attention_weights.append(attn)
        return self.final_norm(x), attention_weights if return_attention else None

class ConditionalRandomField(nn.Module):
    def _init_(self, num_tags: int, batch_first: bool = True):
        super()._init_()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
    
    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, mask: Optional[torch.Tensor] = None, reduction: str = 'mean') -> torch.Tensor:
        if not self.batch_first: emissions, tags, mask = emissions.transpose(0, 1), tags.transpose(0, 1), mask.transpose(0, 1) if mask is not None else None
        if mask is None: mask = torch.ones_like(tags, dtype=torch.bool)
        nll = self._compute_normalizer(emissions, mask) - self._compute_score(emissions, tags, mask)
        return nll.mean() if reduction == 'mean' else nll.sum() if reduction == 'sum' else nll

    def _compute_score(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = tags.shape
        score = self.start_transitions[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)
        for i in range(1, seq_len):
            score += (self.transitions[tags[:, i], tags[:, i-1]] + emissions[:, i].gather(1, tags[:, i].unsqueeze(1)).squeeze(1)) * mask[:, i].float()
        seq_ends = mask.long().sum(dim=1) - 1
        score += self.end_transitions[tags.gather(1, seq_ends.unsqueeze(1)).squeeze(1)]
        return score

    def _compute_normalizer(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        score = self.start_transitions + emissions[:, 0]
        for i in range(1, emissions.shape[1]):
            next_score = score.unsqueeze(2) + self.transitions + emissions[:, i].unsqueeze(1)
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[:, i].unsqueeze(1), next_score, score)
        return torch.logsumexp(score + self.end_transitions, dim=1)

    def decode(self, emissions: torch.Tensor, mask: Optional[torch.Tensor] = None) -> list:
        if not self.batch_first: emissions = emissions.transpose(0, 1)
        if mask is None: mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)
        elif not self.batch_first: mask = mask.transpose(0, 1)
        
        batch_size, seq_len, num_tags = emissions.shape
        score = self.start_transitions + emissions[:, 0]
        history = []
        for i in range(1, seq_len):
            next_score = score.unsqueeze(2) + self.transitions + emissions[:, i].unsqueeze(1)
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[:, i].unsqueeze(1), next_score, score)
            history.append(indices)
        score += self.end_transitions
        
        seq_ends = mask.long().sum(dim=1) - 1
        best_tags_list = []
        for idx in range(batch_size):
            best_last_tag = score[idx].argmax().item()
            best_tags = [best_last_tag]
            for hist in reversed(history[:seq_ends[idx].item()]):
                best_last_tag = hist[idx][best_last_tag].item()
                best_tags.append(best_last_tag)
            best_tags_list.append(best_tags[::-1])
        return best_tags_list