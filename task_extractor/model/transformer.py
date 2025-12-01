import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from .components import TokenEmbedding, PositionalEncoding, TransformerEncoder, ConditionalRandomField

class TaskExtractionTransformer(nn.Module):
    def init(self, vocab_size: int, d_model: int = 256, num_heads: int = 8, num_layers: int = 6, d_ff: int = 1024, max_seq_len: int = 512, num_assignees: int = 5, num_priorities: int = 4, num_bio_tags: int = 9, dropout: float = 0.1, use_crf: bool = True, pre_norm: bool = True):
        super().init()
        self.d_model = d_model
        self.use_crf = use_crf
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, dropout, pre_norm)
        
        self.bio_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, num_bio_tags))
        self.task_detection_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 2))
        self.assignee_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, num_assignees + 1))
        self.priority_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, num_priorities + 1))
        self.deadline_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 3))
        
        if use_crf: self.crf = ConditionalRandomField(num_bio_tags, batch_first=True)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None: nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def create_padding_mask(self, input_ids: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
        return (input_ids != pad_token_id).unsqueeze(1).unsqueeze(2)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, bio_labels: Optional[torch.Tensor] = None, assignee_labels: Optional[torch.Tensor] = None, priority_labels: Optional[torch.Tensor] = None, deadline_labels: Optional[torch.Tensor] = None, return_attention: bool = False) -> Dict[str, torch.Tensor]:
        mask = self.create_padding_mask(input_ids) if attention_mask is None else attention_mask.unsqueeze(1).unsqueeze(2)
        embeddings = self.positional_encoding(self.token_embedding(input_ids))
        encoder_output, attention_weights = self.encoder(embeddings, mask=mask, return_attention=return_attention)
        
        bio_logits = self.bio_head(encoder_output)
        outputs = {
            'bio_logits': bio_logits,
            'assignee_logits': self.assignee_head(encoder_output),
            'priority_logits': self.priority_head(encoder_output),
            'deadline_logits': self.deadline_head(encoder_output),
            'encoder_output': encoder_output
        }
        if return_attention: outputs['attention_weights'] = attention_weights
        
        if bio_labels is not None:
            outputs['loss'] = self._compute_loss(bio_logits, outputs['assignee_logits'], outputs['priority_logits'], outputs['deadline_logits'], bio_labels, assignee_labels, priority_labels, deadline_labels, mask.squeeze(1).squeeze(1))
        
        return outputs
    
    def _compute_loss(self, bio_logits, assignee_logits, priority_logits, deadline_logits, bio_labels, assignee_labels, priority_labels, deadline_labels, attention_mask):
        total_loss = 0.0
        if self.use_crf:
            crf_labels = bio_labels.clone()
            crf_labels[crf_labels == -100] = 0
            total_loss += self.crf(bio_logits, crf_labels, mask=attention_mask.bool(), reduction='mean')
        else:
            total_loss += nn.functional.cross_entropy(bio_logits.view(-1, bio_logits.size(-1)), bio_labels.view(-1), ignore_index=-100, reduction='mean')
        
        if assignee_labels is not None: total_loss += nn.functional.cross_entropy(assignee_logits.view(-1, assignee_logits.size(-1)), assignee_labels.view(-1), ignore_index=-100, reduction='mean')
        if priority_labels is not None: total_loss += nn.functional.cross_entropy(priority_logits.view(-1, priority_logits.size(-1)), priority_labels.view(-1), ignore_index=-100, reduction='mean')
        if deadline_labels is not None: total_loss += nn.functional.cross_entropy(deadline_logits.view(-1, deadline_logits.size(-1)), deadline_labels.view(-1), ignore_index=-100, reduction='mean')
        
        return total_loss
    
    def decode(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            mask = attention_mask.bool() if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.bool)
            bio_predictions = self.crf.decode(outputs['bio_logits'], mask) if self.use_crf else outputs['bio_logits'].argmax(dim=-1).tolist()
            
            return {
                'bio_predictions': bio_predictions,
                'assignee_predictions': outputs['assignee_logits'].argmax(dim=-1).tolist(),
                'priority_predictions': outputs['priority_logits'].argmax(dim=-1).tolist(),
                'deadline_predictions': outputs['deadline_logits'].argmax(dim=-1).tolist()
            }
    
    @classmethod
    def from_config(cls, config: Dict) -> 'TaskExtractionTransformer':
        return cls(vocab_size=30000, d_model=config.get('d_model', 256), num_heads=config.get('num_heads', 8), num_layers=config.get('num_layers', 6), d_ff=config.get('d_ff', 1024), max_seq_len=config.get('max_seq_len', 512), num_assignees=config.get('num_assignees', 5), num_priorities=config.get('num_priorities', 4), num_bio_tags=config.get('num_bio_tags', 9), dropout=config.get('dropout', 0.1), use_crf=config.get('use_crf', True), pre_norm=config.get('pre_norm', True))
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)