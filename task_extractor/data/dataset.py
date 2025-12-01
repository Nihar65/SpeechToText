import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Any, Tuple
import re
from dataclasses import dataclass

BIO_TAGS = {
    'O': 0, 'B-TASK': 1, 'I-TASK': 2, 'B-ASSIGNEE': 3, 'I-ASSIGNEE': 4,
    'B-DEADLINE': 5, 'I-DEADLINE': 6, 'B-PRIORITY': 7, 'I-PRIORITY': 8,
}
ID_TO_BIO_TAG = {v: k for k, v in BIO_TAGS.items()}
PRIORITY_LABELS = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
DEFAULT_ASSIGNEES = ['mohit', 'lata', 'arjun', 'sakshi']

class TaskExtractionDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: Any, max_length: int = 512, assignees: Optional[List[str]] = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.assignees = [a.lower() for a in (assignees or DEFAULT_ASSIGNEES)]
        self.assignee_to_id = {name: i for i, name in enumerate(self.assignees)}
        self.samples = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[Dict]:
        samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): samples.append(json.loads(line))
        return samples
    
    def _create_bio_labels(self, text: str, tasks: List[Dict], tokens: List[str]) -> Tuple[List[int], List[int], List[int], List[int]]:
        text_lower = text.lower()
        bio_labels = [BIO_TAGS['O']] * len(tokens)
        assignee_labels = [-100] * len(tokens)
        priority_labels = [-100] * len(tokens)
        deadline_labels = [0] * len(tokens)
        
        char_to_token = self._build_char_to_token_map(text, tokens)
        
        for task in tasks:
            # Task Description
            task_desc = task.get('description', '').lower()
            if task_desc:
                self._label_span(text_lower, task_desc, char_to_token, tokens, bio_labels, BIO_TAGS['B-TASK'], BIO_TAGS['I-TASK'])
            
            # Assignee
            assignee = task.get('assigned_to', '').lower()
            if assignee:
                self._label_span(text_lower, assignee, char_to_token, tokens, bio_labels, BIO_TAGS['B-ASSIGNEE'], BIO_TAGS['I-ASSIGNEE'])
                assignee_id = self.assignee_to_id.get(assignee, -100)
                if assignee_id != -100:
                    start_idx = text_lower.find(assignee)
                    if start_idx != -1:
                        for i in range(start_idx, start_idx + len(assignee)):
                            if i in char_to_token: assignee_labels[char_to_token[i]] = assignee_id
            
            # Deadline
            deadline = (task.get('deadline') or '').lower()
            if deadline:
                self._label_span(text_lower, deadline, char_to_token, tokens, bio_labels, BIO_TAGS['B-DEADLINE'], BIO_TAGS['I-DEADLINE'])
                start_idx = text_lower.find(deadline)
                if start_idx != -1:
                    first_token = True
                    for i in range(start_idx, start_idx + len(deadline)):
                        if i in char_to_token:
                            token_idx = char_to_token[i]
                            if first_token:
                                deadline_labels[token_idx] = 1
                                first_token = False
                            else:
                                if deadline_labels[token_idx] == 0: deadline_labels[token_idx] = 2

            # Priority
            priority = (task.get('priority') or '').lower()
            if priority:
                self._label_span(text_lower, priority, char_to_token, tokens, bio_labels, BIO_TAGS['B-PRIORITY'], BIO_TAGS['I-PRIORITY'])
                priority_id = PRIORITY_LABELS.get(priority, -100)
                if priority_id != -100:
                    match = re.search(priority + r'\s+priority', text_lower)
                    if match:
                        for i in range(match.start(), match.end()):
                            if i in char_to_token: priority_labels[char_to_token[i]] = priority_id
        
        return bio_labels, assignee_labels, priority_labels, deadline_labels
    
    def _build_char_to_token_map(self, text: str, tokens: List[str]) -> Dict[int, int]:
        char_to_token = {}
        text_lower = text.lower()
        current_pos = 0
        for token_idx, token in enumerate(tokens):
            token_lower = token.lower()
            pos = text_lower.find(token_lower, current_pos)
            if pos != -1:
                for i in range(pos, pos + len(token)): char_to_token[i] = token_idx
                current_pos = pos + len(token)
            else:
                pos = text_lower.find(token_lower)
                if pos != -1:
                    for i in range(pos, pos + len(token)):
                        if i not in char_to_token: char_to_token[i] = token_idx
        return char_to_token
    
    def _label_span(self, text: str, span: str, char_to_token: Dict[int, int], tokens: List[str], labels: List[int], b_tag: int, i_tag: int) -> None:
        start_idx = text.find(span)
        if start_idx == -1: return
        first_token = True
        labeled_tokens = set()
        for i in range(start_idx, start_idx + len(span)):
            if i in char_to_token:
                token_idx = char_to_token[i]
                if token_idx not in labeled_tokens:
                    labels[token_idx] = b_tag if first_token else i_tag
                    first_token = False
                    labeled_tokens.add(token_idx)
    
    def len(self) -> int: return len(self.samples)
    
    def getitem(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        text = sample['text']
        tasks = sample.get('tasks', [])
        tokens = self.tokenizer.tokenize(text)
        bio_labels, assignee_labels, priority_labels, deadline_labels = self._create_bio_labels(text, tasks, tokens)
        encoded = self.tokenizer.encode(text, add_special_tokens=True, max_length=self.max_length, truncation=True)
        
        # Adjust labels for special tokens and padding
        seq_len = len(encoded['input_ids'])
        def pad_labels(labels, default):
            l = [default] + labels[:self.max_length - 2] + [default]
            return l[:seq_len] + [default] * (seq_len - len(l))
            
        return {
            'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long),
            'bio_labels': torch.tensor(pad_labels(bio_labels, -100), dtype=torch.long),
            'assignee_labels': torch.tensor(pad_labels(assignee_labels, -100), dtype=torch.long),
            'priority_labels': torch.tensor(pad_labels(priority_labels, -100), dtype=torch.long),
            'deadline_labels': torch.tensor(pad_labels(deadline_labels, 0), dtype=torch.long),
            'text': text,
            'tasks': tasks
        }

@dataclass
class DataCollator:
    tokenizer: Any
    max_length: int = 512
    
    def call(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = min(max(len(f['input_ids']) for f in features), self.max_length)
        batch = {k: [] for k in ['input_ids', 'attention_mask', 'bio_labels', 'assignee_labels', 'priority_labels', 'deadline_labels']}
        
        for f in features:
            for key in batch.keys():
                val = f[key].tolist() if isinstance(f[key], torch.Tensor) else f[key]
                pad_val = self.tokenizer.pad_token_id if key == 'input_ids' else (-100 if 'labels' in key and key != 'deadline_labels' else 0)
                if len(val) < max_len: val = val + [pad_val] * (max_len - len(val))
                else: val = val[:max_len]
                batch[key].append(val)
        
        for key in batch.keys(): batch[key] = torch.tensor(batch[key], dtype=torch.long)
        return batch

def create_data_loaders(train_path: str, tokenizer: Any, batch_size: int = 32, max_length: int = 512, val_split: float = 0.1, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    dataset = TaskExtractionDataset(train_path, tokenizer, max_length)
    val_size = int(len(dataset) * val_split)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset) - val_size, val_size])
    collator = DataCollator(tokenizer, max_length)
    
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collator, pin_memory=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collator, pin_memory=True)
    )