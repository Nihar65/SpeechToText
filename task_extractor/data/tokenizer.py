
import json
import re
from collections import Counter
from typing import Dict, List, Optional, Set

class TaskTokenizer:
    # Special tokens
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    CLS_TOKEN = "[CLS]"
    SEP_TOKEN = "[SEP]"
    MASK_TOKEN = "[MASK]"
    
    # Entity-specific tokens
    TASK_START = "[TASK]"
    TASK_END = "[/TASK]"
    ASSIGNEE_START = "[ASSIGNEE]"
    ASSIGNEE_END = "[/ASSIGNEE]"
    DEADLINE_START = "[DEADLINE]"
    DEADLINE_END = "[/DEADLINE]"
    PRIORITY_START = "[PRIORITY]"
    PRIORITY_END = "[/PRIORITY]"
    
    DEFAULT_SPECIAL_TOKENS = [
        PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN,
        TASK_START, TASK_END, ASSIGNEE_START, ASSIGNEE_END,
        DEADLINE_START, DEADLINE_END, PRIORITY_START, PRIORITY_END
    ]
    
    def __init__(self, vocab_size: int = 30000, min_frequency: int = 2, special_tokens: Optional[List[str]] = None):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens or self.DEFAULT_SPECIAL_TOKENS
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        for i, token in enumerate(self.special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
        
        self.word_pattern = re.compile(r'\w+|[^\w\s]')
        self.team_members: Set[str] = set()
    
    @property
    def vocab_size_actual(self) -> int: return len(self.token_to_id)
    @property
    def pad_token_id(self) -> int: return self.token_to_id[self.PAD_TOKEN]
    @property
    def unk_token_id(self) -> int: return self.token_to_id[self.UNK_TOKEN]
    @property
    def cls_token_id(self) -> int: return self.token_to_id[self.CLS_TOKEN]
    @property
    def sep_token_id(self) -> int: return self.token_to_id[self.SEP_TOKEN]
    
    def _get_word_frequencies(self, texts: List[str]) -> Counter:
        word_freq = Counter()
        for text in texts:
            words = self.word_pattern.findall(text.lower())
            word_freq.update(words)
        return word_freq
    
    def _extract_team_members(self, texts: List[str]) -> None:
        name_pattern = re.compile(r'([A-Z][a-z]+),?\s+please\s+handle', re.IGNORECASE)
        for text in texts:
            matches = name_pattern.findall(text)
            self.team_members.update(match.lower() for match in matches)
    
    def train(self, texts: List[str], verbose: bool = True) -> None:
        if verbose: print("Training tokenizer...")
        self._extract_team_members(texts)
        word_freq = self._get_word_frequencies(texts)
        filtered_words = [(word, freq) for word, freq in word_freq.items() if freq >= self.min_frequency]
        filtered_words.sort(key=lambda x: -x[1])
        
        current_id = len(self.token_to_id)
        max_words = self.vocab_size - current_id
        
        for word, freq in filtered_words[:max_words]:
            if word not in self.token_to_id:
                self.token_to_id[word] = current_id
                self.id_to_token[current_id] = word
                current_id += 1
        
        if verbose: print(f"Vocabulary size: {len(self.token_to_id)}")
    
    def tokenize(self, text: str) -> List[str]:
        return self.word_pattern.findall(text.lower())
    
    def encode(self, text: str, add_special_tokens: bool = True, max_length: Optional[int] = None, padding: bool = False, truncation: bool = True) -> Dict[str, List[int]]:
        tokens = self.tokenize(text)
        ids = [self.token_to_id.get(token, self.token_to_id[self.UNK_TOKEN]) for token in tokens]
        
        if add_special_tokens:
            ids = [self.token_to_id[self.CLS_TOKEN]] + ids + [self.token_to_id[self.SEP_TOKEN]]
        
        if truncation and max_length and len(ids) > max_length:
            ids = ids[:max_length - 1] + [self.token_to_id[self.SEP_TOKEN]]