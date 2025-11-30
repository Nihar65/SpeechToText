
import torch, re, logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from ..models import TaskExtractionTransformer
from ..data import TaskTokenizer, MeetingPreprocessor, BIO_TAGS, ID_TO_BIO_TAG

logger = logging.getLogger(_name_)

@dataclass
class ExtractedTask:
    id: int
    description: str
    assigned_to: str
    deadline: Optional[str]
    priority: Optional[str]
    confidence: float
    def to_dict(self): return asdict(self)

@dataclass
class ExtractionResult:
    tasks: List[ExtractedTask]
    text: str
    
class TaskExtractorService:
    def _init_(self, model_path: Optional[str] = None, tokenizer_path: Optional[str] = None, device: str = 'cpu'):
        self.device = torch.device(device)
        self.tokenizer = TaskTokenizer.load(tokenizer_path) if tokenizer_path else TaskTokenizer()
        self.model = TaskExtractionTransformer.from_config({'vocab_size': self.tokenizer.vocab_size_actual})
        if model_path: self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()
        self.preprocessor = MeetingPreprocessor.create_default()
    
    def extract_tasks(self, text: str) -> ExtractionResult:
        processed_text, _ = self.preprocessor.preprocess_for_model(text)
        # Placeholder for inference logic - typically involves tokenizing and model forward pass
        return ExtractionResult(tasks=[], text=text)