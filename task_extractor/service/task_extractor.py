import torch
import re
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

from ..model.transformer import TaskExtractionTransformer
from ..data.tokenizer import TaskTokenizer
from ..data.preprocessor import MeetingPreprocessor

logger = logging.getLogger(_name_)


@dataclass
class ExtractedTask:
    """Represents an extracted task."""
    id: int
    description: str
    assigned_to: str
    deadline: Optional[str]
    priority: Optional[str]
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractionResult:
    """Result of task extraction."""
    tasks: List[ExtractedTask]
    original_text: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'tasks': [t.to_dict() for t in self.tasks],
            'original_text': self.original_text
        }


class TaskExtractorService:
    """Simple task extractor service."""
    
    # Default team members with their expertise areas
    DEFAULT_TEAM = {
        'mohit': ['backend', 'api', 'database', 'performance'],
        'lata': ['frontend', 'ui', 'design', 'css'],
        'arjun': ['testing', 'qa', 'automation'],
        'sakshi': ['devops', 'deployment', 'infrastructure']
    }
    
    # Deadline extraction patterns
    DEADLINE_PATTERNS = [
        (r'by\s+(end\s+of\s+(?:the\s+)?(?:day|week|month))', 'By {0}'),
        (r'by\s+(tomorrow|today|tonight)', 'By {0}'),
        (r'by\s+(next\s+(?:week|monday|tuesday|wednesday|thursday|friday))', 'By {0}'),
        (r'within\s+(\d+\s+(?:hours?|days?|weeks?))', 'Within {0}'),
        (r'(asap|immediately|urgent)', '{0}'),
    ]
    
    # Priority keywords
    PRIORITY_KEYWORDS = {
        'critical': ['urgent', 'asap', 'critical', 'blocker'],
        'high': ['high priority', 'important'],
        'medium': ['medium priority'],
        'low': ['low priority']
    }
    
    def _init_(self, model_path: Optional[str] = None, tokenizer_path: Optional[str] = None, device: str = 'cpu'):
        self.device = torch.device(device)
        self.tokenizer = TaskTokenizer.load(tokenizer_path) if tokenizer_path else TaskTokenizer()
        self.model = TaskExtractionTransformer.from_config({'vocab_size': self.tokenizer.vocab_size_actual or 30000})
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()
        self.preprocessor = MeetingPreprocessor.create_default()
        self.team_members = self.DEFAULT_TEAM
    
    def extract_tasks(self, text: str) -> ExtractionResult:
        """Extract tasks from text."""
        processed_text, _ = self.preprocessor.preprocess_for_model(text)
        
        # Try rule-based extraction first
        tasks = self._rule_based_extraction(text)
        
        return ExtractionResult(tasks=tasks, original_text=text)
    
    def _rule_based_extraction(self, text: str) -> List[ExtractedTask]:
        """Extract tasks using simple pattern matching."""
        tasks = []
        
        # Simple pattern: "Name, please handle/fix X"
        pattern = re.compile(
            r'(?P<assignee>\w+),?\s+(?:please\s+)?(?:handle|fix|work\s+on)\s+(?:the\s+)?(?P<task>[^.]+)',
            re.IGNORECASE
        )
        
        for match in pattern.finditer(text):
            assignee = match.group('assignee').lower()
            if assignee in self.team_members:
                task_desc = match.group('task').strip()
                
                # Extract deadline from task description
                deadline = self._extract_deadline(task_desc)
                
                # Extract priority
                priority = self._extract_priority(task_desc)
                
                tasks.append(ExtractedTask(
                    id=len(tasks) + 1,
                    description=task_desc,
                    assigned_to=assignee.title(),
                    deadline=deadline,
                    priority=priority,
                    confidence=0.8
                ))
        
        return tasks
    
    def _extract_deadline(self, text: str) -> Optional[str]:
        """Extract deadline from text."""
        text_lower = text.lower()
        for pattern, template in self.DEADLINE_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                captured = match.group(1).strip()
                if '{0}' in template:
                    return template.format(captured.title())
                return template.title()
        return None
    
    def _extract_priority(self, text: str) -> Optional[str]:
        """Extract priority from text."""
        text_lower = text.lower()
        for priority, keywords in self.PRIORITY_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    return priority.title()
        return None