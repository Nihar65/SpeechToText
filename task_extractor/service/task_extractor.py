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
    processed_text: str
    model_confidence: float
    extraction_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'tasks': [t.to_dict() for t in self.tasks],
            'original_text': self.original_text,
            'processed_text': self.processed_text,
            'model_confidence': self.model_confidence,
            'extraction_time_ms': self.extraction_time_ms
        }


class TaskExtractorService:
    """Simple task extractor service."""
    
    # Default team members with their expertise areas
    DEFAULT_TEAM = {
        'mohit': ['backend', 'api', 'database', 'performance', 'query', 'sql', 'server'],
        'lata': ['frontend', 'ui', 'design', 'css', 'dashboard', 'component', 'page'],
        'arjun': ['testing', 'qa', 'automation', 'test', 'bug', 'debug'],
        'sakshi': ['devops', 'deployment', 'infrastructure', 'docker', 'deploy', 'pipeline']
    }
    
    # Deadline extraction patterns
    DEADLINE_PATTERNS = [
        (r'by\s+(end\s+of\s+(?:the\s+)?(?:day|week|month))', 'By {0}'),
        (r'by\s+(tomorrow|today|tonight)', 'By {0}'),
        (r'by\s+(next\s+(?:week|month|monday|tuesday|wednesday|thursday|friday))', 'By {0}'),
        (r'by\s+(this\s+(?:week|friday|weekend))', 'By {0}'),
        (r'within\s+(\d+\s+(?:hours?|days?|weeks?))', 'Within {0}'),
        (r'in\s+(\d+\s+(?:hours?|days?|weeks?))', 'In {0}'),
        (r'(before\s+(?:the\s+)?(?:release|deployment|meeting|demo))', '{0}'),
        (r'(asap|immediately|urgent)', '{0}'),
        (r'(right\s+away)', 'Right Away'),
    ]
    
    # Priority keywords
    PRIORITY_KEYWORDS = {
        'critical': ['urgent', 'asap', 'critical', 'blocker', 'blocking', 'immediately'],
        'high': ['high priority', 'important', 'priority'],
        'medium': ['medium priority'],
        'low': ['low priority', 'when possible']
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
        import time
        start_time = time.time()
        
        processed_text, _ = self.preprocessor.preprocess_for_model(text)
        
        # Try rule-based extraction first
        tasks = self._rule_based_extraction(text)
        
        # Calculate metrics
        extraction_time = (time.time() - start_time) * 1000
        avg_confidence = sum(t.confidence for t in tasks) / len(tasks) if tasks else 0.0
        
        return ExtractionResult(
            tasks=tasks,
            original_text=text,
            processed_text=processed_text,
            model_confidence=avg_confidence,
            extraction_time_ms=extraction_time
        )
    
    def _rule_based_extraction(self, text: str) -> List[ExtractedTask]:
        """Extract tasks using simple pattern matching."""
        tasks = []
        seen_tasks = set()
        
        # Pattern 1: "Name, please handle/fix X"
        pattern1 = re.compile(
            r'(?P<assignee>\w+),?\s+(?:please\s+)?(?:handle|fix|work\s+on|complete|update)\s+(?:the\s+)?(?P<task>[^.]+)',
            re.IGNORECASE
        )
        
        # Pattern 2: "Name will handle X"
        pattern2 = re.compile(
            r'(?P<assignee>\w+)\s+will\s+(?:handle|work\s+on|do|complete)\s+(?:the\s+)?(?P<task>[^.,]+)',
            re.IGNORECASE
        )
        
        # Pattern 3: "Name, we need you to X"
        pattern3 = re.compile(
            r'(?P<assignee>\w+),?\s+(?:we\s+need\s+you\s+to|you\s+should|can\s+you)\s+(?P<task>[^.?!]+)',
            re.IGNORECASE
        )
        
        # Pattern 4: "Name, there is/are X that needs..."
        pattern4 = re.compile(
            r'(?P<assignee>\w+),?\s+there\s+(?:is|are)\s+(?:a\s+)?(?P<task>[^.]+?)(?:\s+that\s+needs|\s+needing)',
            re.IGNORECASE
        )
        
        # Pattern 5: "X should be done by Name"
        pattern5 = re.compile(
            r'(?P<task>[^.]+?)\s+(?:should\s+be\s+done|needs\s+to\s+be\s+handled)\s+by\s+(?P<assignee>\w+)',
            re.IGNORECASE
        )
        
        # Pattern 6: Generic tasks "we need to X" - infer assignee
        pattern6 = re.compile(
            r'(?:we\s+need\s+to|someone\s+should|we\s+should|let\'?s)\s+(?P<task>[^.?!]+)',
            re.IGNORECASE
        )
        
        for pattern in [pattern1, pattern2, pattern3, pattern4, pattern5]:
            for match in pattern.finditer(text):
                assignee = match.group('assignee').lower()
                if assignee in self.team_members:
                    task_desc = match.group('task').strip()
                    
                    # Clean task description
                    task_desc = self._clean_description(task_desc)
                    
                    # Skip duplicates or invalid tasks
                    task_key = task_desc.lower()[:50]
                    if task_key in seen_tasks or len(task_desc) < 5:
                        continue
                    seen_tasks.add(task_key)
                    
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
                        confidence=0.85
                    ))
        
        # Handle generic tasks with inferred assignee
        for match in pattern6.finditer(text):
            task_desc = match.group('task').strip()
            
            # Clean task description
            task_desc = self._clean_description(task_desc)
            
            # Skip duplicates or invalid tasks
            task_key = task_desc.lower()[:50]
            if task_key in seen_tasks or len(task_desc) < 5:
                continue
            
            # Infer assignee based on content
            assignee = self._infer_assignee(task_desc)
            if assignee:
                seen_tasks.add(task_key)
                
                deadline = self._extract_deadline(task_desc)
                priority = self._extract_priority(task_desc)
                
                tasks.append(ExtractedTask(
                    id=len(tasks) + 1,
                    description=task_desc,
                    assigned_to=assignee,
                    deadline=deadline,
                    priority=priority,
                    confidence=0.70
                ))
        
        return tasks
    
    def _clean_description(self, description: str) -> str:
        """Clean up task description."""
        # Remove priority mentions
        description = re.sub(r',?\s*(?:high|medium|low|critical)\s+priority.*$', '', description, flags=re.IGNORECASE)
        # Remove deadline phrases that clutter description
        description = re.sub(r'\s+by\s+(?:tomorrow|today|end\s+of\s+\w+).*$', '', description, flags=re.IGNORECASE)
        # Clean up whitespace
        description = re.sub(r'\s+', ' ', description).strip()
        # Remove trailing punctuation
        description = description.strip(' .,;:')
        return description
    
    def _infer_assignee(self, task_description: str) -> Optional[str]:
        """Infer assignee based on task keywords and team expertise with weighted scoring."""
        task_lower = task_description.lower()
        scores = {}
        
        for member, keywords in self.team_members.items():
            score = 0
            for keyword in keywords:
                if keyword in task_lower:
                    # Give higher weight to longer/more specific keywords
                    weight = len(keyword.split()) + 1
                    score += weight
            
            if score > 0:
                scores[member] = score
        
        if scores:
            best_member = max(scores, key=scores.get)
            return best_member.title()
        return None
    
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