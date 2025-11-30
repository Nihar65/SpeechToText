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
    
    # Default team members with weighted expertise keywords
    DEFAULT_TEAM_WEIGHTED = {
        'mohit': {
            'backend': 3, 'api': 3, 'database': 3, 'db': 3, 'sql': 3, 'query': 3, 'queries': 3,
            'performance': 3, 'optimization': 3, 'optimize': 3, 'slow': 2, 'cache': 3,
            'server': 2, 'endpoint': 2, 'rest': 3
        },
        'lata': {
            'frontend': 3, 'ui': 3, 'ux': 3, 'design': 3, 'redesign': 3, 'layout': 2,
            'css': 2, 'component': 2, 'page': 2, 'dashboard': 3, 'profile': 3,
            'responsive': 2, 'mobile': 2
        },
        'arjun': {
            'test': 3, 'testing': 3, 'tests': 3, 'test suite': 3, 'qa': 3,
            'bug': 2, 'debug': 2, 'review': 2, 'regression': 2
        },
        'sakshi': {
            'deploy': 3, 'deployment': 3, 'devops': 3, 'ci': 3, 'cd': 3, 'pipeline': 3,
            'docker': 3, 'release': 3, 'login': 3, 'hotfix': 3
        }
    }
    
    # Default team members with their expertise areas
    DEFAULT_TEAM = {
        'mohit': ['backend', 'api', 'database', 'performance', 'query', 'sql', 'server'],
        'lata': ['frontend', 'ui', 'design', 'css', 'dashboard', 'component', 'page'],
        'arjun': ['testing', 'qa', 'automation', 'test', 'bug', 'debug'],
        'sakshi': ['devops', 'deployment', 'infrastructure', 'docker', 'deploy', 'pipeline']
    }
    
    # Deadline extraction patterns
    DEADLINE_PATTERNS = [
        (r'by\s+(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)', 'By {0}'),
        (r'by\s+(end\s+of\s+(?:the\s+)?(?:day|week|month))', 'By {0}'),
        (r'by\s+(tomorrow|today|tonight)', 'By {0}'),
        (r'by\s+(next\s+(?:week|month|monday|tuesday|wednesday|thursday|friday))', 'By {0}'),
        (r'by\s+(this\s+(?:week|friday|weekend))', 'By {0}'),
        (r"(before\s+friday'?s?\s+release)", '{0}'),
        (r'(before\s+(?:the\s+)?(?:release|deployment|meeting|demo))', '{0}'),
        (r'within\s+(\d+\s+(?:hours?|days?|weeks?))', 'Within {0}'),
        (r'in\s+(\d+\s+(?:hours?|days?|weeks?))', 'In {0}'),
        (r'(asap|immediately|urgent)', '{0}'),
        (r'(right\s+away)', 'Right Away'),
        (r'by\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm))', 'By {0}'),
        (r'by\s+(morning|afternoon|evening)', 'By {0}'),
    ]
    
    # Priority keywords
    PRIORITY_KEYWORDS = {
        'critical': ['urgent', 'asap', 'critical', 'blocker', 'blocking', 'immediately'],
        'high': ['high priority', 'important', 'priority'],
        'medium': ['medium priority'],
        'low': ['low priority', 'when possible']
    }
    
    def _init_(self, model_path: Optional[str] = None, tokenizer_path: Optional[str] = None, device: str = 'cpu', team_members: Optional[Dict] = None):
        self.device = torch.device(device)
        self.tokenizer = TaskTokenizer.load(tokenizer_path) if tokenizer_path else TaskTokenizer()
        self.model = TaskExtractionTransformer.from_config({'vocab_size': self.tokenizer.vocab_size_actual or 30000})
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()
        self.preprocessor = MeetingPreprocessor.create_default()
        self.team_members = team_members or self.DEFAULT_TEAM
        self.member_list = list(self.team_members.keys())
    
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
        valid_members = list(self.team_members.keys())
        
        # Pattern 1: "Name, please handle/fix X"
        pattern1 = re.compile(
            r'(?P<assignee>\w+),?\s+(?:please\s+)?(?:handle|fix|work\s+on|complete|update|create|implement|design|test|review|optimize)\s+(?:the\s+)?(?P<task>[^.]+)',
            re.IGNORECASE
        )
        
        # Pattern 2: "Name will handle X"
        pattern2 = re.compile(
            r'(?P<assignee>\w+)\s+will\s+(?:handle|work\s+on|do|complete|take\s+care\s+of)\s+(?:the\s+)?(?P<task>[^.,]+)',
            re.IGNORECASE
        )
        
        # Pattern 3: "Name, we need you to X"
        pattern3 = re.compile(
            r'(?P<assignee>\w+),?\s+(?:we\s+need\s+you\s+to|you\s+should|can\s+you|could\s+you)\s+(?P<task>[^.?!]+)',
            re.IGNORECASE
        )
        
        # Pattern 4: "Name, there is/are X that needs..."
        pattern4 = re.compile(
            r'(?P<assignee>\w+),?\s+there\s+(?:is|are)\s+(?:a\s+)?(?P<task>[^.]+?)(?:\s+that\s+needs|\s+needing)',
            re.IGNORECASE
        )
        
        # Pattern 5: "X should be done by Name"
        pattern5 = re.compile(
            r'(?P<task>[^.]+?)\s+(?:should\s+be\s+done|needs\s+to\s+be\s+(?:done|handled|completed|fixed))\s+(?:by|to)\s+(?P<assignee>\w+)',
            re.IGNORECASE
        )
        
        # Pattern 6: "Name, make sure to X"
        pattern6 = re.compile(
            r'(?P<assignee>\w+),?\s+(?:make\s+sure\s+to|ensure\s+(?:that\s+)?|please\s+ensure)\s+(?P<task>[^.?!]+)',
            re.IGNORECASE
        )
        
        # Pattern 7: Generic tasks "we need to X" - infer assignee
        pattern7 = re.compile(
            r'(?:we\s+need\s+to|someone\s+should|we\s+should|let\'?s)\s+(?P<task>[^.?!]+)',
            re.IGNORECASE
        )
        
        # Pattern 8: "there is/got/have a X" - generic task detection
        pattern8 = re.compile(
            r'(?:there\s+is|there\'s|got|have|has)\s+(?:a\s+)?(?P<task>(?:bug|issue|problem|error|task|feature)[^.]*?)(?:\s+(?:and\s+)?(?:it\s+)?(?:needs|that\s+needs))?',
            re.IGNORECASE
        )
        
        for pattern in [pattern1, pattern2, pattern3, pattern4, pattern5, pattern6]:
            for match in pattern.finditer(text):
                assignee = match.group('assignee').lower()
                if assignee in valid_members:
                    task_desc = match.group('task').strip()
                    
                    # Clean task description
                    task_desc = self._clean_description(task_desc)
                    
                    # Skip duplicates or invalid tasks
                    task_key = task_desc.lower()[:50]
                    if task_key in seen_tasks or len(task_desc) < 5:
                        continue
                    seen_tasks.add(task_key)
                    
                    # Extract deadline from context around the match
                    deadline = self._extract_deadline_from_context(text, match.start(), match.end())
                    if not deadline:
                        deadline = self._extract_deadline(task_desc)
                    
                    # Extract priority
                    priority = self._extract_priority(task_desc)
                    
                    tasks.append(ExtractedTask(
                        id=len(tasks) + 1,
                        description=task_desc,
                        assigned_to=assignee.title(),
                        deadline=deadline,
                        priority=priority,
                        confidence=0.90
                    ))
        
        # Handle generic tasks with inferred assignee
        for pattern in [pattern7, pattern8]:
            for match in pattern.finditer(text):
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
                    
                    deadline = self._extract_deadline_from_context(text, match.start(), match.end())
                    if not deadline:
                        deadline = self._extract_deadline(task_desc)
                    priority = self._extract_priority(task_desc)
                    
                    tasks.append(ExtractedTask(
                        id=len(tasks) + 1,
                        description=task_desc,
                        assigned_to=assignee,
                        deadline=deadline,
                        priority=priority,
                        confidence=0.75
                    ))
        
        return tasks
    
    def _clean_description(self, description: str) -> str:
        """Clean up task description."""
        # Remove priority mentions
        description = re.sub(r',?\s*(?:high|medium|low|critical)\s+priority.*$', '', description, flags=re.IGNORECASE)
        # Remove deadline phrases that clutter description
        description = re.sub(r'\s+by\s+(?:tomorrow|today|end\s+of\s+\w+).*$', '', description, flags=re.IGNORECASE)
        description = re.sub(r'\s+(?:and\s+)?(?:it\s+)?(?:needs|should)\s+(?:to\s+be\s+)?(?:done|resolved|fixed|handled|completed).*$', '', description, flags=re.IGNORECASE)
        # Clean up whitespace
        description = re.sub(r'\s+', ' ', description).strip()
        # Remove trailing punctuation
        description = description.strip(' .,;:')
        return description
    
    def assign_based_on_expertise(self, task_description: str) -> tuple:
        """
        Suggest best assignee based on task description and expertise.
        
        Args:
            task_description: Task description text
        
        Returns:
            Tuple of (suggested_assignee, confidence)
        """
        assignee = self._infer_assignee(task_description)
        if assignee:
            # Calculate confidence based on number of keyword matches
            task_lower = task_description.lower()
            max_score = 0
            for member, keywords_weights in self.DEFAULT_TEAM_WEIGHTED.items():
                score = sum(weight for kw, weight in keywords_weights.items() if kw in task_lower)
                if score > max_score:
                    max_score = score
            
            confidence = min(max_score / 10.0, 1.0)
            return (assignee, confidence)
        
        return (None, 0.0)
    
    def save(self, model_path: str, tokenizer_path: str):
        """Save model and tokenizer."""
        import torch
        torch.save(self.model.state_dict(), model_path)
        self.tokenizer.save(tokenizer_path)
    
    @classmethod
    def load(cls, model_path: str, tokenizer_path: str, device: Optional[str] = None) -> 'TaskExtractorService':
        """Load service from saved files."""
        return cls(model_path=model_path, tokenizer_path=tokenizer_path, device=device)
    
    def _infer_assignee(self, task_description: str) -> Optional[str]:
        """Infer assignee based on task keywords and team expertise with weighted scoring."""
        task_lower = task_description.lower()
        scores = {}
        
        # Use weighted scoring from DEFAULT_TEAM_WEIGHTED
        for member, keywords_weights in self.DEFAULT_TEAM_WEIGHTED.items():
            score = 0
            
            # Sort keywords by length (descending) to match longer phrases first
            sorted_keywords = sorted(keywords_weights.keys(), key=len, reverse=True)
            
            for keyword in sorted_keywords:
                if keyword in task_lower:
                    weight = keywords_weights[keyword]
                    # Bonus for longer/more specific matches
                    specificity_bonus = len(keyword.split()) - 1
                    score += weight + specificity_bonus
            
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
                captured = match.group(1).strip() if match.lastindex and match.lastindex >= 1 else match.group(0)
                captured = captured.strip()
                
                # Format using template
                if '{0}' in template:
                    # Capitalize properly, handle possessives
                    formatted = ' '.join(word.capitalize() if not word.endswith("'s") 
                                        else word[:-2].capitalize() + "'s" 
                                        for word in captured.split())
                    return template.format(formatted)
                return template.title()
        return None
    
    def _extract_deadline_from_context(self, text: str, task_start: int, task_end: int) -> Optional[str]:
        """Extract deadline from the context around a task mention."""
        # Look in a window around the task (before and after)
        context_start = max(0, task_start - 50)
        context_end = min(len(text), task_end + 100)
        context = text[context_start:context_end]
        
        return self._extract_deadline(context)
    
    def _extract_priority(self, text: str) -> Optional[str]:
        """Extract priority from text."""
        text_lower = text.lower()
        for priority, keywords in self.PRIORITY_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    return priority.title()
        
        # Check for blocking/urgent indicators
        if re.search(r'blocking|urgent|asap|immediately', text_lower):
            return 'Critical'
        
        return None