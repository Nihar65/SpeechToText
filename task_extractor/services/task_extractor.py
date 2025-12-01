"""
Task Extractor Service
======================
Main service for extracting tasks from meeting transcripts.
"""

import torch
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import re

from ..model import TaskExtractionTransformer
from ..data import TaskTokenizer, MeetingPreprocessor
from ..data.dataset import BIO_TAGS, ID_TO_BIO_TAG, PRIORITY_LABELS

logger = logging.getLogger(__name__)


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
    """
    Production-ready task extraction service.
    
    Combines the transformer model with preprocessing to provide
    end-to-end task extraction from meeting transcripts.
    
    Args:
        model_path: Path to saved model weights
        tokenizer_path: Path to saved tokenizer
        config: Model configuration
        device: Device to run inference on
        team_members: List of known team members
    """
    
    # Default team members with comprehensive expertise keywords and weights
    # Format: {member: {keyword: weight}} - higher weight = more relevant
    DEFAULT_TEAM_WEIGHTED = {
        'mohit': {
            # Backend & API (core expertise)
            'backend': 3, 'api': 3, 'api documentation': 3, 'rest': 3, 'graphql': 3, 'endpoint': 3, 'server': 2, 'microservice': 3,
            # Database (core expertise)
            'database': 3, 'db': 3, 'sql': 3, 'query': 3, 'queries': 3, 'mysql': 3, 'postgres': 3, 'mongodb': 3, 'redis': 3,
            # Performance (core expertise)
            'performance': 3, 'optimization': 3, 'optimize': 3, 'speed': 2, 'slow': 2, 'latency': 2, 'cache': 3, 'caching': 3,
            # Authentication
            'authentication': 2, 'auth': 2, 'session': 2, 'token': 2, 'jwt': 2, 'middleware': 2,
            # General backend
            'service': 1, 'logic': 1, 'algorithm': 2, 'data structure': 2
        },
        'lata': {
            # Frontend & UI (core expertise)
            'frontend': 3, 'front-end': 3, 'ui': 3, 'ux': 3, 'user interface': 3, 'user experience': 3,
            # Design (core expertise)
            'design': 3, 'redesign': 3, 'layout': 3, 'style': 2, 'css': 3, 'scss': 3, 'styling': 3,
            # Components (core expertise)
            'component': 3, 'page': 3, 'view': 2, 'screen': 2, 'modal': 2, 'form': 2, 'button': 2, 'menu': 2,
            # Frameworks
            'react': 3, 'vue': 3, 'angular': 3, 'html': 2, 'javascript': 2, 'typescript': 2,
            # Specific UI elements
            'dashboard': 3, 'profile': 3, 'user profile': 3, 'navigation': 2, 'header': 2, 'footer': 2, 'sidebar': 2, 'responsive': 2,
            # Visual
            'icon': 2, 'image': 2, 'animation': 2, 'color': 2, 'theme': 2,
            # Documentation (UI/UX docs)
            'user documentation': 2, 'user guide': 2
        },
        'arjun': {
            # Testing (core expertise)
            'test': 3, 'testing': 3, 'tests': 3, 'test suite': 3, 'unit test': 3, 'integration test': 3, 'e2e': 3,
            # QA (core expertise)
            'qa': 3, 'quality': 3, 'quality assurance': 3, 'verification': 2, 'validation': 2,
            # Bugs & Issues (core expertise)
            'bug': 3, 'bugs': 3, 'issue': 2, 'issues': 2, 'fix': 2, 'debug': 3, 'debugging': 3, 'error': 2, 'errors': 2,
            # Automation
            'automation': 3, 'automated': 3, 'selenium': 3, 'cypress': 3, 'jest': 3, 'pytest': 3,
            # Review
            'review': 2, 'code review': 2, 'regression': 3, 'coverage': 2, 'test case': 3
        },
        'sakshi': {
            # DevOps (core expertise)
            'devops': 3, 'deploy': 3, 'deployment': 3, 'release': 3, 'rollout': 3, 'rollback': 3,
            # CI/CD (core expertise)
            'ci': 3, 'cd': 3, 'cicd': 3, 'ci/cd': 3, 'pipeline': 3, 'jenkins': 3, 'github actions': 3, 'gitlab': 3,
            # Infrastructure (core expertise)
            'infrastructure': 3, 'infra': 3, 'cloud': 3, 'aws': 3, 'azure': 3, 'gcp': 3,
            # Containers
            'docker': 3, 'container': 3, 'kubernetes': 3, 'k8s': 3, 'helm': 3,
            # Monitoring
            'monitoring': 3, 'logs': 2, 'logging': 2, 'metrics': 2, 'alerts': 2, 'grafana': 3, 'prometheus': 3,
            # Security & Auth
            'security': 3, 'ssl': 3, 'certificate': 3, 'firewall': 3, 'login': 3, 'login bug': 3,
            # Environment
            'environment': 2, 'config': 2, 'configuration': 2, 'secrets': 3, 'env': 2
        }
    }
    
    # Simple list format for backward compatibility
    DEFAULT_TEAM = {
        'mohit': ['backend', 'api', 'database', 'performance', 'optimization', 'query', 'queries', 'sql', 'server', 'endpoint', 'cache', 'slow'],
        'lata': ['frontend', 'ui', 'ux', 'design', 'css', 'page', 'component', 'dashboard', 'profile', 'user profile', 'redesign', 'layout', 'responsive'],
        'arjun': ['testing', 'qa', 'automation', 'quality', 'test', 'test suite', 'unit test', 'regression', 'coverage', 'verification'],
        'sakshi': ['devops', 'deployment', 'infrastructure', 'monitoring', 'deploy', 'release', 'docker', 'login', 'security', 'pipeline', 'ci', 'cd', 'bug', 'login bug']
    }
    
    # Deadline patterns for extraction - patterns return (full_match, deadline_value)
    DEADLINE_PATTERNS = [
        # Explicit dates (by DATE)
        (r'by\s+(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)', 'By {0}'),
        (r'by\s+(\w+\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)', 'By {0}'),
        (r'due\s+(?:on\s+)?(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)', 'Due {0}'),
        (r'due\s+(?:on\s+)?(\w+\s+\d{1,2}(?:st|nd|rd|th)?)', 'Due {0}'),
        # Relative time
        (r'by\s+(end\s+of\s+(?:the\s+)?(?:day|week|month|sprint))', 'By {0}'),
        (r'by\s+(tomorrow|today|tonight)', 'By {0}'),
        (r'by\s+(next\s+(?:week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday))', 'By {0}'),
        (r'by\s+(this\s+(?:week|month|friday|weekend))', 'By {0}'),
        (r'within\s+(\d+\s+(?:hours?|days?|weeks?))', 'Within {0}'),
        (r'in\s+(\d+\s+(?:hours?|days?|weeks?))', 'In {0}'),
        # Before events - capture the full phrase
        (r'(before\s+(?:the\s+)?(?:release|deploy|deployment|launch|meeting|demo|sprint\s*end))', '{0}'),
        (r'(before\s+(?:we\s+)?(?:deploy|release|launch|ship))', '{0}'),
        # ASAP indicators
        (r'(asap)', 'ASAP'),
        (r'(immediately|immediate)', 'Immediately'),
        (r'(urgent(?:ly)?)', 'Urgent'),
        (r'(right\s+away)', 'Right Away'),
        (r'(as\s+soon\s+as\s+possible)', 'ASAP'),
        # Time of day
        (r'by\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm))', 'By {0}'),
        (r'by\s+(morning|afternoon|evening|noon|midnight)', 'By {0}'),
        # Days of week
        (r'by\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)', 'By {0}'),
        (r'on\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)', 'On {0}'),
        # End of period
        (r'(end\s+of\s+(?:the\s+)?(?:day|week|month|quarter|year|sprint))', 'By {0}'),
    ]
    
    # Priority mapping (reverse)
    PRIORITY_ID_TO_LABEL = {v: k for k, v in PRIORITY_LABELS.items()}
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        config: Optional[Dict] = None,
        device: Optional[str] = None,
        team_members: Optional[Dict[str, List[str]]] = None
    ):
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Team configuration
        self.team_members = team_members or self.DEFAULT_TEAM
        self.member_list = list(self.team_members.keys())
        
        # Load or create tokenizer
        if tokenizer_path and Path(tokenizer_path).exists():
            self.tokenizer = TaskTokenizer.load(tokenizer_path)
        else:
            self.tokenizer = TaskTokenizer()
            self.tokenizer.team_members = set(self.member_list)
        
        # Load or create model
        default_config = {
            'vocab_size': self.tokenizer.vocab_size_actual or 30000,
            'd_model': 256,
            'num_heads': 8,
            'num_layers': 6,
            'd_ff': 1024,
            'max_seq_len': 512,
            'num_assignees': len(self.member_list),
            'num_priorities': 4,
            'num_bio_tags': len(BIO_TAGS),
            'dropout': 0.1,
            'use_crf': True,
            'pre_norm': True
        }
        
        if config:
            default_config.update(config)
        
        self.model = TaskExtractionTransformer.from_config(default_config)
        
        if model_path and Path(model_path).exists():
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Create preprocessor
        self.preprocessor = MeetingPreprocessor.create_default()
        
        # Assignee ID mapping
        self.assignee_id_to_name = {i: name for i, name in enumerate(self.member_list)}
        self.assignee_name_to_id = {name: i for i, name in enumerate(self.member_list)}
    
    def extract_tasks(
        self,
        text: str,
        use_preprocessing: bool = True,
        confidence_threshold: float = 0.5
    ) -> ExtractionResult:
        """
        Extract tasks from meeting transcript text.
        
        Args:
            text: Meeting transcript text
            use_preprocessing: Whether to preprocess text
            confidence_threshold: Minimum confidence for task inclusion
        
        Returns:
            ExtractionResult with extracted tasks
        """
        import time
        start_time = time.time()
        
        original_text = text
        
        # Preprocess if requested
        if use_preprocessing:
            text, _ = self.preprocessor.preprocess_for_model(text)
        
        # First try rule-based extraction (more reliable for structured data)
        tasks = self._rule_based_extraction(text)
        
        # If rule-based found tasks, use them
        if tasks:
            extraction_time = (time.time() - start_time) * 1000
            return ExtractionResult(
                tasks=tasks,
                original_text=original_text,
                processed_text=text,
                model_confidence=0.95,  # High confidence for rule-based
                extraction_time_ms=extraction_time
            )
        
        # Fall back to model-based extraction for unstructured text
        # Encode text
        encoded = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=512,
            padding=True,
            truncation=True
        )
        
        input_ids = torch.tensor([encoded['input_ids']], device=self.device)
        attention_mask = torch.tensor([encoded['attention_mask']], device=self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            predictions = self.model.decode(input_ids, attention_mask)
        
        # Parse predictions into tasks
        tasks = self._parse_predictions(
            text,
            encoded['input_ids'],
            predictions,
            outputs,
            confidence_threshold
        )
        
        # Calculate overall confidence
        bio_probs = torch.softmax(outputs['bio_logits'], dim=-1)
        model_confidence = bio_probs.max(dim=-1).values.mean().item()
        
        extraction_time = (time.time() - start_time) * 1000
        
        return ExtractionResult(
            tasks=tasks,
            original_text=original_text,
            processed_text=text,
            model_confidence=model_confidence,
            extraction_time_ms=extraction_time
        )
    
    def _parse_predictions(
        self,
        text: str,
        input_ids: List[int],
        predictions: Dict,
        outputs: Dict,
        confidence_threshold: float
    ) -> List[ExtractedTask]:
        """Parse model predictions into structured tasks."""
        bio_preds = predictions['bio_predictions'][0]
        assignee_preds = predictions['assignee_predictions'][0]
        priority_preds = predictions['priority_predictions'][0]
        deadline_preds = predictions['deadline_predictions'][0]
        
        # Decode tokens
        tokens = [self.tokenizer.id_to_token.get(id, '[UNK]') for id in input_ids]
        
        # Extract task spans using BIO tags
        tasks = []
        current_task = None
        task_id = 0
        
        for i, (bio_tag, token) in enumerate(zip(bio_preds, tokens)):
            tag_name = ID_TO_BIO_TAG.get(bio_tag, 'O')
            
            if tag_name == 'B-TASK':
                # Start new task
                if current_task:
                    tasks.append(self._finalize_task(current_task, task_id, confidence_threshold))
                    task_id += 1
                
                current_task = {
                    'tokens': [token],
                    'assignee_ids': [],
                    'priority_ids': [],
                    'deadline_tokens': [],
                    'start_idx': i
                }
            
            elif tag_name == 'I-TASK' and current_task:
                current_task['tokens'].append(token)
            
            elif tag_name in ['B-ASSIGNEE', 'I-ASSIGNEE'] and current_task:
                assignee_id = assignee_preds[i]
                if assignee_id < len(self.member_list):
                    current_task['assignee_ids'].append(assignee_id)
            
            elif tag_name in ['B-PRIORITY', 'I-PRIORITY'] and current_task:
                priority_id = priority_preds[i]
                if priority_id < len(PRIORITY_LABELS):
                    current_task['priority_ids'].append(priority_id)
            
            elif tag_name in ['B-DEADLINE', 'I-DEADLINE'] and current_task:
                current_task['deadline_tokens'].append(token)
            
            elif tag_name == 'O' and current_task and len(current_task['tokens']) > 0:
                # Check for deadline in O-tagged tokens after task
                if deadline_preds[i] > 0:
                    current_task['deadline_tokens'].append(token)
        
        # Don't forget last task
        if current_task:
            tasks.append(self._finalize_task(current_task, task_id, confidence_threshold))
        
        # Filter by confidence
        tasks = [t for t in tasks if t is not None and t.confidence >= confidence_threshold]
        
        # If model didn't extract tasks, try rule-based fallback
        if not tasks:
            tasks = self._rule_based_extraction(text)
        
        return tasks
    
    def _finalize_task(
        self,
        task_data: Dict,
        task_id: int,
        confidence_threshold: float
    ) -> Optional[ExtractedTask]:
        """Finalize a task from parsed data."""
        # Get task description - join tokens properly
        # If tokens are single characters, join without spaces
        tokens = task_data['tokens']
        if tokens and all(len(t) == 1 for t in tokens):
            # Character-level tokens - join without spaces
            description = ''.join(tokens)
        else:
            description = ' '.join(tokens)
        description = self._clean_description(description)
        
        if not description or len(description) < 3:
            return None
        
        # Get assignee (most common)
        assignee = 'unassigned'
        if task_data['assignee_ids']:
            from collections import Counter
            most_common = Counter(task_data['assignee_ids']).most_common(1)
            if most_common:
                assignee_id = most_common[0][0]
                assignee = self.assignee_id_to_name.get(assignee_id, 'unassigned')
        
        # Get priority
        priority = None
        if task_data['priority_ids']:
            from collections import Counter
            most_common = Counter(task_data['priority_ids']).most_common(1)
            if most_common:
                priority_id = most_common[0][0]
                priority = self.PRIORITY_ID_TO_LABEL.get(priority_id)
        
        # Get deadline
        deadline = None
        if task_data['deadline_tokens']:
            deadline_tokens = task_data['deadline_tokens']
            if deadline_tokens and all(len(t) == 1 for t in deadline_tokens):
                deadline = ''.join(deadline_tokens)
            else:
                deadline = ' '.join(deadline_tokens)
            deadline = self._clean_deadline(deadline)
        
        # Calculate confidence (simplified)
        confidence = 0.8  # Base confidence
        if assignee != 'unassigned':
            confidence += 0.1
        if priority:
            confidence += 0.05
        if deadline:
            confidence += 0.05
        
        return ExtractedTask(
            id=task_id + 1,
            description=description,
            assigned_to=assignee.title(),
            deadline=deadline,
            priority=priority.title() if priority else None,
            confidence=min(confidence, 1.0)
        )
    
    def _clean_description(self, description: str) -> str:
        """Clean task description text."""
        # Remove special tokens
        description = re.sub(r'\[.*?\]', '', description)
        # Remove redundant phrases
        description = re.sub(r'\s+and\s+it\s+needs\s+to\s+be\s+resolved.*$', '', description, flags=re.IGNORECASE)
        description = re.sub(r'\s+(?:it\s+)?needs\s+to\s+be\s+(?:resolved|fixed|done|handled|completed).*$', '', description, flags=re.IGNORECASE)
        description = re.sub(r'\s+should\s+be\s+(?:done|fixed|resolved|handled|completed).*$', '', description, flags=re.IGNORECASE)
        description = re.sub(r'\s+that\s+needs\s+(?:to\s+be\s+)?(?:immediate\s+)?attention.*$', '', description, flags=re.IGNORECASE)
        # Remove priority mentions at end
        description = re.sub(r',?\s*(?:by\s+)?(?:highest|high|medium|low|critical)\s+priority.*$', '', description, flags=re.IGNORECASE)
        # Clean up spaces
        description = re.sub(r'\s+', ' ', description)
        # Remove leading/trailing punctuation
        description = description.strip(' .,;:')
        return description
    
    def _clean_deadline(self, deadline: str) -> str:
        """Clean deadline text."""
        deadline = re.sub(r'\[.*?\]', '', deadline)
        deadline = re.sub(r'\s+', ' ', deadline)
        deadline = deadline.strip(' .,;:')
        return deadline if deadline else None
    
    def _rule_based_extraction(self, text: str) -> List[ExtractedTask]:
        """Rule-based task extraction with multiple patterns."""
        tasks = []
        text_lower = text.lower()
        seen_tasks = set()  # Avoid duplicates
        
        # Pattern 1: "Name, please handle the task: Description. It's priority and should be done by deadline."
        pattern1 = re.compile(
            r'(?P<assignee>\w+),?\s+please\s+handle\s+(?:the\s+)?(?:task:?\s*)?(?P<task>[^.]+)\.'
            r'(?:\s*It\'?s\s+(?P<priority>\w+)\s+priority)?'
            r'(?:\s*(?:and\s+)?should\s+be\s+done\s+by\s+(?P<deadline>[^.\n]+))?',
            re.IGNORECASE
        )
        
        # Pattern 2: "Name please/will fix/handle/work on X by deadline"
        pattern2 = re.compile(
            r'(?P<assignee>\w+),?\s+(?:please\s+)?(?:will\s+)?(?:can\s+you\s+)?'
            r'(?:fix|handle|work\s+on|complete|update|create|implement|design|test|review|write|optimize|tackle)\s+'
            r'(?:the\s+)?(?P<task>.+?)'
            r'(?:\s+by\s+(?P<deadline>[^.,]+)|[.,]|$)',
            re.IGNORECASE
        )
        
        # Pattern 3: "Name will handle X"
        pattern3 = re.compile(
            r'(?P<assignee>\w+)\s+will\s+(?:handle|work\s+on|take\s+care\s+of|do|complete)\s+'
            r'(?:the\s+)?(?P<task>[^.,]+)',
            re.IGNORECASE
        )
        
        # Pattern 4: "Name, we need you to..." or "Name, you should..." or "Name, can you..."
        pattern4 = re.compile(
            r'(?P<assignee>\w+),?\s+(?:we\s+need\s+(?:you\s+|someone\s+)?to|you\s+should|can\s+you|could\s+you)\s+'
            r'(?P<task>[^.?!]+)',
            re.IGNORECASE
        )
        
        # Pattern 5: "Name, there is/are X that needs..." (e.g., "Sakshi, there is a login bug")
        pattern5 = re.compile(
            r'(?P<assignee>\w+),?\s+there\s+(?:is|are)\s+(?:a\s+)?(?P<task>[^.]+?)(?:\s+that\s+needs|\s+needing|\s+requiring)',
            re.IGNORECASE
        )
        
        # Pattern 6: "Name, make sure to..." or "Name, ensure..."
        pattern6 = re.compile(
            r'(?P<assignee>\w+),?\s+(?:make\s+sure\s+to|ensure\s+(?:that\s+)?|please\s+ensure)\s+'
            r'(?P<task>[^.?!]+)',
            re.IGNORECASE
        )
        
        # Pattern 7: "Name, you're good with X right? We should..."
        pattern7 = re.compile(
            r'(?P<assignee>\w+),?\s+you\'?(?:re|r)\s+good\s+with\s+(?P<context>[^?]+)\??\s*'
            r'(?:We\s+should\s+(?P<task>[^.]+))?',
            re.IGNORECASE
        )
        
        # Pattern 8: "X should be done by Name" or "X needs to be handled by Name" (assignee at end)
        pattern8 = re.compile(
            r'(?P<task>[^.]+?)\s+(?:should\s+be\s+done|needs\s+to\s+be\s+(?:done|handled|completed|fixed)|will\s+be\s+handled|is\s+assigned)\s+(?:by|to)\s+(?P<assignee>\w+)',
            re.IGNORECASE
        )
        
        # Pattern 9: "there is/got a X" or "we have a X" - generic task detection (infer assignee)
        pattern9 = re.compile(
            r'(?:there\s+is|there\'s|got|have|has)\s+(?:a\s+)?(?P<task>(?:bug|issue|problem|error|task|feature|request)[^.]*?)(?:\s+(?:and\s+)?(?:it\s+)?(?:needs|that\s+needs|requiring|should))?',
            re.IGNORECASE
        )
        
        # Pattern 10: "X needs to be resolved/fixed/done" (task first, infer assignee)
        pattern10 = re.compile(
            r'(?P<task>[^.]*?(?:bug|issue|problem|error|task|feature)[^.]*?)\s+(?:needs\s+to\s+be|should\s+be|must\s+be)\s+(?:resolved|fixed|done|completed|handled)',
            re.IGNORECASE
        )
        
        # Extract priority separately
        priority_pattern = re.compile(r'(?P<priority>high|highest|medium|low|critical)\s+priority', re.IGNORECASE)
        priority_match = priority_pattern.search(text)
        default_priority = priority_match.group('priority') if priority_match else None
        
        # Also check for "this is critical/high priority"
        priority_pattern2 = re.compile(r'this\s+is\s+(?P<priority>high|highest|medium|low|critical)\s+priority', re.IGNORECASE)
        priority_match2 = priority_pattern2.search(text)
        if priority_match2:
            default_priority = priority_match2.group('priority')
        
        # Check for "highest priority" mention
        if not default_priority and re.search(r'highest\s+priority', text_lower):
            default_priority = 'critical'
        
        # Check for "it's blocking" or "critical" mentions
        if not default_priority:
            if re.search(r'blocking|urgent|asap|immediately|immediate\s+attention', text_lower):
                default_priority = 'critical'
            elif re.search(r'important|soon', text_lower):
                default_priority = 'high'
        
        valid_members = [m.lower() for m in self.team_members.keys()]
        seen_descriptions = set()  # Track to avoid duplicate tasks
        
        def add_task(description: str, assignee: str, deadline: str = None, priority: str = None, confidence: float = 0.9, match_obj=None):
            """Helper to add task if not duplicate. Auto-extracts deadline if not provided."""
            desc_lower = description.lower().strip()
            desc_key = desc_lower[:50]  # Use first 50 chars for dedup
            
            # Check if this is a duplicate or subset of existing task
            is_duplicate = False
            for seen in seen_descriptions:
                # If new description is contained in existing, or vice versa, it's a duplicate
                if desc_lower[:30] in seen or seen[:30] in desc_lower:
                    is_duplicate = True
                    break
            
            if not is_duplicate and assignee.lower() in valid_members:
                seen_descriptions.add(desc_key)
                
                # Try to extract deadline from description if not provided
                final_deadline = deadline
                if not final_deadline:
                    final_deadline = self._extract_deadline(description)
                
                # Try to extract deadline from context around the match
                if not final_deadline and match_obj:
                    final_deadline = self._extract_deadline_from_context(text, match_obj.start(), match_obj.end())
                
                # Infer priority from task description if not set
                final_priority = priority
                if not final_priority:
                    desc_lower = description.lower()
                    if any(kw in desc_lower for kw in ['urgent', 'asap', 'critical', 'blocker', 'blocking', 'immediately', 'immediate']):
                        final_priority = 'critical'
                    elif any(kw in desc_lower for kw in ['important', 'soon', 'priority']):
                        final_priority = 'high'
                
                # Normalize priority
                priority_to_use = final_priority if final_priority else default_priority
                if priority_to_use:
                    priority_to_use = priority_to_use.lower()
                    # Map 'highest' to 'critical'
                    if priority_to_use == 'highest':
                        priority_to_use = 'critical'
                    priority_to_use = priority_to_use.title()
                
                tasks.append(ExtractedTask(
                    id=len(tasks) + 1,
                    description=description,
                    assigned_to=assignee.title(),
                    deadline=final_deadline.strip().title() if final_deadline else None,
                    priority=priority_to_use,
                    confidence=confidence
                ))
        
        # Pattern 1: Structured format "Name, please handle the task: Description"
        for match in pattern1.finditer(text):
            add_task(
                description=match.group('task').strip(),
                assignee=match.group('assignee'),
                deadline=match.group('deadline'),
                priority=match.group('priority'),
                confidence=0.95,
                match_obj=match
            )
        
        # Pattern 4: Conversational "Name, we need you to/can you/you should..."
        for match in pattern4.finditer(text):
            add_task(
                description=match.group('task').strip(),
                assignee=match.group('assignee'),
                confidence=0.90,
                match_obj=match
            )
        
        # Pattern 5: "Name, there is X that needs..." (implicit assignment)
        for match in pattern5.finditer(text):
            assignee = match.group('assignee')
            task_desc = match.group('task').strip()
            # This implies the named person should handle it
            add_task(
                description=task_desc,
                assignee=assignee,
                confidence=0.85,
                match_obj=match
            )
        
        # Pattern 6: "Name, make sure to..." or "Name, ensure..."
        for match in pattern6.finditer(text):
            add_task(
                description=match.group('task').strip(),
                assignee=match.group('assignee'),
                confidence=0.90,
                match_obj=match
            )
        
        # Pattern 7: Indirect "Name, you're good with X right?" 
        for match in pattern7.finditer(text):
            assignee = match.group('assignee')
            context_hint = match.group('context')
            explicit_task = match.group('task') if match.lastgroup == 'task' and match.group('task') else None
            
            # Look for actual task in nearby text (within 150 chars after this match)
            end_pos = match.end()
            nearby_text = text[end_pos:end_pos + 150]
            
            # Look for task-like phrases after the context hint
            task_patterns = [
                r"(?:can you|could you|please|we need to|let's)\s+([^.?!]+)",
                r"(?:so|then)\s+([^.?!]+)",
            ]
            
            task_found = False
            for tp in task_patterns:
                task_match = re.search(tp, nearby_text, re.IGNORECASE)
                if task_match:
                    add_task(
                        description=task_match.group(1).strip(),
                        assignee=assignee,
                        confidence=0.85,
                        match_obj=match
                    )
                    task_found = True
                    break
            
            # If no explicit task found, infer from context hint
            if not task_found and context_hint:
                # Convert context hint into potential task
                hint_lower = context_hint.lower()
                if any(kw in hint_lower for kw in ['optimization', 'performance', 'database', 'api', 'backend', 'frontend', 'testing', 'deploy']):
                    inferred_task = f"look at the {context_hint.strip()}"
                    add_task(
                        description=inferred_task,
                        assignee=assignee,
                        confidence=0.75,
                        match_obj=match
                    )
        
        # Pattern 2: "Name please/will fix/handle X"
        for match in pattern2.finditer(text):
            description = match.group('task').strip()
            description = re.sub(r',?\s*(high|highest|medium|low|critical)\s+priority.*$', '', description, flags=re.IGNORECASE)
            description = description.strip(' ,.')
            
            add_task(
                description=description,
                assignee=match.group('assignee'),
                deadline=match.group('deadline') if match.group('deadline') else None,
                confidence=0.90,
                match_obj=match
            )
        
        # Pattern 3: "Name will handle X"
        for match in pattern3.finditer(text):
            add_task(
                description=match.group('task').strip(),
                assignee=match.group('assignee'),
                confidence=0.80,
                match_obj=match
            )
        
        # Pattern 8: "X should be done by Name" (assignee at end)
        for match in pattern8.finditer(text):
            task_desc = match.group('task').strip()
            # Use clean description helper
            task_desc = self._clean_description(task_desc)
            
            if task_desc and len(task_desc) > 3:
                add_task(
                    description=task_desc,
                    assignee=match.group('assignee'),
                    confidence=0.90,
                    match_obj=match
                )
        
        # Pattern 9 & 10: Generic task patterns - infer assignee from expertise
        for pattern in [pattern9, pattern10]:
            for match in pattern.finditer(text):
                task_desc = match.group('task').strip()
                task_desc = self._clean_description(task_desc)
                
                if task_desc and len(task_desc) > 3:
                    # Try to infer assignee based on expertise
                    inferred_assignee = self._infer_assignee_by_expertise(task_desc)
                    if inferred_assignee:
                        add_task(
                            description=task_desc,
                            assignee=inferred_assignee,
                            confidence=0.75,
                            match_obj=match
                        )
        
        # Unassigned tasks pattern - "we need to X", "someone should X"
        unassigned_pattern = re.compile(
            r'(?:we need to|someone should|we should|let\'s|we have to|we must)\s+(?P<task>[^.?!]+)',
            re.IGNORECASE
        )
        
        for match in unassigned_pattern.finditer(text):
            task_desc = match.group('task').strip()
            task_desc = self._clean_description(task_desc)
            # Try to infer assignee based on expertise
            inferred_assignee = self._infer_assignee_by_expertise(task_desc)
            if inferred_assignee:
                add_task(
                    description=task_desc,
                    assignee=inferred_assignee,
                    confidence=0.70,
                    match_obj=match
                )
        
        return tasks
    
    def _infer_assignee_by_expertise(self, task_description: str) -> Optional[str]:
        """Infer the best assignee based on task description and team expertise using weighted scoring."""
        task_lower = task_description.lower()
        
        # Use weighted scoring from DEFAULT_TEAM_WEIGHTED
        scores = {}
        for member, keywords_weights in self.DEFAULT_TEAM_WEIGHTED.items():
            score = 0
            matched_keywords = []
            
            # Sort keywords by length (descending) to match longer phrases first
            sorted_keywords = sorted(keywords_weights.keys(), key=len, reverse=True)
            
            for keyword in sorted_keywords:
                if keyword in task_lower:
                    weight = keywords_weights[keyword]
                    # Bonus for longer/more specific matches
                    specificity_bonus = len(keyword.split()) - 1
                    score += weight + specificity_bonus
                    matched_keywords.append((keyword, weight))
            
            if score > 0:
                scores[member] = {'score': score, 'matches': matched_keywords}
        
        if not scores:
            return None
        
        # Return the member with highest score
        best_member = max(scores.keys(), key=lambda m: scores[m]['score'])
        
        # Log for debugging (optional)
        # print(f"Task: '{task_description[:50]}...' -> Scores: {[(m, s['score']) for m, s in scores.items()]}")
        
        return best_member.title()
    
    def _extract_deadline(self, text: str) -> Optional[str]:
        """Extract deadline from text using multiple patterns."""
        text_lower = text.lower()
        
        for pattern, template in self.DEADLINE_PATTERNS:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                captured = match.group(1) if match.lastindex and match.lastindex >= 1 else match.group(0)
                captured = captured.strip()
                
                # Format using template
                if '{0}' in template:
                    deadline = template.format(captured.title())
                else:
                    deadline = template
                
                return deadline
        
        return None
    
    def _extract_deadline_from_context(self, text: str, task_start: int, task_end: int) -> Optional[str]:
        """Extract deadline from the context around a task mention."""
        # Look in a window around the task (before and after)
        context_start = max(0, task_start - 50)
        context_end = min(len(text), task_end + 100)
        context = text[context_start:context_end]
        
        return self._extract_deadline(context)
    
    def _find_closest_member(self, name: str) -> Optional[str]:
        """Find closest matching team member name."""
        name_lower = name.lower()
        
        for member in self.team_members:
            if name_lower in member or member in name_lower:
                return member
        
        return None
    
    def assign_based_on_expertise(
        self,
        task_description: str
    ) -> Tuple[str, float]:
        """
        Suggest best assignee based on task description and expertise.
        
        Args:
            task_description: Task description text
        
        Returns:
            Tuple of (suggested_assignee, confidence)
        """
        description_lower = task_description.lower()
        
        scores = {}
        for member, expertise_list in self.team_members.items():
            score = 0
            for expertise in expertise_list:
                if expertise in description_lower:
                    score += 1
            scores[member] = score
        
        if not scores or max(scores.values()) == 0:
            return (list(self.team_members.keys())[0], 0.5)
        
        best_member = max(scores, key=scores.get)
        confidence = min(scores[best_member] / 3.0, 1.0)
        
        return (best_member, confidence)
    
    def save(self, model_path: str, tokenizer_path: str):
        """Save model and tokenizer."""
        torch.save(self.model.state_dict(), model_path)
        self.tokenizer.save(tokenizer_path)
    
    @classmethod
    def load(
        cls,
        model_path: str,
        tokenizer_path: str,
        device: Optional[str] = None
    ) -> 'TaskExtractorService':
        """Load service from saved files."""
        return cls(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            device=device
        )
