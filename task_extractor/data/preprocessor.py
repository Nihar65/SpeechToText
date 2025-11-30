import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class TeamMember:
    name: str
    expertise: List[str]
    aliases: List[str] = None
    
    def _post_init_(self):
        if self.aliases is None: self.aliases = []
        self.aliases.append(self.name.lower())

class MeetingPreprocessor:
    FILLER_WORDS = ['um', 'uh', 'er', 'ah', 'like', 'you know', 'i mean', 'basically', 'actually', 'literally', 'honestly', 'so yeah', 'right', 'okay so']
    PRIORITY_SYNONYMS = {'urgent': 'critical', 'important': 'high', 'normal': 'medium', 'asap': 'critical', 'whenever': 'low'}
    DEADLINE_PATTERNS = [
        (r'\bby\s+end\s+of\s+(day|today)\b', 'today'),
        (r'\bby\s+tomorrow\s+morning\b', 'tomorrow'),
        (r'\bby\s+eod\b', 'today'),
        (r'\bby\s+eow\b', 'end of this week'),
        (r'\bnext\s+week\b', 'next monday'),
    ]
    
    def _init_(self, team_members: Optional[List[TeamMember]] = None, remove_fillers: bool = True, normalize_numbers: bool = True):
        self.team_members = team_members or []
        self.remove_fillers = remove_fillers
        self.normalize_numbers = normalize_numbers
        self.name_lookup = {alias.lower(): member.name for member in self.team_members for alias in member.aliases}
        self.filler_pattern = re.compile(r'\b(' + '|'.join(re.escape(f) for f in self.FILLER_WORDS) + r')\b', re.IGNORECASE)
        self.speaker_pattern = re.compile(r'^[\w\s]+:\s*', re.MULTILINE)
        self.multiple_spaces = re.compile(r'\s+')
        self.sentence_end = re.compile(r'([.!?])\s+')
    
    def clean_text(self, text: str) -> str:
        text = self.speaker_pattern.sub('', text)
        if self.remove_fillers: text = self.filler_pattern.sub('', text)
        text = self.multiple_spaces.sub(' ', text)
        for synonym, standard in self.PRIORITY_SYNONYMS.items():
            text = re.sub(rf'\b{synonym}\b', standard, text, flags=re.IGNORECASE)
        for pattern, replacement in self.DEADLINE_PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        words = text.split()
        normalized = [self.name_lookup.get(w.lower().strip('.,!?'), w) for w in words]
        return ' '.join(normalized).strip()
    
    def segment_sentences(self, text: str) -> List[str]:
        sentences = self.sentence_end.split(text)
        result = []
        for i in range(0, len(sentences) - 1, 2):
            result.append(sentences[i].strip() + (sentences[i+1] if i+1 < len(sentences) else ''))
        if len(sentences) % 2 == 1: result.append(sentences[-1].strip())
        return [s for s in result if s.strip()]
    
    def extract_task_candidates(self, text: str) -> List[Dict]:
        candidates = []
        task_pattern = re.compile(r'(?P<assignee>\w+),?\s+please\s+handle\s+(?:the\s+)?(?:task:?\s*)?(?P<task>[^.]+)\.(?:\s+It\'s\s+(?P<priority>\w+)\s+priority)?(?:\s+(?:and\s+)?should\s+be\s+done\s+by\s+(?P<deadline>[^.]+))?', re.IGNORECASE)
        for match in task_pattern.finditer(text):
            candidates.append({
                'assignee': match.group('assignee'),
                'description': match.group('task').strip(),
                'priority': match.group('priority'),
                'deadline': match.group('deadline'),
                'span': (match.start(), match.end())
            })
        return candidates
    
    def preprocess_for_model(self, text: str, max_length: Optional[int] = None) -> Tuple[str, Dict]:
        cleaned = self.clean_text(text)
        candidates = self.extract_task_candidates(cleaned)
        if max_length and len(cleaned) > max_length: cleaned = cleaned[:max_length]
        return cleaned, {'original_length': len(text), 'cleaned_length': len(cleaned), 'num_candidates': len(candidates), 'candidates': candidates}
    
    @classmethod
    def create_default(cls) -> 'MeetingPreprocessor':
        return cls(team_members=[
            TeamMember('Mohit', ['backend', 'api'], ['mohit', 'moheet']),
            TeamMember('Lata', ['frontend', 'ui'], ['lata', 'lataa']),
            TeamMember('Arjun', ['testing', 'qa'], ['arjun', 'arjoon']),
            TeamMember('Sakshi', ['devops', 'deployment'], ['sakshi', 'sakshee']),
        ])