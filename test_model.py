import torch
import json
import re
from typing import Dict, List, Optional, Any
from task_extractor.model import TaskExtractionTransformer
from task_extractor.data import TaskTokenizer


# Team expertise for assignee inference
TEAM_EXPERTISE = {
    'mohit': {
        'backend': 3, 'api': 3, 'database': 3, 'db': 3, 'sql': 3, 'query': 3, 'queries': 3,
        'performance': 3, 'optimization': 3, 'optimize': 3, 'slow': 2, 'cache': 3,
        'server': 2, 'endpoint': 2, 'rest': 2, 'graphql': 2, 'documentation': 2
    },
    'lata': {
        'frontend': 3, 'ui': 3, 'ux': 3, 'design': 3, 'redesign': 3, 'layout': 2,
        'css': 2, 'component': 2, 'page': 2, 'dashboard': 3, 'profile': 3,
        'user interface': 3, 'responsive': 2, 'mobile': 2, 'user friendly': 2
    },
    'arjun': {
        'test': 3, 'testing': 3, 'tests': 3, 'test suite': 3, 'qa': 3,
        'bug': 2, 'debug': 2, 'review': 2, 'pull request': 2, 'code review': 2,
        'automation': 2, 'regression': 2, 'coverage': 2
    },
    'sakshi': {
        'deploy': 3, 'deployment': 3, 'devops': 3, 'ci': 3, 'cd': 3, 'pipeline': 3,
        'docker': 3, 'kubernetes': 2, 'release': 3, 'production': 2,
        'login': 3, 'security': 2, 'hotfix': 3, 'infrastructure': 2
    }
}

# Deadline patterns
DEADLINE_PATTERNS = [
    (r'by\s+(end\s+of\s+(?:the\s+)?(?:day|week|month|sprint))', 'By {0}'),
    (r'by\s+(tomorrow|today|tonight)', 'By {0}'),
    (r'by\s+(next\s+(?:week|month|monday|tuesday|wednesday|thursday|friday))', 'By {0}'),
    (r'by\s+(this\s+(?:week|friday|weekend))', 'By {0}'),
    (r'by\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm))', 'By {0}'),
    (r'(before\s+(?:the\s+)?(?:release|deploy|demo|meeting|client|sprint)(?:\s+\w+)?(?:\s+\w+)?)', '{0}'),
    (r"(before\s+friday'?s?\s+release)", '{0}'),
    (r'within\s+(\d+\s+(?:hours?|days?|weeks?))', 'Within {0}'),
    (r'(asap|immediately|urgent)', '{0}'),
    (r'(end\s+of\s+(?:the\s+)?day)', 'By {0}'),
]

# Priority patterns
PRIORITY_KEYWORDS = {
    'critical': ['urgent', 'asap', 'critical', 'blocker', 'blocking', 'immediately', 'immediate attention'],
    'high': ['high priority', 'important', 'priority', 'soon'],
    'medium': ['medium priority', 'normal'],
    'low': ['low priority', 'when possible', 'nice to have']
}


def load_model(checkpoint_dir='checkpoints_100m', device='cuda'):
    """Load the trained model and tokenizer."""
    tokenizer = TaskTokenizer.load(f'{checkpoint_dir}/tokenizer.json')
    
    model = TaskExtractionTransformer(
        vocab_size=tokenizer.vocab_size_actual,
        d_model=1024,
        num_heads=16,
        num_layers=8,
        d_ff=2048,
        max_seq_len=512,
        num_assignees=5,
        num_priorities=4,
        num_bio_tags=9,
        dropout=0.1,
        use_crf=True
    )
    
    model.load_state_dict(torch.load(f'{checkpoint_dir}/best_model.pt', map_location=device))
    model.to(device).eval()
    
    return model, tokenizer


def extract_deadline(text: str) -> Optional[str]:
    """Extract deadline from text."""
    text_lower = text.lower()
    for pattern, template in DEADLINE_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            captured = match.group(1).strip()
            if '{0}' in template:
                return template.format(captured.title())
            return template.title()
    return None


def extract_priority(text: str) -> Optional[str]:
    """Extract priority from text."""
    text_lower = text.lower()
    for priority, keywords in PRIORITY_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return priority.title()
    return None


def infer_assignee(text: str) -> Optional[str]:
    """Infer best assignee based on task content."""
    text_lower = text.lower()
    scores = {}
    
    for member, keywords in TEAM_EXPERTISE.items():
        score = 0
        for keyword, weight in keywords.items():
            if keyword in text_lower:
                score += weight
        if score > 0:
            scores[member] = score
    
    if scores:
        return max(scores.keys(), key=lambda m: scores[m]).title()
    return None


def model_predict(model, tokenizer, text: str, device='cuda') -> Dict:
    """Get model predictions."""
    BIO_TAGS = ['O', 'B-TASK', 'I-TASK', 'B-ASSIGNEE', 'I-ASSIGNEE', 
                'B-DEADLINE', 'I-DEADLINE', 'B-PRIORITY', 'I-PRIORITY']
    
    encoded = tokenizer.encode(text, max_length=512, padding=True)
    input_ids = torch.tensor([encoded['input_ids']]).to(device)
    attention_mask = torch.tensor([encoded['attention_mask']]).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    bio_preds = outputs['bio_logits'].argmax(dim=-1)[0].cpu().tolist()
    tokens = tokenizer.tokenize(text)
    
    # Extract entities from model
    entities = {'tasks': [], 'assignees': [], 'deadlines': [], 'priorities': []}
    current_type = None
    current_tokens = []
    
    for i, (token, tag_id) in enumerate(zip(tokens, bio_preds[1:len(tokens)+1])):
        tag = BIO_TAGS[tag_id]
        
        if tag.startswith('B-'):
            if current_type and current_tokens:
                key = current_type.lower() + 's'
                if key == 'prioritys': key = 'priorities'
                entities[key].append(' '.join(current_tokens))
            current_type = tag[2:]
            current_tokens = [token]
        elif tag.startswith('I-') and current_type == tag[2:]:
            current_tokens.append(token)
        else:
            if current_type and current_tokens:
                key = current_type.lower() + 's'
                if key == 'prioritys': key = 'priorities'
                entities[key].append(' '.join(current_tokens))
            current_type = None
            current_tokens = []
    
    if current_type and current_tokens:
        key = current_type.lower() + 's'
        if key == 'prioritys': key = 'priorities'
        entities[key].append(' '.join(current_tokens))
    
    return entities


def rule_based_extract(text: str) -> List[Dict]:
    """Rule-based task extraction with comprehensive patterns."""
    tasks = []
    seen = set()
    valid_members = list(TEAM_EXPERTISE.keys())
    
    # Get global deadline and priority
    global_deadline = extract_deadline(text)
    global_priority = extract_priority(text)
    
    patterns = [
        # Pattern: "Name, please/can you/I need you to [task]"
        r'(?P<assignee>\w+),?\s+(?:please\s+)?(?:can\s+you\s+)?(?:I\s+need\s+you\s+to\s+)?(?P<action>look\s+into|handle|fix|work\s+on|redesign|design|create|implement|update|review|test|run|set\s+up|deploy|optimize|check|take\s+care\s+of)\s+(?:the\s+)?(?P<task>[^.?!,]+?)(?:\s*[.?!,]|$|\s+(?:by|before|this|it|and|we))',
        
        # Pattern: "Name handle/work on [task]" (no comma)
        r'(?P<assignee>\w+)\s+(?P<action>handle|work\s+on|take|do)\s+(?:the\s+)?(?P<task>[^.?!,]+?)(?:\s*[.?!,]|$)',
        
        # Pattern: "Name, there's a [issue]" (implicit assignment)
        r"(?P<assignee>\w+),?\s+there'?s?\s+(?:a\s+)?(?P<task>[^.?!]+?)(?:\s+that\s+needs|\s*[.?!]|$)",
        
        # Pattern: "Name, this needs..." 
        r'(?P<assignee>\w+),?\s+this\s+needs\s+(?P<task>[^.?!]+?)(?:\s*[.?!]|$)',
        
        # Pattern: "we need [task]" -> infer assignee
        r'(?:we\s+need\s+to|we\s+should|let\'s|someone\s+should)\s+(?P<task>[^.?!]+?)(?:\s*[.?!]|$)',
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            groups = match.groupdict()
            
            assignee = groups.get('assignee', '').lower() if groups.get('assignee') else None
            task_desc = groups.get('task', '').strip()
            action = groups.get('action', '').strip() if groups.get('action') else ''
            
            # Combine action with task
            if action and not task_desc.lower().startswith(action.lower()):
                task_desc = f"{action} {task_desc}"
            
            # Clean task description
            task_desc = re.sub(r'\s+', ' ', task_desc).strip()
            task_desc = re.sub(r'^(the|a|an)\s+', '', task_desc, flags=re.IGNORECASE)
            task_desc = re.sub(r'\s+(and|we|it\'s|this|that)$', '', task_desc, flags=re.IGNORECASE)
            
            # Skip if too short or already seen
            task_key = task_desc.lower()[:40]
            if len(task_desc) < 5 or task_key in seen:
                continue
            
            # Validate or infer assignee
            if assignee and assignee not in valid_members:
                assignee = None
            if not assignee:
                assignee = infer_assignee(task_desc)
            
            if assignee:
                seen.add(task_key)
                
                # Extract deadline from task context
                task_context = text[max(0, match.start()-20):min(len(text), match.end()+80)]
                deadline = extract_deadline(task_context) or global_deadline
                priority = extract_priority(task_context) or global_priority
                
                tasks.append({
                    'description': task_desc.strip(' .,'),
                    'assigned_to': assignee.title(),
                    'deadline': deadline,
                    'priority': priority
                })
    
    return tasks


def hybrid_extract(model, tokenizer, text: str, device='cuda') -> List[Dict]:
    """
    Hybrid extraction combining model + rules for best accuracy.
    
    Strategy:
    1. Use rule-based for structured patterns (more reliable)
    2. Use model to validate and enhance extractions
    3. Fall back to model for unstructured text
    """
    # Step 1: Rule-based extraction
    rule_tasks = rule_based_extract(text)
    
    # Step 2: Model extraction
    model_entities = model_predict(model, tokenizer, text, device)
    
    # Step 3: Merge and enhance
    final_tasks = []
    seen_descriptions = set()
    
    # Process rule-based tasks first (more accurate)
    for task in rule_tasks:
        desc_key = task['description'].lower()[:30]
        if desc_key not in seen_descriptions:
            seen_descriptions.add(desc_key)
            
            # Enhance with model entities if missing
            if not task.get('deadline') and model_entities['deadlines']:
                task['deadline'] = model_entities['deadlines'][0].title()
            if not task.get('priority') and model_entities['priorities']:
                task['priority'] = model_entities['priorities'][0].title()
            
            task['id'] = len(final_tasks) + 1
            final_tasks.append(task)
    
    # Add model-detected tasks that rules missed
    for model_task in model_entities['tasks']:
        task_key = model_task.lower()[:30]
        is_new = True
        
        # Check if this overlaps with existing tasks
        for seen in seen_descriptions:
            if task_key in seen or seen in task_key:
                is_new = False
                break
        
        if is_new and len(model_task) > 5:
            seen_descriptions.add(task_key)
            
            # Find assignee from model or infer
            assignee = None
            if model_entities['assignees']:
                assignee = model_entities['assignees'][0].title()
            else:
                assignee = infer_assignee(model_task)
            
            if assignee:
                final_tasks.append({
                    'id': len(final_tasks) + 1,
                    'description': model_task,
                    'assigned_to': assignee,
                    'deadline': model_entities['deadlines'][0].title() if model_entities['deadlines'] else None,
                    'priority': model_entities['priorities'][0].title() if model_entities['priorities'] else None
                })
    
    # Renumber IDs
    for i, task in enumerate(final_tasks):
        task['id'] = i + 1
    
    return final_tasks


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("\nLoading model...")
    model, tokenizer = load_model(device=device)
    print("Model loaded successfully!")
    
    # Test sentences (Zoom meeting transcript style)
    test_texts = [
        "Okay so Mohit can you please look into the database performance issues we discussed? This is high priority and we need it done by end of day.",
        
        "Alright team, let's wrap up. Lata, I need you to redesign the dashboard UI, make it more user friendly. And Sakshi, please deploy the hotfix to production before the client demo tomorrow.",
        
        "Yeah I think that makes sense. Oh by the way Arjun, can you run the full test suite on the new features? We should do that before Friday's release.",
        
        "So here's what we need to do. Mohit handle the API optimization, that's urgent. Lata work on the mobile responsive design. Sakshi set up the CI/CD pipeline by next week. And Arjun please review all the pull requests.",
        
        "I was thinking about this yesterday and I think we should prioritize the login bug fix. Sakshi can you take care of that? It's blocking several customers. High priority please.",
        
        # Additional complex examples
        "Hey team, quick update. We have a database query that's running slow, Mohit you're good with that stuff right? Can you optimize it? Also Lata, the user profile page needs a complete redesign before the release.",
        
        "There's a critical bug in production affecting login. Sakshi, this needs immediate attention. Arjun, once it's fixed, run the full regression test suite.",
    ]
    
    print("\n" + "=" * 70)
    print("TASK EXTRACTION TEST (HYBRID MODEL + RULES)")
    print("=" * 70)
    
    all_results = []
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{'─' * 70}")
        print(f"TEST {i}")
        print(f"{'─' * 70}")
        print(f"\nInput:\n  \"{text}\"")
        
        tasks = hybrid_extract(model, tokenizer, text, device)
        
        # Clean output for display
        display_tasks = []
        for t in tasks:
            clean_task = {'id': t['id'], 'description': t['description'], 'assigned_to': t['assigned_to']}
            if t.get('deadline'):
                clean_task['deadline'] = t['deadline']
            if t.get('priority'):
                clean_task['priority'] = t['priority']
            display_tasks.append(clean_task)
        
        all_results.append({
            'input': text,
            'tasks': display_tasks
        })
        
        print(f"\nExtracted Tasks:")
        print(json.dumps(display_tasks, indent=2))
    
    # Save results
    output_file = 'test_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print(f"Results saved to: {output_file}")
    print("=" * 70)


if __name__ == '_main_':
    main()