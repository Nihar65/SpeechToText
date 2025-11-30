import json, random
from typing import List, Dict

TEAM_MEMBERS = {'Mohit': 'Backend', 'Lata': 'Frontend', 'Arjun': 'QA', 'Sakshi': 'DevOps'}
TASK_TEMPLATES = {
    'backend': ["optimize the {thing} API", "fix slow {thing} query", "implement caching for {thing}"],
    'frontend': ["redesign {thing} page", "fix {thing} styling", "make {thing} responsive"],
    'qa': ["write tests for {thing}", "verify {thing} fix", "run regression on {thing}"],
    'devops': ["deploy {thing}", "fix CI pipeline for {thing}", "monitor {thing} logs"]
}
THINGS = ["user", "order", "payment", "dashboard", "search", "login"]
DEADLINES = ["by tomorrow", "by Friday", "ASAP", "next week"]
PRIORITIES = ["high", "medium", "low", "critical"]

def generate_sample() -> Dict:
    member = random.choice(list(TEAM_MEMBERS.keys()))
    role = TEAM_MEMBERS[member].lower()
    task_type = 'backend' if 'backend' in role else 'frontend' if 'frontend' in role else 'qa' if 'qa' in role else 'devops'
    template = random.choice(TASK_TEMPLATES.get(task_type, TASK_TEMPLATES['backend']))
    task_desc = template.format(thing=random.choice(THINGS))
    deadline = random.choice(DEADLINES)
    priority = random.choice(PRIORITIES)
    
    text = f"{member}, please {task_desc}. It is {priority} priority and due {deadline}."
    return {
        'text': text,
        'tasks': [{'description': task_desc, 'assigned_to': member, 'deadline': deadline, 'priority': priority.title()}]
    }

if _name_ == '_main_':
    with open('training_data_100k.jsonl', 'w') as f:
        for _ in range(100): # Sample size
            f.write(json.dumps(generate_sample()) + '\n')