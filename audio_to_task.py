import os
import sys
import json
import asyncio
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

# Import Deepgram client
from task_extractor.services.deepgram_client import (
    DeepgramSyncClient, 
    DeepgramClient,
    TranscriptionConfig, 
    TranscriptionModel,
    transcribe_audio
)


# ============================================================================
# TASK EXTRACTION (Rule-based - works without trained model)
# ============================================================================

TEAM_EXPERTISE = {
    'mohit': {
        'backend': 3, 'api': 3, 'database': 3, 'db': 3, 'sql': 3, 'query': 3, 'queries': 3,
        'performance': 3, 'optimization': 3, 'optimize': 3, 'slow': 2, 'cache': 3,
        'server': 2, 'endpoint': 2, 'documentation': 2
    },
    'lata': {
        'frontend': 3, 'ui': 3, 'ux': 3, 'design': 3, 'redesign': 3, 'layout': 2,
        'css': 2, 'component': 2, 'page': 2, 'dashboard': 3, 'profile': 3,
        'responsive': 2, 'mobile': 2, 'user friendly': 2
    },
    'arjun': {
        'test': 3, 'testing': 3, 'tests': 3, 'test suite': 3, 'qa': 3,
        'bug': 2, 'debug': 2, 'review': 2, 'pull request': 2, 'regression': 2
    },
    'sakshi': {
        'deploy': 3, 'deployment': 3, 'devops': 3, 'ci': 3, 'cd': 3, 'pipeline': 3,
        'docker': 3, 'release': 3, 'production': 2, 'login': 3, 'hotfix': 3
    }
}

DEADLINE_PATTERNS = [
    (r'by\s+(end\s+of\s+(?:the\s+)?(?:day|week|month))', 'By {0}'),
    (r'by\s+(tomorrow|today|tonight)', 'By {0}'),
    (r'by\s+(next\s+(?:week|monday|tuesday|wednesday|thursday|friday))', 'By {0}'),
    (r'(before\s+(?:the\s+)?(?:release|demo|meeting|client)(?:\s+\w+)?)', '{0}'),
    (r"(before\s+friday'?s?\s+release)", '{0}'),
    (r'within\s+(\d+\s+(?:hours?|days?|weeks?))', 'Within {0}'),
    (r'(asap|immediately|urgent)', '{0}'),
]

PRIORITY_KEYWORDS = {
    'critical': ['urgent', 'asap', 'critical', 'blocker', 'blocking', 'immediately'],
    'high': ['high priority', 'important', 'priority'],
    'medium': ['medium priority'],
    'low': ['low priority', 'when possible']
}


def extract_deadline(text: str) -> Optional[str]:
    text_lower = text.lower()
    for pattern, template in DEADLINE_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            captured = match.group(1).strip()
            if '{0}' in template:
                # Capitalize first letter of each word, but handle possessives
                formatted = ' '.join(word.capitalize() if not word.endswith("'s") 
                                    else word[:-2].capitalize() + "'s" 
                                    for word in captured.split())
                return template.format(formatted)
            return template.title()
    return None


def extract_priority(text: str) -> Optional[str]:
    text_lower = text.lower()
    for priority, keywords in PRIORITY_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return priority.title()
    return None


def infer_assignee(text: str) -> Optional[str]:
    text_lower = text.lower()
    scores = {}
    for member, keywords in TEAM_EXPERTISE.items():
        score = sum(weight for kw, weight in keywords.items() if kw in text_lower)
        if score > 0:
            scores[member] = score
    if scores:
        return max(scores.keys(), key=lambda m: scores[m]).title()
    return None


def extract_tasks_rule_based(text: str) -> List[Dict[str, Any]]:
    """Extract tasks using rule-based patterns."""
    tasks = []
    seen_descriptions = set()
    valid_members = list(TEAM_EXPERTISE.keys())
    
    # Normalize text - clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    global_deadline = extract_deadline(text)
    global_priority = extract_priority(text)
    
    patterns = [
        r'(?P<assignee>\w+),?\s+(?:please\s+)?(?:can\s+you\s+)?(?P<action>look\s+into|handle|fix|work\s+on|redesign|design|create|implement|update|review|test|run|set\s+up|deploy|optimize|take\s+care\s+of)\s+(?:the\s+)?(?P<task>[^.?!,]+?)(?:\s*[.?!,]|$|\s+(?:by|before|this|it|and|we))',
        r'(?P<assignee>\w+)\s+(?P<action>handle|work\s+on|take|do)\s+(?:the\s+)?(?P<task>[^.?!,]+?)(?:\s*[.?!,]|$)',
        r"(?P<assignee>\w+),?\s+there'?s?\s+(?:a\s+)?(?P<task>[^.?!]+?)(?:\s+that\s+needs|\s*[.?!]|$)",
        r'(?P<assignee>\w+),?\s+(?:you\s+should|I\s+need\s+you\s+to)\s+(?P<task>[^.?!]+?)(?:\s*[.?!]|$)',
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            groups = match.groupdict()
            assignee = groups.get('assignee', '').lower() if groups.get('assignee') else None
            task_desc = groups.get('task', '').strip()
            action = groups.get('action', '').strip() if groups.get('action') else ''
            
            if action and not task_desc.lower().startswith(action.lower()):
                task_desc = f"{action} {task_desc}"
            
            # Clean task description
            task_desc = re.sub(r'\s+', ' ', task_desc).strip(' .,')
            
            # Generate a normalized key for deduplication
            task_key = re.sub(r'[^a-z0-9\s]', '', task_desc.lower())
            task_key = ' '.join(task_key.split()[:6])  # First 6 words for matching
            
            if len(task_desc) < 5 or task_key in seen_descriptions:
                continue
            
            if assignee and assignee not in valid_members:
                assignee = None
            if not assignee:
                assignee = infer_assignee(task_desc)
            
            if assignee:
                seen_descriptions.add(task_key)
                task_context = text[max(0, match.start()-20):min(len(text), match.end()+80)]
                
                task_obj = {
                    'id': len(tasks) + 1,
                    'description': task_desc,
                    'assigned_to': assignee.title()
                }
                
                deadline = extract_deadline(task_context) or global_deadline
                priority = extract_priority(task_context) or global_priority
                
                if deadline:
                    # Clean up deadline formatting
                    deadline = re.sub(r'\s+', ' ', deadline).strip()
                    task_obj['deadline'] = deadline
                if priority:
                    task_obj['priority'] = priority
                    
                tasks.append(task_obj)
    
    # Post-processing: Remove semantic duplicates and filter invalid tasks
    final_tasks = []
    seen_normalized = set()
    
    # Filter out non-actionable phrases
    skip_phrases = ['do that', 'do this', 'do it', 'handle that', 'handle this', 'take care']
    
    for task in tasks:
        desc = task['description'].lower()
        
        # Skip non-actionable phrases
        if any(desc.startswith(skip) for skip in skip_phrases):
            continue
            
        # Normalize: keep only alphanumeric and core words
        words = re.findall(r'[a-z]+', desc)
        key_words = [w for w in words if w not in ('the', 'a', 'an', 'to', 'it', 'that', 'this', 'and', 'or', 'do')]
        normalized = ' '.join(key_words[:5])
        
        # Check if this is too similar to existing tasks
        is_duplicate = normalized in seen_normalized
        
        if not is_duplicate:
            # Check for substring matches in existing tasks
            for seen in seen_normalized:
                if normalized in seen or seen in normalized:
                    is_duplicate = True
                    break
        
        if not is_duplicate and len(key_words) >= 2:
            seen_normalized.add(normalized)
            task['id'] = len(final_tasks) + 1
            final_tasks.append(task)
    
    return final_tasks


# ============================================================================
# TRY TO LOAD TRAINED MODEL (OPTIONAL)
# ============================================================================

def try_load_model(checkpoint_dir: str = 'checkpoints_100m', device: str = 'cuda'):
    """Try to load trained model, return None if not available."""
    try:
        from test_model import load_model, hybrid_extract
        model, tokenizer = load_model(checkpoint_dir=checkpoint_dir, device=device)
        return model, tokenizer, hybrid_extract
    except Exception as e:
        print(f"Note: Trained model not available ({e}), using rule-based extraction only.")
        return None, None, None


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def process_audio_file(
    audio_path: str,
    api_key: str = None,
    checkpoint_dir: str = 'checkpoints_100m',
    device: str = 'cuda'
) -> dict:
    """
    Complete pipeline: Audio file -> Transcription -> Task Extraction
    
    Args:
        audio_path: Path to audio file (wav, mp3, m4a, etc.)
        api_key: Deepgram API key (or set DEEPGRAM_API_KEY env var)
        checkpoint_dir: Path to model checkpoints
        device: 'cuda' or 'cpu'
    
    Returns:
        Dict with transcript and extracted tasks
    """
    print(f"Processing: {audio_path}")
    
    # Step 1: Transcribe audio with Deepgram
    print("\n1. Transcribing audio with Deepgram...")
    
    config = TranscriptionConfig(
        model=TranscriptionModel.NOVA_2,
        punctuate=True,
        diarize=True,  # Enable speaker detection
        smart_format=True,
        utterances=True,
    )
    
    result = transcribe_audio(audio_path, api_key=api_key, config=config)
    
    print(f"   Duration: {result.duration:.1f}s")
    print(f"   Confidence: {result.confidence:.2%}")
    print(f"   Speakers detected: {len(result.speakers)}")
    
    # Get formatted transcript with speaker labels
    transcript = result.get_formatted_transcript()
    print(f"\n2. Transcript:\n{'-'*50}")
    print(transcript[:500] + "..." if len(transcript) > 500 else transcript)
    print(f"{'-'*50}")
    
    # Step 2: Extract tasks from transcript
    print("\n3. Extracting tasks...")
    
    # Try to load trained model, fall back to rule-based
    model, tokenizer, hybrid_extract_fn = try_load_model(checkpoint_dir=checkpoint_dir, device=device)
    
    if model is not None:
        tasks = hybrid_extract_fn(model, tokenizer, transcript, device=device)
    else:
        tasks = extract_tasks_rule_based(transcript)
    
    print(f"   Found {len(tasks)} tasks")
    
    # Step 3: Return results
    return {
        'audio_file': audio_path,
        'duration_seconds': result.duration,
        'transcript': transcript,
        'raw_text': result.text,
        'speakers': result.speakers,
        'tasks': tasks
    }


async def process_audio_url(
    audio_url: str,
    api_key: str = None,
    checkpoint_dir: str = 'checkpoints_100m',
    device: str = 'cuda'
) -> dict:
    """
    Process audio from URL (e.g., Zoom recording URL)
    
    Args:
        audio_url: URL of the audio file
        api_key: Deepgram API key
        checkpoint_dir: Path to model checkpoints
        device: 'cuda' or 'cpu'
    
    Returns:
        Dict with transcript and extracted tasks
    """
    print(f"Processing URL: {audio_url}")
    
    config = TranscriptionConfig(
        model=TranscriptionModel.NOVA_2,
        punctuate=True,
        diarize=True,
        smart_format=True,
        utterances=True,
    )
    
    # Use async client for URL
    async with DeepgramClient(api_key=api_key) as client:
        result = await client.transcribe_url(audio_url, config)
    
    transcript = result.get_formatted_transcript()
    
    # Try to load trained model, fall back to rule-based
    model, tokenizer, hybrid_extract_fn = try_load_model(checkpoint_dir=checkpoint_dir, device=device)
    
    if model is not None:
        tasks = hybrid_extract_fn(model, tokenizer, transcript, device=device)
    else:
        tasks = extract_tasks_rule_based(transcript)
    
    return {
        'audio_url': audio_url,
        'duration_seconds': result.duration,
        'transcript': transcript,
        'tasks': tasks
    }


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Audio to Task Extraction Pipeline')
    parser.add_argument('audio', nargs='?', help='Path to audio file or URL')
    parser.add_argument('--api-key', help='Deepgram API key (or set DEEPGRAM_API_KEY env var)')
    parser.add_argument('--checkpoint-dir', default='checkpoints_100m', help='Model checkpoint directory')
    parser.add_argument('--device', default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--output', '-o', help='Output JSON file')
    parser.add_argument('--test', action='store_true', help='Run test with sample transcript')
    
    args = parser.parse_args()
    
    # Test mode
    if args.test or not args.audio:
        test_with_sample_text()
        return
    
    # Check if API key is available
    api_key = args.api_key or os.environ.get('DEEPGRAM_API_KEY')
    if not api_key:
        print("Error: Deepgram API key required!")
        print("Set DEEPGRAM_API_KEY environment variable or use --api-key")
        print("\nGet your API key from: https://console.deepgram.com/")
        return
    
    try:
        # Process audio
        if args.audio.startswith(('http://', 'https://')):
            result = asyncio.run(process_audio_url(
                args.audio,
                api_key=api_key,
                checkpoint_dir=args.checkpoint_dir,
                device=args.device
            ))
        else:
            result = process_audio_file(
                args.audio,
                api_key=api_key,
                checkpoint_dir=args.checkpoint_dir,
                device=args.device
            )
        
        # Print results
        print("\n" + "=" * 70)
        print("EXTRACTED TASKS")
        print("=" * 70)
        print(json.dumps(result['tasks'], indent=2))
        
        # Save to file if specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to: {args.output}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


# Quick test function (without audio file)
def test_with_sample_text():
    """Test task extraction with sample meeting transcript"""
    sample_transcript = """
    Speaker 0: Alright everyone, let's go through the action items from today's standup.
    Speaker 1: Sure. So Mohit, can you please look into the database performance issues? 
    The queries are running really slow and it's affecting the user experience.
    Speaker 0: Yes, that's high priority. We need it done by end of day.
    Speaker 2: I can also help with the frontend optimization. Lata, you should redesign 
    the dashboard UI to make it more responsive.
    Speaker 1: Good idea. And Sakshi, please deploy the hotfix to production before 
    the client demo tomorrow.
    Speaker 3: Got it. I'll also set up the CI/CD pipeline by next week.
    Speaker 0: Perfect. Arjun, can you run the full test suite on all the new features?
    Speaker 2: We should do that before Friday's release.
    """
    
    print("Testing with sample transcript...")
    print(f"\nTranscript:\n{sample_transcript}")
    
    # Check if model is available
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Try to load trained model, fall back to rule-based
    model, tokenizer, hybrid_extract_fn = try_load_model(device=device)
    
    if model is not None:
        tasks = hybrid_extract_fn(model, tokenizer, sample_transcript, device=device)
        print("\n(Using trained model + rules)")
    else:
        tasks = extract_tasks_rule_based(sample_transcript)
        print("\n(Using rule-based extraction - model not found)")
    
    print("\n" + "=" * 70)
    print("EXTRACTED TASKS")
    print("=" * 70)
    print(json.dumps(tasks, indent=2))


if __name__ == '__main__':
    main()