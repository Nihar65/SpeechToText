# Task Extraction from Meeting Audio/Transcripts

A production-ready pipeline for extracting structured task information from meeting audio or transcripts using a custom 70M parameter Transformer model trained from scratch.

## 🎯 Features

- **Audio-to-Tasks Pipeline**: Pass an audio file → Get structured tasks
- **Deepgram STT Integration**: High-quality speech-to-text with speaker diarization
- **Custom Transformer Model**: 70M parameters, trained on 100k samples (98.89% accuracy)
- **Hybrid Extraction**: Rule-based patterns + trained model for maximum accuracy
- **Smart Assignment**: Expertise-based assignee suggestion
- **No Model? No Problem**: Works with rule-based extraction even without the trained model

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/Nihar65/SpeechToText.git
cd TaskDetection

# Create virtual environment
python -m venv task
# Windows:
task\Scripts\activate
# Linux/Mac:
source task/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your Deepgram API key
# Get your free API key at: https://console.deepgram.com/
```

Or set environment variable directly:

```bash

# Already env file given in repo 

# Windows PowerShell:
$env:DEEPGRAM_API_KEY="your_api_key_here"

# Linux/Mac:
export DEEPGRAM_API_KEY="your_api_key_here"
```

### 3. Place Your Audio File

Put your meeting audio file in the project directory (or any path you can reference):

```
TaskDetection/
├── your_meeting.wav      ← Place audio file here
├── audio_to_task.py
├── .env
└── ...
```

**Supported audio formats**: WAV, MP3, M4A, FLAC, OGG, WebM

### 4. Run the Pipeline

```bash
# Basic usage with audio file
python audio_to_task.py your_meeting.wav --api-key YOUR_DEEPGRAM_KEY

# Or if you set DEEPGRAM_API_KEY in .env or environment:
python audio_to_task.py your_meeting.wav

# Process audio from URL (e.g., Zoom recording link)
python audio_to_task.py https://example.com/meeting.mp3

# Save results to JSON file
python audio_to_task.py your_meeting.wav --output results.json

# Use CPU instead of GPU
python audio_to_task.py your_meeting.wav --device cpu

# Test without audio file (uses sample transcript)
python audio_to_task.py
```

## 📋 Usage Examples

### Example 1: Process Local Audio File

```bash
python audio_to_task.py meeting_recording.wav --api-key dg_xxxxxxxxxxxx
```

**Output:**

```
Processing: meeting_recording.wav

1. Transcribing audio with Deepgram...
   Duration: 180.5s
   Confidence: 95.2%
   Speakers detected: 4

2. Transcript:
--------------------------------------------------
Speaker 0: Let's discuss the tasks for this week.
Speaker 1: Mohit, can you fix the login bug by tomorrow?
...
--------------------------------------------------

3. Extracting tasks...
   Found 5 tasks

======================================================================
EXTRACTED TASKS
======================================================================
[
  {
    "id": 1,
    "description": "fix the login bug",
    "assigned_to": "Mohit",
    "deadline": "By Tomorrow",
    "priority": "High"
  },
  ...
]
```

### Example 2: Save Results to JSON

```bash
python audio_to_task.py standup.mp3 --output tasks.json
```

This creates `tasks.json` with:

```json
{
  "audio_file": "standup.mp3",
  "duration_seconds": 300.5,
  "transcript": "Speaker 0: ...",
  "tasks": [
    {
      "id": 1,
      "description": "implement user authentication",
      "assigned_to": "Mohit",
      "deadline": "By End Of Week",
      "priority": "High"
    }
  ]
}
```

### Example 3: Test Without Audio (Sample Transcript)

```bash
python audio_to_task.py
```

This runs a test with a built-in sample meeting transcript.

## 📁 Project Structure

```
SpeechT0Text/
├── audio_to_task.py         # 🎯 MAIN PIPELINE - Run this!
├── test_model.py             # Test trained model
├── train_colab.py            # Training script for Kaggle/Colab
├── .env.example              # Environment variables template
├── requirements.txt          # Python dependencies
│
├── checkpoints_100m/         # Trained model (download from Kaggle)
│   ├── best_model.pt         # Model weights (70M params)
│   └── tokenizer.json        # Tokenizer vocabulary
│
├── task_extractor/           # Core library
│   ├── model/               # Transformer architecture
│   ├── data/                 # Dataset & tokenizer
│   ├── service/             # Deepgram client, extraction
│   └── api/                  # FastAPI server
│
└── examples/                 # Example scripts
```

## 🔧 Configuration

### Team Members & Expertise

Edit the `TEAM_EXPERTISE` in `audio_to_task.py` to customize:

```python
TEAM_EXPERTISE = {
    'mohit': {'backend': 3, 'api': 3, 'database': 3, 'server': 2},
    'lata': {'frontend': 3, 'ui': 3, 'design': 2, 'css': 2},
    'arjun': {'testing': 3, 'test': 3, 'qa': 3, 'automation': 2},
    'sakshi': {'devops': 3, 'deploy': 3, 'ci/cd': 3, 'infrastructure': 2},
}
```

### Model Checkpoint

If you trained the model on Kaggle:

1. Download `best_model.pt` and `tokenizer.json` from Kaggle output
2. Place them in `checkpoints_100m/` directory

Without the model, the pipeline uses rule-based extraction (still works well!).

## 🎙️ Getting a Deepgram API Key

1. Go to [https://deepgram.com](https://deepgram.com)
2. Sign up for a free account (includes $200 credits)
3. Navigate to Dashboard → API Keys
4. Create a new API key
5. Copy the key to your `.env` file

## 🏃 Running the Pipeline

### Command Line Options

```bash
python audio_to_task.py [AUDIO_PATH_OR_URL] [OPTIONS]

Arguments:
  AUDIO_PATH_OR_URL     Path to audio file or URL

Options:
  --api-key KEY         Deepgram API key (or set DEEPGRAM_API_KEY env var)
  --checkpoint-dir DIR  Model checkpoint directory (default: checkpoints_100m)
  --device DEVICE       cuda or cpu (default: cuda)
  --output, -o FILE     Save results to JSON file
```

### Examples

```bash
# Local WAV file
python audio_to_task.py meeting.wav

# Local MP3 file with explicit API key
python audio_to_task.py standup.mp3 --api-key dg_xxxxx

# Remote audio URL
python audio_to_task.py https://storage.example.com/meeting.wav

# Save to file
python audio_to_task.py meeting.wav -o extracted_tasks.json

# Use CPU
python audio_to_task.py meeting.wav --device cpu

# Test mode (no audio needed)
python audio_to_task.py
```

## 🧪 Testing

### Test with Sample Transcript (No API Key Needed)

```bash
python audio_to_task.py
```

### Test Trained Model

```bash
python test_model.py
```

### Test Extraction Service

```bash
python test_extraction.py
```

## 📊 Model Details

| Specification       | Value                    |
| ------------------- | ------------------------ |
| Parameters          | 70.3M                    |
| Architecture        | Encoder-only Transformer |
| d_model             | 1024                     |
| Layers              | 8                        |
| Attention Heads     | 16                       |
| Feed-Forward Dim    | 2048                     |
| Training Data       | 100,000 samples          |
| Validation Accuracy | 98.89%                   |
| Training Time       | ~2 hours (2x T4 GPU)     |

## 🔄 Pipeline Flow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│ Audio File  │ ──► │   Deepgram   │ ──► │ Task Extraction │ ──► │ Structured   │
│ (.wav/.mp3) │     │   STT API    │     │ (Rules + Model) │     │    Tasks     │
└─────────────┘     └──────────────┘     └─────────────────┘     └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  Transcript  │
                    │  + Speakers  │
                    └──────────────┘
```

## 📝 Output Format

```json
{
  "audio_file": "meeting.wav",
  "duration_seconds": 180.5,
  "transcript": "Speaker 0: ...\nSpeaker 1: ...",
  "speakers": ["Speaker 0", "Speaker 1"],
  "tasks": [
    {
      "id": 1,
      "description": "fix the login bug",
      "assigned_to": "Mohit",
      "deadline": "By Tomorrow",
      "priority": "High"
    },
    {
      "id": 2,
      "description": "update the dashboard UI",
      "assigned_to": "Lata",
      "deadline": "By End Of Week",
      "priority": "Medium"
    }
  ]
}
```

## 🐳 Docker (Optional)

```bash
# Build and run
docker-compose up -d task-extractor

# With API server
docker-compose up -d
```

## 📄 License

MIT License

## 🙏 Acknowledgments

- [Deepgram](https://deepgram.com) for speech-to-text API
- [PyTorch](https://pytorch.org) for deep learning framework
- [Kaggle](https://kaggle.com) for GPU compute
