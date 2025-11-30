from typing import Union, List, Dict
from pathlib import Path
from .deepgram_client import DeepgramSyncClient
from .task_extractor import TaskExtractorService

class InferencePipeline:
    def _init_(self, task_extractor: TaskExtractorService, deepgram_api_key: str = None):
        self.extractor = task_extractor
        self.stt = DeepgramSyncClient(deepgram_api_key) if deepgram_api_key else None
        
    def process_audio(self, audio_path: Union[str, Path]) -> Dict:
        if not self.stt: raise ValueError("STT not configured")
        with open(audio_path, 'rb') as f:
            transcript = self.stt.transcribe_file(f).text
        return self.extractor.extract_tasks(transcript)
        
    def process_text(self, text: str) -> Dict:
        return self.extractor.extract_tasks(text)