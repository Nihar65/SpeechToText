import asyncio, aiohttp, json, os, logging
from typing import Optional, Dict, Any, BinaryIO
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(_name_)

class TranscriptionModel(Enum):
    NOVA_2 = "nova-2"

@dataclass
class TranscriptionConfig:
    model: TranscriptionModel = TranscriptionModel.NOVA_2
    smart_format: bool = True
    def to_query_params(self) -> Dict[str, str]:
        return {"model": self.model.value, "smart_format": str(self.smart_format).lower()}

@dataclass
class TranscriptionResult:
    text: str
    confidence: float
    speakers: list
    duration: float
    
    @classmethod
    def from_response(cls, response: Dict[str, Any]) -> 'TranscriptionResult':
        res = response.get('results', {}).get('channels', [{}])[0].get('alternatives', [{}])[0]
        return cls(text=res.get('transcript', ''), confidence=res.get('confidence', 0.0), speakers=[], duration=0.0)

class DeepgramClient:
    BASE_URL = "https://api.deepgram.com/v1"
    def _init_(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('DEEPGRAM_API_KEY')
        if not self.api_key: raise ValueError("Deepgram API key required")
    
    async def transcribe_file(self, audio: Any, config: Optional[TranscriptionConfig] = None) -> TranscriptionResult:
        async with aiohttp.ClientSession(headers={"Authorization": f"Token {self.api_key}"}) as session:
            async with session.post(f"{self.BASE_URL}/listen", params=config.to_query_params() if config else {}, data=audio) as resp:
                resp.raise_for_status()
                return TranscriptionResult.from_response(await resp.json())

class DeepgramSyncClient:
    def _init_(self, api_key: Optional[str] = None):
        self.async_client = DeepgramClient(api_key)
    def transcribe_file(self, audio: Any, config: Optional[TranscriptionConfig] = None) -> TranscriptionResult:
        return asyncio.run(self.async_client.transcribe_file(audio, config))