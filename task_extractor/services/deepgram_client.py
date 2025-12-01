import asyncio
import aiohttp
import json
import os
from typing import Optional, Dict, Any, BinaryIO, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import logging
import websockets
from pathlib import Path

logger = logging.getLogger(__name__)


class TranscriptionModel(Enum):
    """Available Deepgram transcription models."""
    NOVA_2 = "nova-2"
    NOVA = "nova"
    ENHANCED = "enhanced"
    BASE = "base"


class Language(Enum):
    """Supported languages."""
    ENGLISH = "en"
    ENGLISH_US = "en-US"
    ENGLISH_UK = "en-GB"
    ENGLISH_IN = "en-IN"


@dataclass
class TranscriptionConfig:
    """Configuration for transcription requests."""
    model: TranscriptionModel = TranscriptionModel.NOVA_2
    language: Language = Language.ENGLISH_US
    punctuate: bool = True
    diarize: bool = True
    smart_format: bool = True
    utterances: bool = True
    paragraphs: bool = True
    filler_words: bool = False
    numerals: bool = True
    
    def to_query_params(self) -> Dict[str, str]:
        """Convert config to API query parameters."""
        return {
            "model": self.model.value,
            "language": self.language.value,
            "punctuate": str(self.punctuate).lower(),
            "diarize": str(self.diarize).lower(),
            "smart_format": str(self.smart_format).lower(),
            "utterances": str(self.utterances).lower(),
            "paragraphs": str(self.paragraphs).lower(),
            "filler_words": str(self.filler_words).lower(),
            "numerals": str(self.numerals).lower(),
        }


@dataclass
class TranscriptionResult:
    """Result from transcription request."""
    text: str
    confidence: float
    words: list
    utterances: list
    paragraphs: list
    speakers: list
    duration: float
    raw_response: Dict[str, Any]
    
    @classmethod
    def from_response(cls, response: Dict[str, Any]) -> 'TranscriptionResult':
        """Create TranscriptionResult from API response."""
        results = response.get('results', {})
        channels = results.get('channels', [{}])
        
        if not channels:
            return cls(
                text="",
                confidence=0.0,
                words=[],
                utterances=[],
                paragraphs=[],
                speakers=[],
                duration=0.0,
                raw_response=response
            )
        
        alternatives = channels[0].get('alternatives', [{}])
        first_alt = alternatives[0] if alternatives else {}
        
        # Extract text
        text = first_alt.get('transcript', '')
        
        # Extract confidence
        confidence = first_alt.get('confidence', 0.0)
        
        # Extract words with timing
        words = first_alt.get('words', [])
        
        # Extract utterances (speaker-aware segments)
        utterances = results.get('utterances', [])
        
        # Extract paragraphs
        paragraphs = first_alt.get('paragraphs', {}).get('paragraphs', [])
        
        # Extract unique speakers
        speakers = list(set(w.get('speaker', 0) for w in words if 'speaker' in w))
        
        # Get duration
        duration = response.get('metadata', {}).get('duration', 0.0)
        
        return cls(
            text=text,
            confidence=confidence,
            words=words,
            utterances=utterances,
            paragraphs=paragraphs,
            speakers=speakers,
            duration=duration,
            raw_response=response
        )
    
    def get_speaker_text(self, speaker_id: int) -> str:
        """Get all text from a specific speaker."""
        speaker_words = [w['word'] for w in self.words if w.get('speaker') == speaker_id]
        return ' '.join(speaker_words)
    
    def get_formatted_transcript(self) -> str:
        """Get transcript formatted with speaker labels."""
        if not self.utterances:
            return self.text
        
        lines = []
        for utt in self.utterances:
            speaker = utt.get('speaker', 0)
            text = utt.get('transcript', '')
            lines.append(f"Speaker {speaker}: {text}")
        
        return '\n'.join(lines)


class DeepgramClient:
    """
    Production-ready Deepgram API client.
    
    Supports:
    - File transcription (pre-recorded audio)
    - URL transcription (audio from URL)
    - Real-time streaming transcription
    - Async operations
    
    Args:
        api_key: Deepgram API key (or set DEEPGRAM_API_KEY env var)
        base_url: Base API URL
        timeout: Request timeout in seconds
    """
    
    BASE_URL = "https://api.deepgram.com/v1"
    WEBSOCKET_URL = "wss://api.deepgram.com/v1/listen"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 300
    ):
        self.api_key = api_key or os.environ.get('DEEPGRAM_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Deepgram API key required. Set DEEPGRAM_API_KEY env var or pass api_key."
            )
        
        self.base_url = base_url or self.BASE_URL
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
                headers={
                    "Authorization": f"Token {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
        return self._session
    
    async def close(self):
        """Close the client session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def _aenter_(self):
        return self
    
    async def _aexit_(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def transcribe_file(
        self,
        audio_file: str | Path | BinaryIO,
        config: Optional[TranscriptionConfig] = None,
        mimetype: str = "audio/wav"
    ) -> TranscriptionResult:
        """
        Transcribe an audio file.
        
        Args:
            audio_file: Path to audio file or file-like object
            config: Transcription configuration
            mimetype: Audio file MIME type
        
        Returns:
            TranscriptionResult object
        """
        config = config or TranscriptionConfig()
        session = await self._get_session()
        
        # Read file data
        if isinstance(audio_file, (str, Path)):
            with open(audio_file, 'rb') as f:
                audio_data = f.read()
        else:
            audio_data = audio_file.read()

        # Build URL with query params
        url = f"{self.base_url}/listen"
        params = config.to_query_params()
        
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": mimetype
        }
        
        try:
            async with session.post(
                url,
                params=params,
                data=audio_data,
                headers=headers
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return TranscriptionResult.from_response(data)
                
        except aiohttp.ClientError as e:
            logger.error(f"Transcription request failed: {e}")
            raise
    
    async def transcribe_url(
        self,
        audio_url: str,
        config: Optional[TranscriptionConfig] = None
    ) -> TranscriptionResult:
        """
        Transcribe audio from a URL.
        
        Args:
            audio_url: URL of the audio file
            config: Transcription configuration
        
        Returns:
            TranscriptionResult object
        """
        config = config or TranscriptionConfig()
        session = await self._get_session()
        
        url = f"{self.base_url}/listen"
        params = config.to_query_params()
        
        payload = {"url": audio_url}
        
        try:
            async with session.post(
                url,
                params=params,
                json=payload
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return TranscriptionResult.from_response(data)
                
        except aiohttp.ClientError as e:
            logger.error(f"URL transcription request failed: {e}")
            raise
    
    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        config: Optional[TranscriptionConfig] = None,
        sample_rate: int = 16000,
        encoding: str = "linear16"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Transcribe audio in real-time using WebSocket.
        
        Args:
            audio_stream: Async generator yielding audio chunks
            config: Transcription configuration
            sample_rate: Audio sample rate
            encoding: Audio encoding format
        
        Yields:
            Partial transcription results
        """
        config = config or TranscriptionConfig()
        params = config.to_query_params()
        params.update({
            "encoding": encoding,
            "sample_rate": str(sample_rate),
            "interim_results": "true"
        })
        
        # Build WebSocket URL
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        ws_url = f"{self.WEBSOCKET_URL}?{query_string}"
        
        headers = {"Authorization": f"Token {self.api_key}"}
        
        try:
            async with websockets.connect(ws_url, extra_headers=headers) as ws:
                # Start receiving task
                async def receive():
                    async for message in ws:
                        data = json.loads(message)
                        yield data
                
                # Start sending task
                async def send():
                    async for chunk in audio_stream:
                        await ws.send(chunk)
                    # Send close message
                    await ws.send(json.dumps({"type": "CloseStream"}))
                
                # Run both tasks
                receive_task = asyncio.create_task(receive()._anext_())
                send_task = asyncio.create_task(send())
                
                async for result in receive():
                    yield result
                
        except Exception as e:
            logger.error(f"Streaming transcription failed: {e}")
            raise


class DeepgramSyncClient:
    """
    Synchronous wrapper for DeepgramClient.
    
    For use in non-async contexts.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 300
    ):
        self._async_client = DeepgramClient(api_key, base_url, timeout)
    
    def _run_async(self, coro):
        """Run async coroutine in sync context."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    
    def transcribe_file(
        self,
        audio_file: str | Path | BinaryIO,
        config: Optional[TranscriptionConfig] = None,
        mimetype: str = "audio/wav"
    ) -> TranscriptionResult:
        """Transcribe an audio file (sync)."""
        return self._run_async(
            self._async_client.transcribe_file(audio_file, config, mimetype)
        )
    
    def transcribe_url(
        self,
        audio_url: str,
        config: Optional[TranscriptionConfig] = None
    ) -> TranscriptionResult:
        """Transcribe audio from URL (sync)."""
        return self._run_async(
            self._async_client.transcribe_url(audio_url, config)
        )
    
    def close(self):
        """Close the client."""
        self._run_async(self._async_client.close())


def transcribe_audio(
    audio_source: str,
    api_key: Optional[str] = None,
    config: Optional[TranscriptionConfig] = None
) -> TranscriptionResult:
    """
    Convenience function for one-off transcription.
    
    Args:
        audio_source: File path or URL
        api_key: Deepgram API key
        config: Transcription configuration
    
    Returns:
        TranscriptionResult object
    """
    client = DeepgramSyncClient(api_key)
    
    try:
        if audio_source.startswith(('http://', 'https://')):
            return client.transcribe_url(audio_source, config)
        else:
            return client.transcribe_file(audio_source, config)
    finally:
        client.close()