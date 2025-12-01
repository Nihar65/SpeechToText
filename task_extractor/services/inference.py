"""
Inference Pipeline
==================
End-to-end pipeline for task extraction from audio or text.
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import logging
import time

from .deepgram_client import DeepgramClient, DeepgramSyncClient, TranscriptionConfig, TranscriptionResult
from .task_extractor import TaskExtractorService, ExtractionResult, ExtractedTask

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete result from the inference pipeline."""
    tasks: List[Dict]
    transcription: Optional[str]
    original_input_type: str  # 'audio', 'url', or 'text'
    processing_time_ms: float
    transcription_time_ms: Optional[float]
    extraction_time_ms: float
    speaker_count: int
    audio_duration_seconds: Optional[float]
    model_confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), indent=2)


class InferencePipeline:
    """
    Production inference pipeline for task extraction.
    
    Supports:
    - Audio file input (with Deepgram transcription)
    - Audio URL input (with Deepgram transcription)
    - Direct text input
    - Batch processing
    
    Args:
        task_extractor: TaskExtractorService instance
        deepgram_api_key: Deepgram API key for audio transcription
        transcription_config: Configuration for Deepgram transcription
    """
    
    def __init__(
        self,
        task_extractor: Optional[TaskExtractorService] = None,
        deepgram_api_key: Optional[str] = None,
        transcription_config: Optional[TranscriptionConfig] = None,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None
    ):
        # Initialize task extractor
        if task_extractor:
            self.task_extractor = task_extractor
        else:
            self.task_extractor = TaskExtractorService(
                model_path=model_path,
                tokenizer_path=tokenizer_path
            )
        
        # Initialize Deepgram client
        self.deepgram_key = deepgram_api_key
        self.transcription_config = transcription_config or TranscriptionConfig()
        
        # Lazy initialization of Deepgram client
        self._deepgram_client: Optional[DeepgramSyncClient] = None
        self._async_deepgram_client: Optional[DeepgramClient] = None
    
    def _get_deepgram_client(self) -> DeepgramSyncClient:
        """Get or create sync Deepgram client."""
        if self._deepgram_client is None:
            if not self.deepgram_key:
                raise ValueError("Deepgram API key required for audio transcription")
            self._deepgram_client = DeepgramSyncClient(self.deepgram_key)
        return self._deepgram_client
    
    async def _get_async_deepgram_client(self) -> DeepgramClient:
        """Get or create async Deepgram client."""
        if self._async_deepgram_client is None:
            if not self.deepgram_key:
                raise ValueError("Deepgram API key required for audio transcription")
            self._async_deepgram_client = DeepgramClient(self.deepgram_key)
        return self._async_deepgram_client
    
    def process_audio_file(
        self,
        audio_path: Union[str, Path],
        confidence_threshold: float = 0.5
    ) -> PipelineResult:
        """
        Process an audio file to extract tasks.
        
        Args:
            audio_path: Path to the audio file
            confidence_threshold: Minimum confidence for task inclusion
        
        Returns:
            PipelineResult with extracted tasks
        """
        start_time = time.time()
        
        # Transcribe audio
        transcription_start = time.time()
        client = self._get_deepgram_client()
        transcription_result = client.transcribe_file(
            audio_path,
            config=self.transcription_config
        )
        transcription_time = (time.time() - transcription_start) * 1000
        
        # Extract tasks from transcription
        extraction_result = self.task_extractor.extract_tasks(
            transcription_result.text,
            confidence_threshold=confidence_threshold
        )
        
        total_time = (time.time() - start_time) * 1000
        
        return PipelineResult(
            tasks=[t.to_dict() for t in extraction_result.tasks],
            transcription=transcription_result.text,
            original_input_type='audio',
            processing_time_ms=total_time,
            transcription_time_ms=transcription_time,
            extraction_time_ms=extraction_result.extraction_time_ms,
            speaker_count=len(transcription_result.speakers),
            audio_duration_seconds=transcription_result.duration,
            model_confidence=extraction_result.model_confidence
        )
    
    def process_audio_url(
        self,
        audio_url: str,
        confidence_threshold: float = 0.5
    ) -> PipelineResult:
        """
        Process audio from a URL to extract tasks.
        
        Args:
            audio_url: URL of the audio file
            confidence_threshold: Minimum confidence for task inclusion
        
        Returns:
            PipelineResult with extracted tasks
        """
        start_time = time.time()
        
        # Transcribe audio
        transcription_start = time.time()
        client = self._get_deepgram_client()
        transcription_result = client.transcribe_url(
            audio_url,
            config=self.transcription_config
        )
        transcription_time = (time.time() - transcription_start) * 1000
        
        # Extract tasks from transcription
        extraction_result = self.task_extractor.extract_tasks(
            transcription_result.text,
            confidence_threshold=confidence_threshold
        )
        
        total_time = (time.time() - start_time) * 1000
        
        return PipelineResult(
            tasks=[t.to_dict() for t in extraction_result.tasks],
            transcription=transcription_result.text,
            original_input_type='url',
            processing_time_ms=total_time,
            transcription_time_ms=transcription_time,
            extraction_time_ms=extraction_result.extraction_time_ms,
            speaker_count=len(transcription_result.speakers),
            audio_duration_seconds=transcription_result.duration,
            model_confidence=extraction_result.model_confidence
        )
    
    def process_text(
        self,
        text: str,
        confidence_threshold: float = 0.5
    ) -> PipelineResult:
        """
        Process text directly to extract tasks.
        
        Args:
            text: Meeting transcript text
            confidence_threshold: Minimum confidence for task inclusion
        
        Returns:
            PipelineResult with extracted tasks
        """
        start_time = time.time()
        
        # Extract tasks
        extraction_result = self.task_extractor.extract_tasks(
            text,
            confidence_threshold=confidence_threshold
        )
        
        total_time = (time.time() - start_time) * 1000
        
        return PipelineResult(
            tasks=[t.to_dict() for t in extraction_result.tasks],
            transcription=text,
            original_input_type='text',
            processing_time_ms=total_time,
            transcription_time_ms=None,
            extraction_time_ms=extraction_result.extraction_time_ms,
            speaker_count=0,
            audio_duration_seconds=None,
            model_confidence=extraction_result.model_confidence
        )
    
    async def process_audio_file_async(
        self,
        audio_path: Union[str, Path],
        confidence_threshold: float = 0.5
    ) -> PipelineResult:
        """Async version of process_audio_file."""
        start_time = time.time()
        
        # Transcribe audio
        transcription_start = time.time()
        client = await self._get_async_deepgram_client()
        transcription_result = await client.transcribe_file(
            audio_path,
            config=self.transcription_config
        )
        transcription_time = (time.time() - transcription_start) * 1000
        
        # Extract tasks (sync for now, could be made async)
        extraction_result = self.task_extractor.extract_tasks(
            transcription_result.text,
            confidence_threshold=confidence_threshold
        )
        
        total_time = (time.time() - start_time) * 1000
        
        return PipelineResult(
            tasks=[t.to_dict() for t in extraction_result.tasks],
            transcription=transcription_result.text,
            original_input_type='audio',
            processing_time_ms=total_time,
            transcription_time_ms=transcription_time,
            extraction_time_ms=extraction_result.extraction_time_ms,
            speaker_count=len(transcription_result.speakers),
            audio_duration_seconds=transcription_result.duration,
            model_confidence=extraction_result.model_confidence
        )
    
    def process_batch(
        self,
        inputs: List[Dict[str, Any]],
        confidence_threshold: float = 0.5
    ) -> List[PipelineResult]:
        """
        Process a batch of inputs.
        
        Args:
            inputs: List of dicts with 'type' ('audio', 'url', 'text') and 'data' keys
            confidence_threshold: Minimum confidence for task inclusion
        
        Returns:
            List of PipelineResults
        """
        results = []
        
        for inp in inputs:
            input_type = inp.get('type', 'text')
            data = inp.get('data')
            
            if input_type == 'audio':
                result = self.process_audio_file(data, confidence_threshold)
            elif input_type == 'url':
                result = self.process_audio_url(data, confidence_threshold)
            else:
                result = self.process_text(data, confidence_threshold)
            
            results.append(result)
        
        return results
    
    async def process_batch_async(
        self,
        inputs: List[Dict[str, Any]],
        confidence_threshold: float = 0.5
    ) -> List[PipelineResult]:
        """Async version of process_batch."""
        tasks = []
        
        for inp in inputs:
            input_type = inp.get('type', 'text')
            data = inp.get('data')
            
            if input_type == 'audio':
                tasks.append(self.process_audio_file_async(data, confidence_threshold))
            elif input_type == 'url':
                # Wrap sync method
                tasks.append(asyncio.to_thread(
                    self.process_audio_url, data, confidence_threshold
                ))
            else:
                tasks.append(asyncio.to_thread(
                    self.process_text, data, confidence_threshold
                ))
        
        return await asyncio.gather(*tasks)
    
    def suggest_assignees(
        self,
        tasks: List[ExtractedTask]
    ) -> List[ExtractedTask]:
        """
        Suggest better assignees based on expertise matching.
        
        Args:
            tasks: List of extracted tasks
        
        Returns:
            Tasks with suggested assignees
        """
        for task in tasks:
            if task.assigned_to.lower() == 'unassigned':
                suggested, confidence = self.task_extractor.assign_based_on_expertise(
                    task.description
                )
                task.assigned_to = suggested.title()
                task.confidence = min(task.confidence, confidence)
        
        return tasks
    
    def cleanup(self):
        """Clean up resources."""
        if self._deepgram_client:
            self._deepgram_client.close()
        if self._async_deepgram_client:
            asyncio.run(self._async_deepgram_client.close())


def create_pipeline(
    model_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    deepgram_api_key: Optional[str] = None,
    team_members: Optional[Dict[str, List[str]]] = None
) -> InferencePipeline:
    """
    Factory function to create an inference pipeline.
    
    Args:
        model_path: Path to saved model
        tokenizer_path: Path to saved tokenizer
        deepgram_api_key: Deepgram API key
        team_members: Team member configuration
    
    Returns:
        Configured InferencePipeline
    """
    task_extractor = TaskExtractorService(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        team_members=team_members
    )
    
    return InferencePipeline(
        task_extractor=task_extractor,
        deepgram_api_key=deepgram_api_key
    )
