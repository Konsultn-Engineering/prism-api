from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, BinaryIO
import numpy as np
from adapters.types import ModelInfo, TranscriptionResult, TranscriptionSegment


class SpeechRecognizer(ABC):
    """Abstract interface for speech recognition models."""

    @abstractmethod
    def transcribe(
            self,
            audio: Union[str, np.ndarray, BinaryIO],
            language: Optional[str] = None,
            task: str = "transcribe",
            **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.

        Args:
            audio: Audio input (file path, numpy array, or file-like object)
            language: Language code (e.g., 'en', 'es') or None for auto-detect
            task: Task type ('transcribe' or 'translate' for multilingual models)
            **kwargs: Model-specific parameters

        Returns:
            TranscriptionResult with text and metadata
        """
        pass

    @abstractmethod
    def transcribe_batch(
            self,
            audio_files: List[Union[str, np.ndarray]],
            language: Optional[str] = None,
            batch_size: Optional[int] = None,
            **kwargs
    ) -> List[TranscriptionResult]:
        """Transcribe multiple audio files."""
        pass

    @abstractmethod
    def detect_language(
            self,
            audio: Union[str, np.ndarray, BinaryIO],
            **kwargs
    ) -> Dict[str, float]:
        """
        Detect language of audio.

        Returns:
            Dict mapping language codes to confidence scores
        """
        pass

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """Get information about the model."""
        pass


class SpeechSynthesizer(ABC):
    """Abstract interface for text-to-speech models."""

    @abstractmethod
    def synthesize(
            self,
            text: str,
            voice: Optional[str] = None,
            language: Optional[str] = None,
            **kwargs
    ) -> np.ndarray:
        """
        Convert text to speech.

        Args:
            text: Text to synthesize
            voice: Voice ID or speaker name
            language: Language code
            **kwargs: Model-specific parameters

        Returns:
            Audio array (numpy array)
        """
        pass

    @abstractmethod
    def get_voices(self) -> List[Dict[str, Any]]:
        """Get available voices."""
        pass