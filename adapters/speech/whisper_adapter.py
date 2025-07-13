from functools import lru_cache
from typing import Dict, Optional, List, Union
import torch
import numpy as np
import whisper
from whisper.tokenizer import LANGUAGES as WHISPER_LANGUAGES
from adapters.types import TranscriptionResult, TranscriptionSegment, ModelInfo
from adapters.speech.base import SpeechRecognizer

_DEFAULT_WHISPER_MODEL = "base"


@lru_cache(maxsize=4)
def get_whisper_model(model_name: str = _DEFAULT_WHISPER_MODEL, device: str = "cpu"):
    """Load and cache Whisper model directly on the desired device."""
    return whisper.load_model(model_name, device=device)


class WhisperAdapter(SpeechRecognizer):
    """
    Adapter for OpenAI Whisper ASR model with lazy loading.
    """

    def __init__(
            self,
            model_name: str = _DEFAULT_WHISPER_MODEL,
            device: Optional[str] = None
    ):
        self._model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None  # Lazy loading
        self._model_info_cache = None

    @property
    def model(self):
        """Lazy load and cache the model."""
        if self._model is None:
            self._model = get_whisper_model(self._model_name, self.device)
        return self._model

    @property
    def is_multilingual(self) -> bool:
        """Determine if model is multilingual from name."""
        return self._model_name not in ["tiny.en", "base.en", "small.en", "medium.en"]

    @property
    def supported_languages(self) -> List[str]:
        """Get supported languages."""
        return list(WHISPER_LANGUAGES.keys()) if self.is_multilingual else ["en"]

    def transcribe(
            self,
            audio: Union[str, np.ndarray],
            language: Optional[str] = None,
            task: str = "transcribe",
            **kwargs
    ) -> TranscriptionResult:
        """Transcribe audio file to text using Whisper."""
        if isinstance(audio, str):
            audio_input = audio
        elif isinstance(audio, np.ndarray):
            # Ensure float32 and normalized
            audio_input = audio.astype(np.float32)
            audio_input = whisper.pad_or_trim(audio_input)
        else:
            raise ValueError(f"Unsupported audio type: {type(audio)}")

        if task not in ["transcribe", "translate"]:
            raise ValueError(f"Invalid task: {task}. Must be 'transcribe' or 'translate'")

        if not self.is_multilingual and language and language != "en":
            raise ValueError(f"Model {self._model_name} only supports English")

        result = self.model.transcribe(
            audio_input,
            language=language,
            task=task,
            **kwargs
        )

        segments = []
        for seg in result.get("segments", []):
            segments.append(TranscriptionSegment(
                text=seg["text"],
                start=seg["start"],
                end=seg["end"],
                confidence=seg.get("confidence"),
                words=seg.get("words")
            ))

        return TranscriptionResult(
            text=result["text"],
            language=result.get("language"),
            segments=segments,
            metadata={
                "task": task,
                "duration": result.get("duration"),
                "model": self._model_name
            }
        )

    def transcribe_batch(
            self,
            audio_files: List[Union[str, np.ndarray]],
            language: Optional[str] = None,
            batch_size: Optional[int] = None,
            **kwargs
    ) -> List[TranscriptionResult]:
        """Transcribe multiple audio files."""
        return [
            self.transcribe(audio, language=language, **kwargs)
            for audio in audio_files
        ]

    def detect_language(
            self,
            audio: Union[str, np.ndarray],
            **kwargs
    ) -> Dict[str, float]:
        """Detect language using Whisper."""
        if not self.is_multilingual:
            return {"en": 1.0}

        if isinstance(audio, str):
            audio_array = whisper.load_audio(audio)
        elif isinstance(audio, np.ndarray):
            audio_array = whisper.pad_or_trim(audio.astype(np.float32))
        else:
            raise ValueError(f"Unsupported audio type for language detection: {type(audio)}")

        mel = whisper.log_mel_spectrogram(audio_array).to(self.device)
        _, probs = self.model.detect_language(mel)

        return dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))

    def get_model_info(self) -> ModelInfo:
        """Return model information."""
        if self._model_info_cache is None:
            is_multilingual = self.is_multilingual
            model_dims = {
                "tiny": {"n_audio_state": 384, "n_audio_ctx": 1500},
                "tiny.en": {"n_audio_state": 384, "n_audio_ctx": 1500},
                "base": {"n_audio_state": 512, "n_audio_ctx": 1500},
                "base.en": {"n_audio_state": 512, "n_audio_ctx": 1500},
                "small": {"n_audio_state": 768, "n_audio_ctx": 1500},
                "small.en": {"n_audio_state": 768, "n_audio_ctx": 1500},
                "medium": {"n_audio_state": 1024, "n_audio_ctx": 1500},
                "medium.en": {"n_audio_state": 1024, "n_audio_ctx": 1500},
                "large": {"n_audio_state": 1280, "n_audio_ctx": 1500},
                "large-v2": {"n_audio_state": 1280, "n_audio_ctx": 1500},
                "large-v3": {"n_audio_state": 1280, "n_audio_ctx": 1500},
            }
            dims = model_dims.get(self._model_name, {"n_audio_state": 1280, "n_audio_ctx": 1500})

            self._model_info_cache = ModelInfo(
                name=f"whisper:{self._model_name}",
                embedding_dim=dims["n_audio_state"],
                max_length=dims["n_audio_ctx"],
                metadata={
                    "backend": "whisper",
                    "model_variant": self._model_name,
                    "multilingual": is_multilingual,
                    "supported_languages": self.supported_languages,
                    "n_mels": 80,
                    "sample_rate": 16000,
                    "supported_tasks": ["transcribe", "translate"] if is_multilingual else ["transcribe"]
                }
            )
        return self._model_info_cache
