from adapters.text import ContextualTextEncoder
from adapters.vision import MultiModalEncoder
from typing import List, Literal, Dict, cast, Optional
from core.types import EmbeddingOperation, EmbeddingItem
from utils import track_latency
from core.aggregators import EmbeddingAggregator


class EmbeddingService:
    def __init__(
            self,
            text_encoder: ContextualTextEncoder,
            aggregator: EmbeddingAggregator,
            # video_encoder: MultiModalEncoder
    ):
        self.text_encoder = text_encoder
        self.aggregator = aggregator
        # self.video_encoder = video_encoder

    def embed_text(
            self,
            model_name: Optional[str],
            text: str | List[str],
            normalize: bool = True,
            strategy: Literal["auto", "truncate", "mean_pool", "max_pool", "stride"] | None = "auto",
            chunk_size: int | None = None,
            stride: int = 50,
            contextual_focus: list[str] | dict[str, float] | None = None,
            focus_weight: float = 2.0) -> EmbeddingOperation:
        if model_name:
            self.text_encoder = self.text_encoder.with_model(model_name)

        with track_latency() as meta:
            vec = self.text_encoder.encode(
                text,
                normalize,
                strategy,
                chunk_size,
                stride,
                contextual_focus,
                focus_weight
            )

        return EmbeddingOperation(
            vector=cast(List[float], vec.tolist()),
            model_info=self.text_encoder.get_model_info(),
            latency_ms=meta.get('latency_ms', None)
        )

    def aggregate_embeddings(
            self,
            weighted_embeddings: List[EmbeddingItem],
            strategy: str = "weighted_mean",
            model_name: Optional[str] = None,
            query: Optional[List[float]] = None
    ) -> EmbeddingOperation:

        with track_latency() as meta:
            vec = self.aggregator.aggregate(weighted_embeddings, strategy, query)

        return EmbeddingOperation(
            vector=vec,
            latency_ms=meta.get('latency_ms', None)
        )
