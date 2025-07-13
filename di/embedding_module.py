import injector
from core.services.embedding_service import EmbeddingService
from adapters.text.sentence_transformer import SentenceTransformerAdapter
from adapters.vision.clip import CLIPAdapter
from adapters.text.base import ContextualTextEncoder
from adapters.vision.base import MultiModalEncoder
from core.aggregators import EmbeddingAggregator


class EmbeddingModule(injector.Module):
    def configure(self, binder: injector.Binder) -> None:
        binder.bind(ContextualTextEncoder, to=SentenceTransformerAdapter, scope=injector.singleton)
        binder.bind(EmbeddingAggregator, to=EmbeddingAggregator, scope=injector.singleton)
        # binder.bind(MultiModalEncoder, to=CLIPAdapter, scope=injector.singleton)

    @injector.singleton
    @injector.provider
    def provide_embedding_service(
        self,
        text_encoder: ContextualTextEncoder,
        aggregator: EmbeddingAggregator
        # video_encoder: MultiModalEncoder
    ) -> EmbeddingService:
        return EmbeddingService(text_encoder=text_encoder, aggregator=aggregator)
