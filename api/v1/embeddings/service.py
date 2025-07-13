from core.services.embedding_service import EmbeddingService
from di.container import get_dependency


def embedding_service() -> EmbeddingService:
    return get_dependency("embedding_service")
