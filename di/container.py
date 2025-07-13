from injector import Injector
from .embedding_module import EmbeddingModule
from core.services.embedding_service import EmbeddingService
from typing import Dict, Any

injector = Injector([EmbeddingModule()])

bindings: Dict[str, Any] = {
    "embedding_service": EmbeddingService
}


def get_dependency(name: str):
    return injector.get(bindings.get(name))
