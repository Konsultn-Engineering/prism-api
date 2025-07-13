from fastapi import FastAPI
from api import v1 as v1_router  # <--- import the shared router


def create_app():
    # config = load_config()
    app = FastAPI(title="Embeddings Server")
    # app.state.config = config

    # Mount the shared embeddings router with a top-level prefix (e.g., /v1)
    app.include_router(v1_router, prefix="/api")

    return app
