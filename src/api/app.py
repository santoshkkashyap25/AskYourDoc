"""FastAPI application factory and middleware configuration."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.routes import router
from src.config import settings
from src.logging_config import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle handler."""
    logger.info("Starting %s...", settings.APP_NAME)
    settings.ensure_directories()

    # Pre-warm core components so the first user query has zero lag
    try:
        from src.services.container import container
        logger.info("Pre-warming RAG components (embeddings, vector store, LLM)...")
        _ = container.embeddings
        _ = container.vector_store
        _ = container.llm
        logger.info("RAG components pre-warmed and ready.")
    except Exception as e:
        logger.warning("Failed to pre-warm RAG components: %s", e)

    yield
    logger.info("Shutting down %s...", settings.APP_NAME)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.APP_NAME,
        description="Modular, Production-Ready OOP Standard RAG Platform for PDF Q&A",
        version="2.0.0",
        lifespan=lifespan
    )

    # Cross-Origin Resource Sharing (CORS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static assets
    static_dir = settings.BASE_DIR / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Include routes
    app.include_router(router)

    return app
