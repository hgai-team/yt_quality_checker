from fastapi import FastAPI
from contextlib import asynccontextmanager
from config import get_settings
from api.router import channels
from core.database import init_db
from core.qdrant_client import vector_store
import structlog

settings = get_settings()
logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing database...")
    await init_db()

    logger.info("Initializing vector store...")
    await vector_store.init_collections()

    logger.info("YouTube Quality Checker ready")

    yield

    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title=settings.app_name,
    version=settings.api_version,
    lifespan=lifespan
)

app.include_router(channels.router, prefix=f"/api/{settings.api_version}")


@app.get("/")
async def root():
    return {
        "app": settings.app_name,
        "version": settings.api_version,
        "status": "operational"
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}
