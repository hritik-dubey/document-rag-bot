import logging
from fastapi import FastAPI
from app.services.vector_store import create_collection_if_not_exists, get_qdrant_client
from app.core.config import settings
from app.core import db

app = FastAPI(title="RAG Chatbot API", version="0.1.0")

@app.on_event("startup")
async def startup_event():
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=settings.LOG_LEVEL.upper())
    logger.info("Starting up the application...")
    get_qdrant_client() # Initialize client
    create_collection_if_not_exists(settings.QDRANT_COLLECTION_NAME) # Ensure collection exists

    logger.info("Initializing database engine...")
    db.init_db_engine()
    await db.create_db_and_tables()
    logger.info("Database initialization complete.")
    logger.info("Application startup complete.")

@app.get("/health", tags=["Health Check"])
async def health_check():
    return {"status": "ok"}

from app.api.v1 import auth_router, documents_router, chat_router

app.include_router(auth_router.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(documents_router.router, prefix="/api/v1/documents", tags=["Documents"])
app.include_router(chat_router.router, prefix="/api/v1/chat", tags=["Chat"])
