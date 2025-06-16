from sqlmodel import create_async_engine, SQLModel, Field, Relationship, Column, DateTime
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.orm import sessionmaker # For async session maker
from sqlalchemy.ext.asyncio import AsyncEngine # For type hinting
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, List
import datetime

from app.core.config import settings
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.LOG_LEVEL.upper())

engine: Optional[AsyncEngine] = None
AsyncSessionLocal: Optional[sessionmaker] = None

def init_db_engine():
    global engine, AsyncSessionLocal
    if not settings.DATABASE_URL:
        logger.error("DATABASE_URL not configured. Cannot initialize database engine.")
        raise ValueError("DATABASE_URL is not set")
    try:
        engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True)
        # future=True enables 2.0 style execution
        AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        logger.info(f"Async database engine initialized for URL: {settings.DATABASE_URL.split('@')[-1] if '@' in settings.DATABASE_URL else settings.DATABASE_URL}")
    except Exception as e:
        logger.error(f"Failed to initialize database engine: {e}")
        engine = None
        AsyncSessionLocal = None
        raise

async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    if AsyncSessionLocal is None:
        logger.error("Database not initialized. AsyncSessionLocal is None.")
        # This indicates init_db_engine was not called or failed.
        # Depending on desired behavior, could raise an error or try to init here.
        raise RuntimeError("Database session factory not available.")
    async with AsyncSessionLocal() as session:
        yield session

async def create_db_and_tables():
    if engine is None:
        logger.error("Database engine not initialized. Cannot create tables.")
        return
    async with engine.begin() as conn:
        # await conn.run_sync(SQLModel.metadata.drop_all) # Use with caution in dev
        await conn.run_sync(SQLModel.metadata.create_all)
    logger.info("Database tables created (if they didn't exist).")
