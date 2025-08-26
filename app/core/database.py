from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import sessionmaker


from config import get_settings, get_sql_db_path
from models.database import Base

settings = get_settings()

async def init_db():
    async_engine = PostgresEngineManager.init_engine()
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

class PostgresEngineManager:
    """
    Singleton manager for AsyncEngine and AsyncSession creation.
    """
    _engine: Optional[AsyncEngine] = None
    _SessionLocal: Optional[sessionmaker] = None

    @classmethod
    def init_engine(cls, uri: Optional[str] = None) -> AsyncEngine:
        if cls._engine is None:
            db_uri = uri or get_sql_db_path()
            cls._engine = create_async_engine(
                db_uri,
                echo=settings.debug,
                pool_pre_ping=True,
                pool_size=15,
                max_overflow=30,
            )
        return cls._engine

    @classmethod
    def init_sessionmaker(cls) -> sessionmaker:
        if cls._SessionLocal is None:
            engine = cls.init_engine()
            cls._SessionLocal = sessionmaker(
                bind=engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
        return cls._SessionLocal

    @classmethod
    @asynccontextmanager
    async def get_session(cls) -> AsyncGenerator[AsyncSession, None]:
        SessionLocal = cls.init_sessionmaker()
        async with SessionLocal() as session:
            try:
                yield session
                await session.commit()
            except:
                await session.rollback()
                raise
            finally:
                await session.close()

    @classmethod
    async def dispose_engine(cls) -> None:
        if cls._engine:
            await cls._engine.dispose()
            cls._engine = None
            cls._SessionLocal = None
