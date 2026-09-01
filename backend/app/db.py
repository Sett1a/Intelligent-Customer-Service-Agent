"""数据库引擎与会话工厂(SQLite + aiosqlite)。"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import DB_PATH
from app.models import Base

engine = create_async_engine(f"sqlite+aiosqlite:///{DB_PATH.as_posix()}")
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    async with SessionLocal() as session:
        yield session


async def one_or_none(db: AsyncSession, stmt):
    return (await db.execute(stmt)).scalar_one_or_none()


async def session_by_id(db: AsyncSession, session_id: str):
    from app.models import SessionModel

    return await one_or_none(db, select(SessionModel).where(SessionModel.id == session_id))
