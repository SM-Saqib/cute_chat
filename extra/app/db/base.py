# create db sessions , using sql alchemy , for postgresql


from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from sqlalchemy import text
from sqlalchemy import inspect

from sqlalchemy.ext.asyncio import AsyncSession,create_async_engine



SQLALCHEMY_DATABASE_URL = f"postgresql://{os.getenv('PGUSER')}:{os.getenv('PGPASSWORD')}@{os.getenv('PGHOST')}:{os.getenv('PGPORT')}/{os.getenv('PGDATABASE')}"

SQLALCHEMY_DATABASE_URL_ASYNC = f"postgresql+asyncpg://{os.getenv('PGUSER')}:{os.getenv('PGPASSWORD')}@{os.getenv('PGHOST')}:{os.getenv('PGPORT')}/{os.getenv('PGDATABASE')}"

print(SQLALCHEMY_DATABASE_URL)

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_size=30,  # Default is 5
    max_overflow=20,  # Default is 10
    pool_timeout=30,  # Default is 10 seconds
    pool_recycle=1800,  # Recycle connections after 1 hour
)

async_engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL_ASYNC,
    pool_size=30,  # Default is 5
    max_overflow=20,  # Default is 10
    pool_timeout=30,  # Default is 10 seconds
    pool_recycle=1800,  # Recycle connections after 1 hour
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()


def get_async_db():
    db = AsyncSession(bind=async_engine)
    try:
        return db
    finally:
        db.close()

