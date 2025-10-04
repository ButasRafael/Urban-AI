from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor

POSTGRES_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@db:5432/urban_ai"
)

engine = create_engine(
    POSTGRES_URL,
    echo=True,
    # Connection pool configuration
    pool_size=20,  # Number of connections to maintain in pool
    max_overflow=40,  # Maximum overflow connections beyond pool_size
    pool_pre_ping=True,  # Test connections before using them
    pool_recycle=3600,  # Recycle connections after 1 hour
    connect_args={
        "connect_timeout": 10,  # Connection timeout in seconds
        "application_name": "urban-ai",  # Helps identify connections in pg_stat_activity
        "options": "-c statement_timeout=30000"  # 30 second query timeout
    }
)
SQLAlchemyInstrumentor().instrument(engine=engine)
Psycopg2Instrumentor().instrument()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db() -> None:
    from app.models import media, rag, notification
    Base.metadata.create_all(bind=engine)
