from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor

POSTGRES_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@db:5432/urban_ai"
)

engine = create_engine(POSTGRES_URL, echo=True)
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
    from app.models import media
    Base.metadata.create_all(bind=engine)