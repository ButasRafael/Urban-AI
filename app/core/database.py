from sqlalchemy import create_engine, event, text
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

@event.listens_for(engine, "connect")
def _set_ivfflat_probes(dbapi_conn, _conn_record):
    cur = dbapi_conn.cursor()
    try:
        try:
            cur.execute("SET ivfflat.probes = 10;")
        except Exception:
            pass
    finally:
        cur.close()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db() -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

    from app.models import media, rag 
    Base.metadata.create_all(bind=engine)
