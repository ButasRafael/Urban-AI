"""
Shared pytest configuration for all tests
"""
import pytest
from sqlalchemy import create_engine, text
from app.core.database import Base

# Import all models to ensure they're registered with Base.metadata
import app.models.user
import app.models.media
import app.models.revoked
import app.models.rag
import app.models.conversation


# Database configuration for testing
SQLALCHEMY_DATABASE_URL = "postgresql://postgres:postgres@db:5432/urban_ai_test"
engine = create_engine(SQLALCHEMY_DATABASE_URL)


@pytest.fixture(scope="session")
def anyio_backend():
    """Configure anyio to only use asyncio backend"""
    return "asyncio"


@pytest.fixture(scope="session", autouse=True)
def prepare_db():
    """Create and cleanup test database tables with migrations"""
    # Create required PostgreSQL extensions BEFORE creating tables
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS unaccent"))
        conn.commit()

    # Create all tables
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    # Apply Alembic migration changes that aren't in model definitions
    # This creates the tsv column for RAG full-text search
    with engine.connect() as conn:

        # Create immutable unaccent function
        conn.execute(text("""
            CREATE OR REPLACE FUNCTION immutable_unaccent(text)
            RETURNS text AS $$
                SELECT unaccent($1)
            $$ LANGUAGE sql IMMUTABLE PARALLEL SAFE;
        """))

        # Add tsv generated column for full-text search
        conn.execute(text("""
            ALTER TABLE rag_chunks
            ADD COLUMN IF NOT EXISTS tsv tsvector GENERATED ALWAYS AS (
                setweight(to_tsvector('english', immutable_unaccent(coalesce(class_name, ''))), 'A') ||
                setweight(to_tsvector('romanian', immutable_unaccent(coalesce(address,    ''))), 'B') ||
                setweight(to_tsvector('english', immutable_unaccent(coalesce(chunk,      ''))), 'C')
            ) STORED
        """))

        # Create GIN index for full-text search
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_rag_chunks_tsv ON rag_chunks USING GIN (tsv)"))

        conn.commit()

    yield

    # Cleanup after all tests
    Base.metadata.drop_all(bind=engine)