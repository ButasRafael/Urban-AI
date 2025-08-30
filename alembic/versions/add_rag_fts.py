"""Add FTS support for BM25 search in RAG chunks

Revision ID: add_rag_fts
Revises: 31c58f570c1a
Create Date: 2025-08-30

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision = 'add_rag_fts'
down_revision = '31c58f570c1a'
branch_labels = None
depends_on = None


def upgrade():
    # Create unaccent extension for handling diacritics
    op.execute(text("CREATE EXTENSION IF NOT EXISTS unaccent"))
    
    # Create immutable wrapper for unaccent (required for generated columns)
    op.execute(text("""
        CREATE OR REPLACE FUNCTION immutable_unaccent(text)
        RETURNS text AS $$
            SELECT unaccent($1)
        $$ LANGUAGE sql IMMUTABLE PARALLEL SAFE;
    """))
    
    # Add FTS column with weighted fields: class_name > address > chunk
    # Bilingual: 'english' for descriptions, 'romanian' for addresses
    op.execute(text("""
        ALTER TABLE rag_chunks
        ADD COLUMN IF NOT EXISTS tsv tsvector GENERATED ALWAYS AS (
            setweight(to_tsvector('english', immutable_unaccent(coalesce(class_name, ''))), 'A') ||
            setweight(to_tsvector('romanian', immutable_unaccent(coalesce(address,    ''))), 'B') ||
            setweight(to_tsvector('english', immutable_unaccent(coalesce(chunk,      ''))), 'C')
        ) STORED
    """))
    
    # Create GIN index for fast FTS queries
    op.execute(text("CREATE INDEX IF NOT EXISTS idx_rag_chunks_tsv ON rag_chunks USING GIN (tsv)"))
    
    # Optional: Add pg_trgm for fuzzy matching support (commented out for now)
    # op.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))


def downgrade():
    # Drop the FTS index
    op.execute(text("DROP INDEX IF EXISTS idx_rag_chunks_tsv"))
    
    # Drop the FTS column
    op.execute(text("ALTER TABLE rag_chunks DROP COLUMN IF EXISTS tsv"))
    
    # Drop the immutable wrapper function
    op.execute(text("DROP FUNCTION IF EXISTS immutable_unaccent(text)"))
    
    # Optionally drop unaccent extension (might be used elsewhere)
    op.execute(text("DROP EXTENSION IF EXISTS unaccent"))