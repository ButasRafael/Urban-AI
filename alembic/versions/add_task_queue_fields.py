"""Add task queue fields to media table

Revision ID: add_task_queue_fields
Revises: add_rag_fts
Create Date: 2025-01-01

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'add_task_queue_fields'
down_revision = 'add_rag_fts'
branch_labels = None
depends_on = None


def upgrade():
    # Create ProcessingStatus enum type
    processing_status_enum = postgresql.ENUM(
        'pending', 'processing', 'completed', 'failed',
        name='processing_status_enum',
        create_type=True
    )
    processing_status_enum.create(op.get_bind(), checkfirst=True)

    # Add new columns to media table
    op.add_column('media', sa.Column('task_id', sa.String(), nullable=True))
    op.add_column('media', sa.Column(
        'processing_status',
        sa.Enum('pending', 'processing', 'completed', 'failed',
                name='processing_status_enum'),
        nullable=False,
        server_default='completed'  # Default for existing records
    ))
    op.add_column('media', sa.Column('error_message', sa.Text(), nullable=True))
    op.add_column('media', sa.Column('started_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('media', sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True))

    # Create index on task_id for faster lookups
    op.create_index('ix_media_task_id', 'media', ['task_id'])

    # Update existing records to have completed status (they were processed synchronously)
    op.execute("UPDATE media SET processing_status = 'completed' WHERE processing_status IS NULL")


def downgrade():
    # Drop the index
    op.drop_index('ix_media_task_id', table_name='media')

    # Drop the columns
    op.drop_column('media', 'completed_at')
    op.drop_column('media', 'started_at')
    op.drop_column('media', 'error_message')
    op.drop_column('media', 'processing_status')
    op.drop_column('media', 'task_id')

    # Drop the enum type
    processing_status_enum = postgresql.ENUM(
        'pending', 'processing', 'completed', 'failed',
        name='processing_status_enum'
    )
    processing_status_enum.drop(op.get_bind(), checkfirst=True)