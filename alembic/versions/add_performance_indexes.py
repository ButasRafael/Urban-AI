"""Add performance indexes for media and detection tables

Revision ID: add_performance_indexes
Revises: add_task_queue_fields
Create Date: 2025-01-28 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_performance_indexes'
down_revision = 'add_task_queue_fields'
branch_labels = None
depends_on = None


def upgrade():
    # Create composite index on media table for user queries sorted by date
    # This speeds up queries like: SELECT * FROM media WHERE user_username = ? ORDER BY created_at DESC
    op.create_index(
        'ix_media_user_created',
        'media',
        ['user_username', sa.text('created_at DESC')],
        postgresql_using='btree'
    )

    # Create index on detection table for frame_id lookups
    # This speeds up joins between detection and frame tables
    op.create_index(
        'ix_detection_frame_id',
        'detection',
        ['frame_id'],
        postgresql_using='btree'
    )

    # Index for media status queries (finding processing/failed items)
    # This speeds up queries like: SELECT * FROM media WHERE processing_status = 'processing'
    op.create_index(
        'ix_media_processing_status',
        'media',
        ['processing_status'],
        postgresql_using='btree'
    )

    # Composite index for frame queries by media_id
    # This speeds up queries like: SELECT * FROM frame WHERE media_id = ? ORDER BY frame_index
    op.create_index(
        'ix_frame_media_id_index',
        'frame',
        ['media_id', 'frame_index'],
        postgresql_using='btree'
    )


def downgrade():
    op.drop_index('ix_frame_media_id_index', table_name='frame')
    op.drop_index('ix_media_processing_status', table_name='media')
    op.drop_index('ix_detection_frame_id', table_name='detection')
    op.drop_index('ix_media_user_created', table_name='media')