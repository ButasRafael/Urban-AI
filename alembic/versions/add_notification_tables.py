"""Add notification tables

Revision ID: add_notification_tables
Revises: add_performance_indexes
Create Date: 2025-01-04 20:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'add_notification_tables'
down_revision = 'add_performance_indexes'
branch_labels = None
depends_on = None


def upgrade():
    # Create enum types
    event_type_enum = postgresql.ENUM(
        'issue_created',
        'issue_status_changed',
        'issue_assigned',
        'issue_severity_changed',
        'issue_verified',
        name='event_type_enum',
        create_type=True
    )
    event_type_enum.create(op.get_bind(), checkfirst=True)

    notification_channel_enum = postgresql.ENUM(
        'email',
        'push',
        name='notification_channel_enum',
        create_type=True
    )
    notification_channel_enum.create(op.get_bind(), checkfirst=True)

    notification_status_enum = postgresql.ENUM(
        'pending',
        'sent',
        'failed',
        name='notification_status_enum',
        create_type=True
    )
    notification_status_enum.create(op.get_bind(), checkfirst=True)

    # Create notification_preferences table
    op.create_table(
        'notification_preferences',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('username', sa.String(), nullable=False),
        sa.Column('email_enabled', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('push_enabled', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('event_preferences', postgresql.JSON(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('min_severity', sa.String(), nullable=False, server_default='low'),
        sa.Column('quiet_hours', postgresql.JSON(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('digest_enabled', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('digest_frequency', sa.String(), nullable=False, server_default='daily'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('fcm_token', sa.String(), nullable=True),
        sa.ForeignKeyConstraint(['username'], ['users.username'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_notification_preferences_username', 'notification_preferences', ['username'])

    # Create notification_templates table
    op.create_table(
        'notification_templates',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('event_type', event_type_enum, nullable=False),
        sa.Column('channel', notification_channel_enum, nullable=False),
        sa.Column('subject', sa.String(), nullable=True),
        sa.Column('body', sa.Text(), nullable=False),
        sa.Column('variables', postgresql.JSON(astext_type=sa.Text()), nullable=False, server_default='[]'),
        sa.Column('language', sa.String(), nullable=False, server_default='en'),
        sa.Column('version', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_notification_templates_event_channel', 'notification_templates', ['event_type', 'channel', 'is_active'])

    # Create notification_log table
    op.create_table(
        'notification_log',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('username', sa.String(), nullable=False),
        sa.Column('event_type', event_type_enum, nullable=False),
        sa.Column('channel', notification_channel_enum, nullable=False),
        sa.Column('status', notification_status_enum, nullable=False, server_default='pending'),
        sa.Column('event_data', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('sent_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('next_retry_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.ForeignKeyConstraint(['username'], ['users.username'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_notification_log_username_created', 'notification_log', ['username', 'created_at'])
    op.create_index('ix_notification_log_status', 'notification_log', ['status'])


def downgrade():
    # Drop tables
    op.drop_index('ix_notification_log_status', table_name='notification_log')
    op.drop_index('ix_notification_log_username_created', table_name='notification_log')
    op.drop_table('notification_log')

    op.drop_index('ix_notification_templates_event_channel', table_name='notification_templates')
    op.drop_table('notification_templates')

    op.drop_index('ix_notification_preferences_username', table_name='notification_preferences')
    op.drop_table('notification_preferences')

    # Drop enum types
    sa.Enum(name='notification_status_enum').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='notification_channel_enum').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='event_type_enum').drop(op.get_bind(), checkfirst=True)
