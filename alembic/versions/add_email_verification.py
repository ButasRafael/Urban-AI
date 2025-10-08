"""Add email verification fields to users

Revision ID: add_email_verification
Revises: add_notification_tables
Create Date: 2025-01-06 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'add_email_verification'
down_revision = 'add_notification_tables'
branch_labels = None
depends_on = None


def upgrade():
    # Add email column
    op.add_column('users', sa.Column('email', sa.String(), nullable=True))

    # Add email_verified column
    op.add_column('users', sa.Column('email_verified', sa.Boolean(), nullable=False, server_default='false'))

    # Add email_verification_token column
    op.add_column('users', sa.Column('email_verification_token', sa.String(), nullable=True))

    # Add email_verification_expires column
    op.add_column('users', sa.Column('email_verification_expires', sa.DateTime(timezone=True), nullable=True))

    # Add password_reset_token column
    op.add_column('users', sa.Column('password_reset_token', sa.String(), nullable=True))

    # Add password_reset_expires column
    op.add_column('users', sa.Column('password_reset_expires', sa.DateTime(timezone=True), nullable=True))

    # Create unique index on email
    op.create_index('ix_users_email', 'users', ['email'], unique=True)

    # Update existing users with placeholder emails (username@placeholder.local)
    # This allows the migration to succeed while requiring new users to have valid emails
    op.execute("""
        UPDATE users
        SET email = username || '@placeholder.local'
        WHERE email IS NULL
    """)

    # Now make email NOT NULL
    op.alter_column('users', 'email', nullable=False)


def downgrade():
    # Remove index
    op.drop_index('ix_users_email', table_name='users')

    # Remove columns
    op.drop_column('users', 'password_reset_expires')
    op.drop_column('users', 'password_reset_token')
    op.drop_column('users', 'email_verification_expires')
    op.drop_column('users', 'email_verification_token')
    op.drop_column('users', 'email_verified')
    op.drop_column('users', 'email')