from celery import Celery
from kombu import Queue
import os
from dotenv import load_dotenv

load_dotenv()

celery_app = Celery(
    "urban_ai",
    broker=os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/1"),  # Use separate DB for results
    include=["app.services.tasks"],
)

# Configure task queues for GPU and CPU separation
celery_app.conf.task_queues = (
    Queue('gpu', routing_key='gpu.#'),
    Queue('cpu', routing_key='cpu.#'),
    Queue('default', routing_key='default.#'),
)

celery_app.conf.task_routes = {
    'tasks.process_image':      {'queue': 'gpu', 'routing_key': 'gpu.inference'},
    'tasks.process_video':      {'queue': 'gpu', 'routing_key': 'gpu.inference'},
    'tasks.process_embeddings': {'queue': 'cpu', 'routing_key': 'cpu.embeddings'},
    'tasks.cleanup_temp_files': {'queue': 'cpu', 'routing_key': 'cpu.maintenance'},
}

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Celery Beat schedule for periodic tasks
    beat_schedule={
        'cleanup-temp-files': {
            'task': 'tasks.cleanup_temp_files',
            'schedule': 604800.0,  # Run once a week (604800 seconds = 7 days)
        },
    },
    result_expires=1800,  # Reduced from 3600 to save memory
    task_track_started=True,
    task_send_sent_event=True,
    worker_send_task_events=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=20,  # Reduced for GPU workers to prevent memory leaks
    task_soft_time_limit=900,  # 15 min for videos
    task_time_limit=1200,  # 20 min hard limit
    task_default_retry_delay=60,
    task_max_retries=3,
    task_compression='gzip',  # Compress task payloads
    result_compression='gzip',  # Compress results
    task_retry_backoff=True,
    task_retry_backoff_max=600,
    task_retry_jitter=True,
    task_reject_on_worker_lost=True,
    broker_connection_retry_on_startup=True,
    broker_heartbeat=30,
    broker_pool_limit=10,
    broker_transport_options={
        'visibility_timeout': 3600,  # 1 hour visibility timeout
        'confirm_publish': True,
        'max_retries': 3,
    },
    result_backend_transport_options={
        'retry_on_timeout': True,
        'socket_keepalive': True,
    },
    worker_hijack_root_logger=False,
    worker_log_format="[%(asctime)s: %(levelname)s/%(processName)s] %(message)s",
    worker_task_log_format="[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s",
    worker_pool_restarts=True,
    worker_max_memory_per_child=8000000,
)