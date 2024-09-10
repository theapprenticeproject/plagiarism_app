from celery import Celery

# Use the Frappe Redis server as the broker for Celery
app = Celery('plagiarism_app', broker='redis://localhost:6379/0')

# Celery configuration (you can adjust settings as needed)
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_acks_late=True  # Acknowledge tasks after completion for fault tolerance
)
