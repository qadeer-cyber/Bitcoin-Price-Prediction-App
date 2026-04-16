"""Celery tasks for background processing"""

from celery import Celery
import os

# Initialize Celery
celery_app = Celery('polysignal')
celery_app.config_from_object('celeryconfig')

# Import tasks
from app.tasks import train_ml
from app.tasks import backfill


@celery_app.task(name='app.tasks.healthcheck')
def healthcheck():
    """Health check task"""
    return {'status': 'healthy'}