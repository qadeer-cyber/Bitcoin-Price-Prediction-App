# APScheduler configuration for low-RAM environment
import os

SCHEDULER_CONFIG = {
    'apscheduler.jobstores.default': {
        'class': 'apscheduler.jobstores.memory.JobStore'
    },
    'apscheduler.executors.default': {
        'class': 'apscheduler.executors.threadpool.ThreadPoolExecutor',
        'max_workers': 2
    },
    'apscheduler.job_defaults.max_instances': 1,
    'apscheduler.job_defaults.coalesce': True,
    'apscheduler.job_defaults.max_age': 3600
}

# Check available memory before running tasks
def check_memory_guardrail(threshold_percent=85):
    """Check memory usage and return False if above threshold"""
    try:
        import psutil
        memory_percent = psutil.virtual_memory().percent
        return memory_percent < threshold_percent
    except:
        return True

# Lightweight scheduler (replaces Celery for background tasks)
def schedule_lightweight_task(func, trigger='interval', minutes=60):
    """Decorator for lightweight background tasks"""
    def decorator(f):
        f._lightweight_task = True
        f._trigger = trigger
        f._interval_minutes = minutes
        return f
    return decorator