# Celery Configuration for Polysignal BTC

# Broker
CELERY_BROKER_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')

# Task settings
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TIMEZONE = 'UTC'
CELERY_ENABLE_UTC = True

# Task routing
CELERY_TASK_ROUTES = {
    'app.tasks.train_ml.*': 'ml',
    'app.tasks.backfill.*': 'data',
    'app.tasks.bot_worker.*': 'bot',
}

# Beat schedule (if using celery beat)
CELERY_BEAT_SCHEDULE = {
    'train-ml-daily': {
        'task': 'app.tasks.train_ml.train_model',
        'schedule': 86400.0,  # Daily
    },
    'refresh-data': {
        'task': 'app.tasks.backfill.refresh_markets',
        'schedule': 300.0,  # Every 5 minutes
    },
}