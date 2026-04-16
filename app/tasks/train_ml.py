"""ML training background task"""

from celery import Celery
import logging

logger = logging.getLogger(__name__)

celery_app = Celery('polysignal')
celery_app.config_from_object('celeryconfig')


@celery_app.task(name='app.tasks.train_ml.train_model')
def train_ml_model(min_samples: int = 100):
    """Train ML model in background"""
    try:
        from app.services.ml_service import ml_prediction_engine
        
        result = ml_prediction_engine.train_model(min_samples=min_samples)
        
        logger.info(f'ML training completed: {result}')
        
        return result
    
    except Exception as e:
        logger.error(f'ML training failed: {e}')
        return {'error': str(e)}


@celery_app.task(name='app.tasks.train_ml.schedule_training')
def schedule_training():
    """Schedule periodic ML training"""
    from datetime import datetime, timedelta
    
    return {
        'status': 'scheduled',
        'next_run': (datetime.now() + timedelta(days=1)).isoformat()
    }