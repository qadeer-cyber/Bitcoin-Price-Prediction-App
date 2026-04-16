"""Historical data backfill task"""

from celery import Celery
import logging

logger = logging.getLogger(__name__)

celery_app = Celery('polysignal')
celery_app.config_from_object('celeryconfig')


@celery_app.task(name='app.tasks.backfill.refresh_markets')
def refresh_markets():
    """Refresh market data in background"""
    try:
        from app.services.polymarket_service import polymarket_service
        
        market = polymarket_service.get_current_btc_5min_market()
        
        return {
            'status': 'completed',
            'market_id': market.get('id') if market else None
        }
    
    except Exception as e:
        logger.error(f'Market refresh failed: {e}')
        return {'error': str(e)}


@celery_app.task(name='app.tasks.backfill.backfill_history')
def backfill_history(days: int = 7):
    """Backfill historical data"""
    try:
        from app.services.polymarket_service import polymarket_service
        
        markets = polymarket_service.get_historical_resolved_markets(days=days)
        
        return {
            'status': 'completed',
            'count': len(markets)
        }
    
    except Exception as e:
        logger.error(f'Backfill failed: {e}')
        return {'error': str(e)}