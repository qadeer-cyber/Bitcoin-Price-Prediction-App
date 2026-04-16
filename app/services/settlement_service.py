import logging
import os
from datetime import datetime, timezone
from typing import Optional

from app.models.db import db, SignalLog, MarketSnapshot, ResolvedMarket
from app.services.polymarket_service import polymarket_service

logger = logging.getLogger(__name__)

# Check if Celery is available
CELERY_AVAILABLE = False
try:
    from celery import Celery
    CELERY_AVAILABLE = True
except ImportError:
    logger.info('Celery not available - using synchronous execution')


def get_celery_app():
    """Get Celery app with graceful degradation"""
    if not CELERY_AVAILABLE:
        return None
    
    try:
        celery_app = Celery('polysignal')
        redis_url = os.environ.get('REDIS_URL')
        if redis_url:
            celery_app.conf.broker_url = redis_url
            celery_app.conf.result_backend = redis_url
            return celery_app
    except Exception as e:
        logger.warning(f'Celery initialization failed: {e}')
    
    return None


class SettlementService:
    def __init__(self):
        self.last_check = None
        self._celery_app = get_celery_app() if CELERY_AVAILABLE else None
    
    def check_pending_markets(self) -> int:
        pending_signals = SignalLog.query.filter(
            SignalLog.resolved_outcome.is_(None)
        ).all()
        
        if not pending_signals:
            return 0
        
        resolved_count = 0
        
        for signal in pending_signals:
            market = MarketSnapshot.query.filter_by(
                market_id=signal.market_id
            ).order_by(MarketSnapshot.snapshot_time.desc()).first()
            
            if not market:
                continue
            
            if market.status == 'resolved':
                outcome = self._get_outcome(market.market_id)
                
                if outcome:
                    signal.resolved_outcome = outcome
                    signal.is_correct = self._evaluate_signal(signal, outcome)
                    db.session.commit()
                    resolved_count += 1
            else:
                outcome = self._get_outcome(market.market_id)
                
                if outcome:
                    market.status = 'resolved'
                    
                    resolved = ResolvedMarket(
                        market_id=market.market_id,
                        event_title=market.event_title,
                        window_start=market.window_start,
                        window_end=market.window_end,
                        price_to_beat=market.price_to_beat,
                        final_price=market.live_price,
                        outcome=outcome
                    )
                    db.session.add(resolved)
                    
                    signal.resolved_outcome = outcome
                    signal.is_correct = self._evaluate_signal(signal, outcome)
                    
                    db.session.commit()
                    resolved_count += 1
        
        return resolved_count
    
    def _get_outcome(self, market_id: str) -> Optional[str]:
        try:
            outcome = polymarket_service.get_resolved_outcome(market_id)
            return outcome
        except Exception as e:
            logger.error(f'Error getting outcome for {market_id}: {e}')
            return None
    
    def _evaluate_signal(self, signal: SignalLog, actual_outcome: str) -> Optional[bool]:
        if signal.signal_direction == 'NO_TRADE':
            return None
        
        predicted = signal.signal_direction
        actual = actual_outcome
        
        return predicted == actual
    
    def resolve_signal_manually(self, signal_id: str, actual_outcome: str) -> bool:
        signal = SignalLog.query.filter_by(signal_id=signal_id).first()
        
        if not signal:
            return False
        
        signal.resolved_outcome = actual_outcome
        signal.is_correct = self._evaluate_signal(signal, actual_outcome)
        
        db.session.commit()
        
        return True
    
    def get_pending_resolutions(self) -> int:
        return SignalLog.query.filter(
            SignalLog.resolved_outcome.is_(None)
        ).count()
    
    def run_async(self, task_name: str, *args, **kwargs):
        """Run task with Celery or fallback to synchronous"""
        if self._celery_app:
            try:
                task = self._celery_app.send_task(task_name, args=args, kwargs=kwargs)
                return task.id
            except Exception as e:
                logger.warning(f'Celery task failed, running synchronously: {e}')
        
        # Fallback to synchronous execution
        logger.info(f'Running {task_name} synchronously')
        return None


settlement_service = SettlementService() 