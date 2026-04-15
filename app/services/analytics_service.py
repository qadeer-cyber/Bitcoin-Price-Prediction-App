import logging
from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta
from sqlalchemy import func, and_, or_

from app.models.db import db, SignalLog, Settings

logger = logging.getLogger(__name__)


class AnalyticsService:
    def get_today_stats(self) -> Dict:
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        
        signals = SignalLog.query.filter(
            SignalLog.timestamp >= today_start
        ).all()
        
        total_signals = len(signals)
        up_signals = sum(1 for s in signals if s.signal_direction == 'UP')
        down_signals = sum(1 for s in signals if s.signal_direction == 'DOWN')
        no_trade_signals = sum(1 for s in signals if s.signal_direction == 'NO_TRADE')
        
        resolved = [s for s in signals if s.resolved_outcome is not None]
        pending = [s for s in signals if s.resolved_outcome is None]
        
        correct = sum(1 for s in resolved if s.is_correct == True)
        wrong = sum(1 for s in resolved if s.is_correct == False)
        skipped = sum(1 for s in resolved if s.is_correct is None)
        
        win_rate = (correct / len(resolved) * 100) if len(resolved) > 0 else 0
        actionable_win_rate = (correct / (correct + wrong) * 100) if (correct + wrong) > 0 else 0
        
        confidences = [s.confidence for s in signals if s.confidence > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        median_confidence = sorted(confidences)[len(confidences)//2] if confidences else 0
        max_confidence = max(confidences) if confidences else 0
        min_confidence = min(confidences) if confidences else 0
        
        streak = self._calculate_current_streak(signals)
        max_win_streak = self._calculate_max_streak(signals, True)
        max_loss_streak = self._calculate_max_streak(signals, False)
        
        return {
            'total_signals': total_signals,
            'up_signals': up_signals,
            'down_signals': down_signals,
            'no_trade_signals': no_trade_signals,
            'resolved_count': len(resolved),
            'pending_count': len(pending),
            'correct_count': correct,
            'wrong_count': wrong,
            'skipped_count': skipped,
            'win_rate': win_rate,
            'actionable_win_rate': actionable_win_rate,
            'avg_confidence': avg_confidence,
            'median_confidence': median_confidence,
            'max_confidence': max_confidence,
            'min_confidence': min_confidence,
            'current_streak': streak,
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak
        }
    
    def _calculate_current_streak(self, signals: List) -> int:
        if not signals:
            return 0
        
        sorted_signals = sorted(signals, key=lambda x: x.timestamp, reverse=True)
        
        if not sorted_signals[0].resolved_outcome:
            return 0
        
        first_outcome = sorted_signals[0].is_correct
        if first_outcome is None:
            return 0
        
        streak = 0
        for signal in sorted_signals:
            if signal.resolved_outcome is None:
                break
            if signal.is_correct == first_outcome:
                streak += 1
            else:
                break
        
        return streak if first_outcome else -streak
    
    def _calculate_max_streak(self, signals: List, win: bool) -> int:
        if not signals:
            return 0
        
        resolved = [s for s in signals if s.resolved_outcome is not None and s.is_correct is not None]
        if not resolved:
            return 0
        
        max_streak = 0
        current_streak = 0
        
        for signal in resolved:
            if signal.is_correct == win:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def get_accuracy_by_confidence_bucket(self) -> Dict:
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        
        signals = SignalLog.query.filter(
            and_(
                SignalLog.timestamp >= today_start,
                SignalLog.resolved_outcome.isnot(None),
                SignalLog.is_correct.isnot(None)
            )
        ).all()
        
        buckets = {
            '90+': {'correct': 0, 'total': 0},
            '80-89': {'correct': 0, 'total': 0},
            '70-79': {'correct': 0, 'total': 0},
            '60-69': {'correct': 0, 'total': 0},
            '50-59': {'correct': 0, 'total': 0}
        }
        
        for signal in signals:
            conf = signal.confidence
            if conf >= 90:
                bucket = '90+'
            elif conf >= 80:
                bucket = '80-89'
            elif conf >= 70:
                bucket = '70-79'
            elif conf >= 60:
                bucket = '60-69'
            else:
                bucket = '50-59'
            
            buckets[bucket]['total'] += 1
            if signal.is_correct:
                buckets[bucket]['correct'] += 1
        
        result = {}
        for bucket, data in buckets.items():
            accuracy = (data['correct'] / data['total'] * 100) if data['total'] > 0 else 0
            result[bucket] = {
                'accuracy': accuracy,
                'correct': data['correct'],
                'total': data['total']
            }
        
        return result
    
    def get_accuracy_by_hour(self) -> Dict:
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        
        signals = SignalLog.query.filter(
            and_(
                SignalLog.timestamp >= today_start,
                SignalLog.resolved_outcome.isnot(None),
                SignalLog.is_correct.isnot(None)
            )
        ).all()
        
        hour_stats = {}
        for hour in range(24):
            hour_stats[hour] = {'correct': 0, 'total': 0}
        
        for signal in signals:
            hour = signal.timestamp.hour
            hour_stats[hour]['total'] += 1
            if signal.is_correct:
                hour_stats[hour]['correct'] += 1
        
        result = {}
        for hour, data in hour_stats.items():
            accuracy = (data['correct'] / data['total'] * 100) if data['total'] > 0 else 0
            result[hour] = {
                'accuracy': accuracy,
                'correct': data['correct'],
                'total': data['total']
            }
        
        return result
    
    def get_accuracy_by_regime(self) -> Dict:
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        
        signals = SignalLog.query.filter(
            and_(
                SignalLog.timestamp >= today_start,
                SignalLog.resolved_outcome.isnot(None),
                SignalLog.is_correct.isnot(None),
                SignalLog.regime.isnot(None)
            )
        ).all()
        
        regime_stats = {
            'trending': {'correct': 0, 'total': 0},
            'sideways': {'correct': 0, 'total': 0},
            'breakout': {'correct': 0, 'total': 0},
            'whipsaw': {'correct': 0, 'total': 0}
        }
        
        for signal in signals:
            regime = signal.regime
            if regime in regime_stats:
                regime_stats[regime]['total'] += 1
                if signal.is_correct:
                    regime_stats[regime]['correct'] += 1
        
        result = {}
        for regime, data in regime_stats.items():
            accuracy = (data['correct'] / data['total'] * 100) if data['total'] > 0 else 0
            result[regime] = {
                'accuracy': accuracy,
                'correct': data['correct'],
                'total': data['total']
            }
        
        return result
    
    def get_up_vs_down_performance(self) -> Dict:
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        
        signals = SignalLog.query.filter(
            and_(
                SignalLog.timestamp >= today_start,
                SignalLog.resolved_outcome.isnot(None),
                SignalLog.is_correct.isnot(None),
                SignalLog.signal_direction.in_(['UP', 'DOWN'])
            )
        ).all()
        
        up_signals = [s for s in signals if s.signal_direction == 'UP']
        down_signals = [s for s in signals if s.signal_direction == 'DOWN']
        
        up_correct = sum(1 for s in up_signals if s.is_correct)
        down_correct = sum(1 for s in down_signals if s.is_correct)
        
        return {
            'up': {
                'total': len(up_signals),
                'correct': up_correct,
                'accuracy': (up_correct / len(up_signals) * 100) if up_signals else 0
            },
            'down': {
                'total': len(down_signals),
                'correct': down_correct,
                'accuracy': (down_correct / len(down_signals) * 100) if down_signals else 0
            }
        }
    
    def get_confusion_matrix(self) -> Dict:
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        
        signals = SignalLog.query.filter(
            and_(
                SignalLog.timestamp >= today_start,
                SignalLog.resolved_outcome.isnot(None),
                SignalLog.is_correct.isnot(None),
                SignalLog.signal_direction.in_(['UP', 'DOWN'])
            )
        ).all()
        
        true_up = 0
        false_up = 0
        true_down = 0
        false_down = 0
        
        for signal in signals:
            predicted_up = signal.signal_direction == 'UP'
            actual_up = signal.resolved_outcome == 'UP'
            
            if predicted_up and actual_up:
                true_up += 1
            elif predicted_up and not actual_up:
                false_up += 1
            elif not predicted_up and not actual_up:
                true_down += 1
            elif not predicted_up and actual_up:
                false_down += 1
        
        return {
            'true_positives': true_up,
            'false_positives': false_up,
            'true_negatives': true_down,
            'false_negatives': false_down
        }
    
    def get_rolling_accuracy(self, window_size: int = 10) -> List[Dict]:
        signals = SignalLog.query.filter(
            and_(
                SignalLog.resolved_outcome.isnot(None),
                SignalLog.is_correct.isnot(None),
                SignalLog.signal_direction.in_(['UP', 'DOWN'])
            )
        ).order_by(SignalLog.timestamp.desc()).limit(window_size * 2).all()
        
        if len(signals) < window_size:
            return []
        
        result = []
        for i in range(len(signals) - window_size + 1):
            window = signals[i:i+window_size]
            correct = sum(1 for s in window if s.is_correct)
            accuracy = (correct / window_size) * 100
            result.append({
                'window_start': window[-1].timestamp.isoformat(),
                'accuracy': accuracy,
                'correct': correct,
                'total': window_size
            })
        
        return list(reversed(result))
    
    def get_simulated_pnl(self, stake: float = 1.0) -> Dict:
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        
        signals = SignalLog.query.filter(
            and_(
                SignalLog.timestamp >= today_start,
                SignalLog.resolved_outcome.isnot(None),
                SignalLog.is_correct.isnot(None),
                SignalLog.signal_direction.in_(['UP', 'DOWN'])
            )
        ).all()
        
        if not signals:
            return {
                'total_staked': 0,
                'total_won': 0,
                'total_lost': 0,
                'net_pnl': 0,
                'roi': 0,
                'profit_factor': 0
            }
        
        won = sum(1 for s in signals if s.is_correct)
        lost = sum(1 for s in signals if not s.is_correct)
        
        total_staked = len(signals) * stake
        total_won = won * stake
        total_lost = lost * stake
        
        net_pnl = total_won - total_lost
        roi = (net_pnl / total_staked * 100) if total_staked > 0 else 0
        profit_factor = (total_won / total_lost) if total_lost > 0 else 0
        
        return {
            'total_staked': total_staked,
            'total_won': total_won,
            'total_lost': total_lost,
            'net_pnl': net_pnl,
            'roi': roi,
            'profit_factor': profit_factor
        }
    
    def get_signal_history(self, limit: int = 50, 
                           direction: str = None,
                           min_confidence: int = 0,
                           correct: bool = None) -> List[Dict]:
        query = SignalLog.query
        
        if direction:
            query = query.filter(SignalLog.signal_direction == direction)
        
        if min_confidence > 0:
            query = query.filter(SignalLog.confidence >= min_confidence)
        
        if correct is not None:
            query = query.filter(SignalLog.is_correct == correct)
        
        signals = query.order_by(SignalLog.timestamp.desc()).limit(limit).all()
        
        return [s.to_dict() for s in signals]


analytics_service = AnalyticsService()