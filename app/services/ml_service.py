import logging
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from collections import deque

from app.models.db import db, SignalLog
from app.services.chainlink_service import binance_service

logger = logging.getLogger(__name__)


class MLFeatureEngine:
    """Feature engineering for ML models"""
    
    def __init__(self):
        self._price_history = deque(maxlen=1000)
        self._signal_history = deque(maxlen=500)
    
    def add_price(self, price: float, timestamp: Optional[datetime] = None) -> None:
        """Add price to history"""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        self._price_history.append({
            'price': price,
            'timestamp': timestamp
        })
    
    def add_signal(self, signal: Dict) -> None:
        """Add signal to history"""
        self._signal_history.append({
            **signal,
            'timestamp': datetime.now(timezone.utc)
        })
    
    def get_features(self) -> Dict:
        """Get current features for ML model"""
        if len(self._price_history) < 20:
            return self._default_features()
        
        prices = [p['price'] for p in self._price_history]
        
        features = {
            'price': prices[-1],
            'price_change_1m': self._price_change(prices, 1),
            'price_change_5m': self._price_change(prices, 5),
            'price_change_15m': self._price_change(prices, 15),
            'volatility_5m': self._volatility(prices, 5),
            'volatility_15m': self._volatility(prices, 15),
            'rsi_14': self._rsi(prices, 14),
            'ma_ratio_5_20': self._ma_ratio(prices, 5, 20),
            'ma_ratio_20_50': self._ma_ratio(prices, 20, 50),
            'momentum_5': self._momentum(prices, 5),
            'momentum_15': self._momentum(prices, 15),
            'volume_estimate': self._estimate_volume(prices),
            'trend_strength': self._trend_strength(prices),
        }
        
        if len(self._signal_history) >= 10:
            recent_signals = list(self._signal_history)[-10:]
            features['signal_accuracy_10'] = sum(
                1 for s in recent_signals if s.get('is_correct')
            ) / len(recent_signals) * 100
            features['signal_count_10'] = len(recent_signals)
        
        return features
    
    def _default_features(self) -> Dict:
        """Return default features when insufficient data"""
        return {
            'price': 0,
            'price_change_1m': 0,
            'price_change_5m': 0,
            'price_change_15m': 0,
            'volatility_5m': 0,
            'volatility_15m': 0,
            'rsi_14': 50,
            'ma_ratio_5_20': 1.0,
            'ma_ratio_20_50': 1.0,
            'momentum_5': 0,
            'momentum_15': 0,
            'volume_estimate': 0,
            'trend_strength': 0,
            'signal_accuracy_10': 50,
            'signal_count_10': 0,
        }
    
    def _price_change(self, prices: List[float], periods: int) -> float:
        """Calculate price change over N periods"""
        if len(prices) < periods + 1:
            return 0
        return ((prices[-1] - prices[-(periods + 1)]) / prices[-(periods + 1)]) * 100
    
    def _volatility(self, prices: List[float], periods: int) -> float:
        """Calculate volatility (std dev) over N periods"""
        if len(prices) < periods:
            return 0
        recent = prices[-periods:]
        mean = sum(recent) / len(recent)
        variance = sum((p - mean) ** 2 for p in recent) / len(recent)
        return variance ** 0.5
    
    def _rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50
        
        gains = []
        losses = []
        for i in range(len(prices) - period, len(prices)):
            change = prices[i] - prices[i - 1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _ma_ratio(self, prices: List[float], short: int, long: int) -> float:
        """Calculate moving average ratio"""
        if len(prices) < long:
            return 1.0
        
        short_ma = sum(prices[-short:]) / short
        long_ma = sum(prices[-long:]) / long
        
        return short_ma / long_ma if long_ma > 0 else 1.0
    
    def _momentum(self, prices: List[float], periods: int) -> float:
        """Calculate momentum"""
        if len(prices) < periods + 1:
            return 0
        return prices[-1] - prices[-(periods + 1)]
    
    def _estimate_volume(self, prices: List[float]) -> float:
        """Estimate volume from price movements"""
        if len(prices) < 5:
            return 0
        
        changes = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
        return sum(changes) / len(changes)
    
    def _trend_strength(self, prices: List[float]) -> float:
        """Calculate trend strength (0-100)"""
        if len(prices) < 20:
            return 0
        
        recent = prices[-20:]
        up_days = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i-1])
        return (up_days / (len(recent) - 1)) * 100


class AnalyticsEngine:
    """Advanced analytics and insights"""
    
    def get_market_regime_analysis(self, days: int = 7) -> Dict:
        """Analyze market regime performance"""
        since = datetime.now(timezone.utc) - timedelta(days=days)
        
        signals = SignalLog.query.filter(
            SignalLog.timestamp >= since
        ).all()
        
        if not signals:
            return {'error': 'No signals found'}
        
        regimes = {}
        outcomes = {'correct': 0, 'total': 0}
        
        for signal in signals:
            regime = signal.regime or 'unknown'
            if regime not in regimes:
                regimes[regime] = {'correct': 0, 'total': 0, 'outcomes': []}
            
            regimes[regime]['total'] += 1
            if signal.is_correct:
                regimes[regime]['correct'] += 1
            
            outcomes['total'] += 1
            if signal.is_correct:
                outcomes['correct'] += 1
        
        for regime, data in regimes.items():
            data['accuracy'] = (data['correct'] / data['total'] * 100) if data['total'] > 0 else 0
        
        return {
            'days': days,
            'total_signals': outcomes['total'],
            'overall_accuracy': (outcomes['correct'] / outcomes['total'] * 100) if outcomes['total'] > 0 else 0,
            'by_regime': regimes
        }
    
    def get_confidence_analysis(self, days: int = 30) -> Dict:
        """Analyze confidence tier accuracy"""
        since = datetime.now(timezone.utc) - timedelta(days=days)
        
        signals = SignalLog.query.filter(
            SignalLog.timestamp >= since
        ).all()
        
        tiers = {
            'elite': {'correct': 0, 'total': 0},
            'strong': {'correct': 0, 'total': 0},
            'moderate': {'correct': 0, 'total': 0},
            'weak': {'correct': 0, 'total': 0},
        }
        
        for signal in signals:
            tier = signal.confidence_tier or 'unknown'
            if tier in tiers:
                tiers[tier]['total'] += 1
                if signal.is_correct:
                    tiers[tier]['correct'] += 1
        
        for tier, data in tiers.items():
            data['accuracy'] = (data['correct'] / data['total'] * 100) if data['total'] > 0 else 0
        
        return {
            'days': days,
            'by_tier': tiers
        }
    
    def get_timing_analysis(self, days: int = 30) -> Dict:
        """Analyze optimal signal timing"""
        since = datetime.now(timezone.utc) - timedelta(days=days)
        
        signals = SignalLog.query.filter(
            SignalLog.timestamp >= since,
            SignalLog.time_remaining_at_signal.isnot(None)
        ).all()
        
        buckets = {
            '0-60': {'correct': 0, 'total': 0},
            '60-120': {'correct': 0, 'total': 0},
            '120-180': {'correct': 0, 'total': 0},
            '180-240': {'correct': 0, 'total': 0},
            '240-300': {'correct': 0, 'total': 0},
        }
        
        for signal in signals:
            remaining = signal.time_remaining_at_signal or 300
            
            if remaining <= 60:
                bucket = '0-60'
            elif remaining <= 120:
                bucket = '60-120'
            elif remaining <= 180:
                bucket = '120-180'
            elif remaining <= 240:
                bucket = '180-240'
            else:
                bucket = '240-300'
            
            buckets[bucket]['total'] += 1
            if signal.is_correct:
                buckets[bucket]['correct'] += 1
        
        for bucket, data in buckets.items():
            data['accuracy'] = (data['correct'] / data['total'] * 100) if data['total'] > 0 else 0
        
        return {
            'days': days,
            'by_timing': buckets
        }
    
    def get_rolling_performance(self, window_hours: int = 24) -> Dict:
        """Get rolling performance"""
        since = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        
        signals = SignalLog.query.filter(
            SignalLog.timestamp >= since
        ).all()
        
        if not signals:
            return {'accuracy': 0, 'signals': 0}
        
        correct = sum(1 for s in signals if s.is_correct)
        
        return {
            'window_hours': window_hours,
            'signals': len(signals),
            'accuracy': (correct / len(signals) * 100) if signals else 0,
            'correct': correct,
            'wrong': len(signals) - correct
        }


ml_feature_engine = MLFeatureEngine()
analytics_engine = AnalyticsEngine()