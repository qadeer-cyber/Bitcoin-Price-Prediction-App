import logging
import json
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from app.models.db import db, SignalLog
from app.services.chainlink_service import binance_service

logger = logging.getLogger(__name__)


class MLPredictionEngine:
    """Machine Learning Prediction Engine for 5-minute BTC outcomes"""
    
    MODEL_PATH = 'models/ml_model.joblib'
    SCALER_PATH = 'models/scaler.joblib'
    METADATA_PATH = 'models/metadata.json'
    
    def __init__(self):
        self._model = None
        self._scaler = StandardScaler()
        self._is_trained = False
        self._metadata = {
            'last_trained': None,
            'accuracy': 0,
            'training_samples': 0,
            'feature_count': 0
        }
        self._load_model()
    
    def _load_model(self) -> None:
        """Load existing model if available"""
        try:
            if os.path.exists(self.MODEL_PATH):
                self._model = joblib.load(self.MODEL_PATH)
                self._is_trained = True
                logger.info('Loaded existing ML model')
            
            if os.path.exists(self.SCALER_PATH):
                self._scaler = joblib.load(self.SCALER_PATH)
            
            if os.path.exists(self.METADATA_PATH):
                with open(self.METADATA_PATH, 'r') as f:
                    self._metadata = json.load(f)
        except Exception as e:
            logger.warning(f'Failed to load model: {e}')
    
    def _save_model(self) -> None:
        """Save model to disk"""
        os.makedirs('models', exist_ok=True)
        
        try:
            joblib.dump(self._model, self.MODEL_PATH)
            joblib.dump(self._scaler, self.SCALER_PATH)
            
            with open(self.METADATA_PATH, 'w') as f:
                json.dump(self._metadata, f)
            
            logger.info('ML model saved')
        except Exception as e:
            logger.warning(f'Failed to save model: {e}')
    
    def _extract_features(self, signal: SignalLog, historical_prices: List[float] = None) -> np.ndarray:
        """Extract features from signal record"""
        features = []
        
        if historical_prices and len(historical_prices) >= 5:
            prices = historical_prices[-20:] if len(historical_prices) >= 20 else historical_prices
            
            features.extend([
                self._price_momentum(prices, 1),
                self._price_momentum(prices, 5),
                self._price_momentum(prices, 15),
                self._rsi(prices, 14),
                self._volatility(prices, 5),
                self._volatility(prices, 15),
                self._ma_ratio(prices, 5, 20),
                self._ma_ratio(prices, 20, 50),
            ])
        else:
            features = [0.0] * 8
        
        features.extend([
            signal.market_probability_up or 0.5,
            signal.confidence / 100.0,
            self._regime_to_numeric(signal.regime),
            signal.time_remaining_at_signal or 300,
        ])
        
        while len(features) < 15:
            features.append(0.0)
        
        return np.array(features[:15])
    
    def _price_momentum(self, prices: List[float], periods: int) -> float:
        if len(prices) < periods + 1:
            return 0.0
        return ((prices[-1] - prices[-(periods + 1)]) / prices[-(periods + 1)]) * 100
    
    def _rsi(self, prices: List[float], period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        
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
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _volatility(self, prices: List[float], periods: int) -> float:
        if len(prices) < periods:
            return 0.0
        recent = prices[-periods:]
        mean = sum(recent) / len(recent)
        variance = sum((p - mean) ** 2 for p in recent) / len(recent)
        return variance ** 0.5
    
    def _ma_ratio(self, prices: List[float], short: int, long: int) -> float:
        if len(prices) < long:
            return 1.0
        
        short_ma = sum(prices[-short:]) / short
        long_ma = sum(prices[-long:]) / long
        
        return short_ma / long_ma if long_ma > 0 else 1.0
    
    def _regime_to_numeric(self, regime: str) -> float:
        mapping = {
            'trending_up': 1.0,
            'trending_down': -1.0,
            'breakout': 0.5,
            'sideways': 0.0,
            'whipsaw': -0.5,
            'unknown': 0.0
        }
        return mapping.get(regime, 0.0)
    
    def train_model(self, min_samples: int = 100) -> Dict:
        """Train the ML model using historical signal data
        
        NO FUTURE LEAKAGE: Only use signal data where:
        - Signal timestamp < Market window end (outcome was unknown at decision time)
        - Features are calculated from data BEFORE signal timestamp
        """
        
        signals = SignalLog.query.filter(
            SignalLog.is_correct.isnot(None),
            SignalLog.resolved_outcome.isnot(None),
            SignalLog.timestamp.isnot(None)
        ).order_by(SignalLog.timestamp.asc()).limit(2000).all()
        
        if len(signals) < min_samples:
            return {
                'error': f'Insufficient training data: {len(signals)}/{min_samples}',
                'samples': len(signals)
            }
        
        X = []
        y = []
        valid_timestamps = []
        
        for signal in signals:
            signal_time = signal.timestamp
            if signal_time and signal.market_id:
                from app.services.polymarket_service import polymarket_service
                market = polymarket_service.get_market_details(signal.market_id)
                if market:
                    end_time_str = market.get('endDate')
                    if end_time_str:
                        from dateutil import parser
                        try:
                            end_time = parser.parse(end_time_str)
                            if signal_time < end_time:
                                features = self._extract_features(signal)
                                X.append(features)
                                target = 1 if signal.resolved_outcome == 'UP' else 0
                                y.append(target)
                                valid_timestamps.append(signal_time)
                        except:
                            pass
        
        if len(X) < min_samples // 2:
            return {
                'error': f'Insufficient valid samples after leak check: {len(X)}/{min_samples}',
                'samples': len(X)
            }
        
        X = np.array(X)
        y = np.array(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self._scaler = StandardScaler()
        X_train_scaled = self._scaler.fit_transform(X_train)
        X_test_scaled = self._scaler.transform(X_test)
        
        self._model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        self._model.fit(X_train_scaled, y_train)
        
        y_pred = self._model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        self._metadata = {
            'last_trained': datetime.now(timezone.utc).isoformat(),
            'accuracy': float(accuracy),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_count': X.shape[1]
        }
        
        self._save_model()
        self._is_trained = True
        
        logger.info(f'ML model trained: accuracy={accuracy:.2%}, samples={len(X_train)}')
        
        return {
            'status': 'trained',
            'accuracy': accuracy,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'last_trained': self._metadata['last_trained']
        }
    
    def predict(self, features: Dict) -> Dict:
        """Predict next 5-minute outcome"""
        
        if not self._is_trained or self._model is None:
            return {
                'probability': 0.5,
                'direction': 'UNKNOWN',
                'confidence': 0,
                'error': 'Model not trained'
            }
        
        try:
            feature_array = self._extract_features_from_dict(features)
            feature_scaled = self._scaler.transform(feature_array.reshape(1, -1))
            
            proba = self._model.predict_proba(feature_scaled)[0]
            up_probability = float(proba[1])
            
            direction = 'UP' if up_probability > 0.5 else 'DOWN'
            confidence = int(abs(up_probability - 0.5) * 200)
            
            return {
                'probability': up_probability,
                'direction': direction,
                'confidence': confidence,
                'model_accuracy': self._metadata.get('accuracy', 0)
            }
        
        except Exception as e:
            logger.warning(f'Prediction error: {e}')
            return {
                'probability': 0.5,
                'direction': 'UNKNOWN',
                'confidence': 0,
                'error': str(e)
            }
    
    def _extract_features_from_dict(self, features: Dict) -> np.ndarray:
        """Convert feature dict to array"""
        arr = [
            features.get('price_change_1m', 0),
            features.get('price_change_5m', 0),
            features.get('price_change_15m', 0),
            features.get('rsi_14', 50),
            features.get('volatility_5m', 0),
            features.get('volatility_15m', 0),
            features.get('ma_ratio_5_20', 1.0),
            features.get('ma_ratio_20_50', 1.0),
            features.get('market_probability_up', 0.5),
            features.get('signal_confidence', 50) / 100.0,
            self._regime_to_numeric(features.get('regime', 'unknown')),
            features.get('time_remaining', 300),
            features.get('momentum_5', 0),
            features.get('momentum_15', 0),
            features.get('trend_strength', 50) / 100.0,
        ]
        return np.array(arr)
    
    def get_status(self) -> Dict:
        """Get model status"""
        return {
            'is_trained': self._is_trained,
            'last_trained': self._metadata.get('last_trained'),
            'accuracy': self._metadata.get('accuracy', 0),
            'training_samples': self._metadata.get('training_samples', 0),
            'feature_count': self._metadata.get('feature_count', 15)
        }


class MLFeatureEngine:
    """Feature engineering for ML models (legacy compatibility)"""
    
    def __init__(self):
        self._price_history = deque(maxlen=1000)
        self._signal_history = deque(maxlen=500)
    
    def add_price(self, price: float, timestamp: Optional[datetime] = None) -> None:
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        self._price_history.append({
            'price': price,
            'timestamp': timestamp
        })
    
    def add_signal(self, signal: Dict) -> None:
        self._signal_history.append({
            **signal,
            'timestamp': datetime.now(timezone.utc)
        })
    
    def get_features(self) -> Dict:
        if len(self._price_history) < 20:
            return self._default_features()
        
        prices = [p['price'] for p in self._price_history]
        
        return {
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
            'trend_strength': self._trend_strength(prices),
            'market_probability_up': 0.5,
            'signal_confidence': 50,
            'regime': 'unknown',
            'time_remaining': 300,
        }
    
    def _default_features(self) -> Dict:
        return {
            'price': 0, 'price_change_1m': 0, 'price_change_5m': 0,
            'price_change_15m': 0, 'volatility_5m': 0, 'volatility_15m': 0,
            'rsi_14': 50, 'ma_ratio_5_20': 1.0, 'ma_ratio_20_50': 1.0,
            'momentum_5': 0, 'momentum_15': 0, 'trend_strength': 0,
            'market_probability_up': 0.5, 'signal_confidence': 50,
            'regime': 'unknown', 'time_remaining': 300,
        }
    
    def _price_change(self, prices: List[float], periods: int) -> float:
        if len(prices) < periods + 1:
            return 0
        return ((prices[-1] - prices[-(periods + 1)]) / prices[-(periods + 1)]) * 100
    
    def _volatility(self, prices: List[float], periods: int) -> float:
        if len(prices) < periods:
            return 0
        recent = prices[-periods:]
        mean = sum(recent) / len(recent)
        return (sum((p - mean) ** 2 for p in recent) / len(recent)) ** 0.5
    
    def _rsi(self, prices: List[float], period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50
        gains = [max(0, prices[i] - prices[i-1]) for i in range(len(prices)-period, len(prices))]
        losses = [max(0, prices[i-1] - prices[i]) for i in range(len(prices)-period, len(prices))]
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss == 0:
            return 100
        return 100 - (100 / (1 + avg_gain / avg_loss))
    
    def _ma_ratio(self, prices: List[float], short: int, long: int) -> float:
        if len(prices) < long:
            return 1.0
        return (sum(prices[-short:]) / short) / (sum(prices[-long:]) / long)
    
    def _momentum(self, prices: List[float], periods: int) -> float:
        if len(prices) < periods + 1:
            return 0
        return prices[-1] - prices[-(periods + 1)]
    
    def _trend_strength(self, prices: List[float]) -> float:
        if len(prices) < 20:
            return 0
        up_days = sum(1 for i in range(1, 20) if prices[-i] > prices[-i-1])
        return (up_days / 19) * 100


class AnalyticsEngine:
    """Advanced analytics (from V3)"""
    
    def get_market_regime_analysis(self, days: int = 7) -> Dict:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        signals = SignalLog.query.filter(SignalLog.timestamp >= since).all()
        
        if not signals:
            return {'error': 'No signals found'}
        
        regimes = {}
        for signal in signals:
            regime = signal.regime or 'unknown'
            if regime not in regimes:
                regimes[regime] = {'correct': 0, 'total': 0}
            regimes[regime]['total'] += 1
            if signal.is_correct:
                regimes[regime]['correct'] += 1
        
        for regime, data in regimes.items():
            data['accuracy'] = (data['correct'] / data['total'] * 100) if data['total'] > 0 else 0
        
        total = sum(d['total'] for d in regimes.values())
        correct = sum(d['correct'] for d in regimes.values())
        
        return {
            'days': days,
            'total_signals': total,
            'overall_accuracy': (correct / total * 100) if total > 0 else 0,
            'by_regime': regimes
        }
    
    def get_confidence_analysis(self, days: int = 30) -> Dict:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        signals = SignalLog.query.filter(SignalLog.timestamp >= since).all()
        
        tiers = {'elite': {}, 'strong': {}, 'moderate': {}, 'weak': {}}
        for signal in signals:
            tier = signal.confidence_tier or 'unknown'
            if tier in tiers:
                tiers[tier]['total'] = tiers[tier].get('total', 0) + 1
                if signal.is_correct:
                    tiers[tier]['correct'] = tiers[tier].get('correct', 0) + 1
        
        for tier, data in tiers.items():
            data['accuracy'] = (data.get('correct', 0) / data.get('total', 1) * 100)
        
        return {'days': days, 'by_tier': tiers}
    
    def get_timing_analysis(self, days: int = 30) -> Dict:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        signals = SignalLog.query.filter(
            SignalLog.timestamp >= since,
            SignalLog.time_remaining_at_signal.isnot(None)
        ).all()
        
        buckets = {'early': {}, 'mid': {}, 'late': {}}
        for signal in signals:
            remaining = signal.time_remaining_at_signal or 300
            bucket = 'early' if remaining > 200 else 'mid' if remaining > 100 else 'late'
            buckets[bucket]['total'] = buckets[bucket].get('total', 0) + 1
            if signal.is_correct:
                buckets[bucket]['correct'] = buckets[bucket].get('correct', 0) + 1
        
        for bucket, data in buckets.items():
            data['accuracy'] = (data.get('correct', 0) / data.get('total', 1) * 100)
        
        return {'days': days, 'by_timing': buckets}
    
    def get_rolling_performance(self, window_hours: int = 24) -> Dict:
        since = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        signals = SignalLog.query.filter(SignalLog.timestamp >= since).all()
        
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


ml_prediction_engine = MLPredictionEngine()
ml_feature_engine = MLFeatureEngine()
analytics_engine = AnalyticsEngine()