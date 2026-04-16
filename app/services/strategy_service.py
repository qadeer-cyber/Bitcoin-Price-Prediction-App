import logging
from typing import Optional, Dict, List
from datetime import datetime, timezone, timedelta

from app.services.chainlink_service import binance_service, chainlink_service
from app.services.polymarket_service import polymarket_service
from app.services.ml_service import ml_prediction_engine, ml_feature_engine
from app.models.db import Settings
from app.utils.math_utils import (
    calculate_ema, calculate_rsi, calculate_macd, calculate_atr,
    calculate_bollinger_bands, calculate_price_momentum, calculate_candle_strength
)
from app.utils.time_utils import get_market_window_times, get_hour_of_day

logger = logging.getLogger(__name__)

ML_BLEND_WEIGHT = 0.6
TECHNICAL_BLEND_WEIGHT = 0.4


class SignalDirection:
    UP = 'UP'
    DOWN = 'DOWN'
    NO_TRADE = 'NO_TRADE'


class ConfidenceTier:
    WEAK = 'weak'
    MODERATE = 'moderate'
    STRONG = 'strong'
    ELITE = 'elite'


class MarketRegime:
    TRENDING_UP = 'trending_up'
    TRENDING_DOWN = 'trending_down'
    SIDEWAYS = 'sideways'
    BREAKOUT = 'breakout'
    WHIPSAW = 'whipsaw'
    CONSOLIDATION = 'consolidation'


class StrategyService:
    def __init__(self):
        self._load_settings()
    
    def _load_settings(self):
        self.ema_short = Settings.get_value('ema_short', 9)
        self.ema_medium = Settings.get_value('ema_medium', 20)
        self.ema_long = Settings.get_value('ema_long', 50)
        
        self.rsi_period = Settings.get_value('rsi_period', 14)
        self.rsi_overbought = Settings.get_value('rsi_overbought', 70)
        self.rsi_oversold = Settings.get_value('rsi_oversold', 30)
        
        self.macd_fast = Settings.get_value('macd_fast', 12)
        self.macd_slow = Settings.get_value('macd_slow', 26)
        self.macd_signal = Settings.get_value('macd_signal', 9)
        
        self.atr_period = Settings.get_value('atr_period', 14)
        
        self.confidence_threshold = Settings.get_value('confidence_threshold', 50)
        self.no_trade_aggressiveness = Settings.get_value('no_trade_aggressiveness', 60)
        
        self.spread_warning = Settings.get_value('spread_warning', 5)
        self.min_liquidity = Settings.get_value('min_liquidity', 1000)
        self.final_minute_penalty = Settings.get_value('final_minute_penalty', 20)
    
    def refresh_settings(self):
        self._load_settings()
    
    def generate_signal(self, market_data: Dict) -> Dict:
        self.refresh_settings()
        
        if not market_data:
            return self._create_no_trade_signal('No market data available')
        
        window_info = get_market_window_times()
        time_remaining = window_info.get('remaining_seconds', 0)
        
        if time_remaining <= 0:
            return self._create_no_trade_signal('Market window ended')
        
        if time_remaining < 30:
            return self._create_no_trade_signal('Too close to expiry', time_remaining=time_remaining)
        
        btc_data = self._get_btc_data()
        if not btc_data or not btc_data.get('price'):
            return self._create_no_trade_signal('No BTC price available')
        
        price = btc_data['price']
        price_to_beat = market_data.get('price_to_beat')
        
        if price_to_beat is None:
            price_to_beat = price
        
        up_prob = market_data.get('up_probability', 0.5)
        
        factors = {}
        weights = {}
        reasoning = []
        
        price_distance = self._analyze_price_distance(price, price_to_beat, factors, reasoning)
        
        time_factor = self._analyze_time_remaining(time_remaining, factors, reasoning)
        
        market_prob_factor = self._analyze_market_probability(up_prob, factors, reasoning)
        
        micro_factors = self._analyze_microstructure(btc_data, factors, reasoning)
        
        regime, regime_conf = self._detect_regime(btc_data)
        
        risk_penalties = self._analyze_risk_factors(
            time_remaining, btc_data, market_data, regime, factors, reasoning
        )
        
        pm_weight = 0.40
        micro_weight = 0.35
        risk_weight = -0.25
        
        pm_score = weighted_sum(factors, {
            'price_distance': 0.15,
            'time_factor': 0.05,
            'market_prob': 0.10,
            'odds_momentum': 0.05,
            'orderbook_imbalance': 0.05
        })
        
        micro_score = weighted_sum(factors, {
            'candle_momentum': 0.08,
            'ema_alignment': 0.07,
            'rsi_level': 0.05,
            'macd_histogram': 0.05,
            'atr_context': 0.05,
            'bollinger_position': 0.05
        })
        
        risk_penalty = risk_penalties
        
        raw_confidence = (pm_score * pm_weight + micro_score * micro_weight) * 100 + risk_penalty
        raw_confidence = max(0, min(100, raw_confidence))
        
        confidence = int(raw_confidence)
        
        direction = SignalDirection.NO_TRADE
        if confidence >= self.confidence_threshold:
            if price >= price_to_beat:
                direction = SignalDirection.UP
            else:
                direction = SignalDirection.DOWN
        
        if confidence >= self.no_trade_aggressiveness:
            pass
        elif confidence < self.confidence_threshold:
            direction = SignalDirection.NO_TRADE
        
        if direction == SignalDirection.NO_TRADE:
            confidence = 0
        
        ml_prediction = self._get_ml_prediction(btc_data, market_data, regime, time_remaining)
        
        final_direction, final_confidence, ml_blend = self._blend_signals(
            direction, confidence, ml_prediction, technical_confidence=confidence
        )
        
        if final_direction == SignalDirection.NO_TRADE:
            final_confidence = 0
        
        tier = self._get_confidence_tier(final_confidence)
        
        if not reasoning:
            reason = 'Insufficient edge for confident signal'
            if final_direction == SignalDirection.NO_TRADE:
                reason = f'NO TRADE: {reason}'
            reasoning.append(reason)
        
        if ml_blend:
            reasoning.append(f"ML: {ml_prediction.get('direction')} ({ml_prediction.get('confidence')}%)")
        
        return {
            'direction': final_direction,
            'confidence': final_confidence,
            'tier': tier,
            'reasoning': reasoning,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'market_id': market_data.get('market_id', 'unknown'),
            'price_to_beat': price_to_beat,
            'live_price': price,
            'regime': regime,
            'time_remaining': time_remaining,
            'ml_prediction': ml_prediction,
            'up_probability': up_prob,
            'factors': factors
        }
    
    def _get_btc_data(self) -> Optional[Dict]:
        price = binance_service.get_btc_usdt_price()
        
        if not price:
            price = chainlink_service.get_btc_usd_price()
        
        if not price:
            return None
        
        klines = binance_service.get_btc_klines(interval='1m', limit=100)
        
        return {
            'price': price,
            'klines': klines
        }
    
    def _analyze_price_distance(self, price: float, price_to_beat: float, factors: Dict, reasoning: List) -> float:
        if price_to_beat is None or price_to_beat == 0:
            factors['price_distance'] = 0
            return 0.0
        
        distance_pct = ((price - price_to_beat) / price_to_beat) * 100
        
        score = 0
        if distance_pct > 0.1:
            score = min(100, distance_pct * 50)
            reasoning.append(f'Price up {distance_pct:.2f}% from Price to Beat')
        elif distance_pct < -0.1:
            score = max(-100, distance_pct * 50)
            reasoning.append(f'Price down {abs(distance_pct):.2f}% from Price to Beat')
        else:
            reasoning.append('Price near Price to Beat - neutral')
        
        factors['price_distance'] = score
        return score
    
    def _analyze_time_remaining(self, time_remaining: int, factors: Dict, reasoning: List) -> float:
        if time_remaining > 240:
            score = 100
            reasoning.append(f'Time remaining: {time_remaining}s - full window')
        elif time_remaining > 180:
            score = 80
            reasoning.append(f'Time remaining: {time_remaining}s - good entry window')
        elif time_remaining > 60:
            score = 50
            reasoning.append(f'Time remaining: {time_remaining}s - late entry')
        else:
            score = 20
            reasoning.append(f'Time remaining: {time_remaining}s - very late')
        
        factors['time_factor'] = score
        return score
    
    def _analyze_market_probability(self, up_prob: float, factors: Dict, reasoning: List) -> float:
        deviation = abs(up_prob - 0.5)
        
        score = deviation * 200
        
        if up_prob > 0.55:
            reasoning.append(f'Polymarket UP probability: {up_prob*100:.1f}%')
        elif up_prob < 0.45:
            reasoning.append(f'Polymarket DOWN probability: {(1-up_prob)*100:.1f}%')
        else:
            reasoning.append(f'Polymarket probabilities near 50/50 ({up_prob*100:.1f}% UP)')
        
        factors['market_prob'] = score
        return score
    
    def _analyze_microstructure(self, btc_data: Dict, factors: Dict, reasoning: List) -> float:
        klines = btc_data.get('klines', [])
        
        if len(klines) < 50:
            factors['candle_momentum'] = 0
            factors['ema_alignment'] = 0
            factors['rsi_level'] = 0
            factors['macd_histogram'] = 0
            factors['atr_context'] = 0
            factors['bollinger_position'] = 0
            return 0
        
        closes = [k['close'] for k in klines]
        highs = [k['high'] for k in klines]
        lows = [k['low'] for k in klines]
        opens = [k['open'] for k in klines]
        
        candle_score = calculate_candle_strength(opens, closes, highs, lows) * 100
        factors['candle_momentum'] = candle_score
        if candle_score > 20:
            reasoning.append(f'Bullish candle momentum: {candle_score:.1f}')
        elif candle_score < -20:
            reasoning.append(f'Bearish candle momentum: {candle_score:.1f}')
        
        ema9 = calculate_ema(closes, self.ema_short)
        ema20 = calculate_ema(closes, self.ema_medium)
        ema50 = calculate_ema(closes, self.ema_long)
        
        ema_score = 0
        if ema9 and ema20 and ema50:
            if ema9 > ema20 > ema50:
                ema_score = 80
                reasoning.append('EMA 9/20/50 bullish alignment')
            elif ema9 < ema20 < ema50:
                ema_score = -80
                reasoning.append('EMA 9/20/50 bearish alignment')
            elif ema9 > ema20:
                ema_score = 40
                reasoning.append('EMA 9 above EMA 20')
            elif ema9 < ema20:
                ema_score = -40
                reasoning.append('EMA 9 below EMA 20')
        
        factors['ema_alignment'] = ema_score
        
        rsi = calculate_rsi(closes, self.rsi_period)
        rsi_score = 0
        if rsi:
            if rsi > self.rsi_overbought:
                rsi_score = -30
                reasoning.append(f'RSI overbought: {rsi:.1f}')
            elif rsi < self.rsi_oversold:
                rsi_score = 30
                reasoning.append(f'RSI oversold: {rsi:.1f}')
            elif rsi > 55:
                rsi_score = 20
                reasoning.append(f'RSI bullish: {rsi:.1f}')
            elif rsi < 45:
                rsi_score = -20
                reasoning.append(f'RSI bearish: {rsi:.1f}')
        
        factors['rsi_level'] = rsi_score
        
        macd_line, signal_line, histogram = calculate_macd(
            closes, self.macd_fast, self.macd_slow, self.macd_signal
        )
        macd_score = 0
        if histogram is not None:
            macd_score = min(100, max(-100, histogram * 100))
            if macd_score > 5:
                reasoning.append(f'MACD histogram positive: {macd_score:.2f}')
            elif macd_score < -5:
                reasoning.append(f'MACD histogram negative: {macd_score:.2f}')
        
        factors['macd_histogram'] = macd_score
        
        atr = calculate_atr(highs, lows, closes, self.atr_period)
        atr_score = 0
        if atr and closes[-1]:
            atr_pct = (atr / closes[-1]) * 100
            if atr_pct > 1.0:
                atr_score = -20
                reasoning.append(f'High volatility - ATR: {atr_pct:.2f}%')
            elif atr_pct < 0.3:
                atr_score = -10
                reasoning.append(f'Low volatility - ATR: {atr_pct:.2f}%')
            else:
                atr_score = 20
                reasoning.append(f'Normal volatility - ATR: {atr_pct:.2f}%')
        
        factors['atr_context'] = atr_score
        
        upper, middle, lower = calculate_bollinger_bands(closes)
        bb_score = 0
        if upper and lower and middle and closes[-1]:
            position = (closes[-1] - lower) / (upper - lower) if upper != lower else 0.5
            bb_score = (position - 0.5) * 100
            if position > 0.9:
                reasoning.append('Price at upper Bollinger Band')
            elif position < 0.1:
                reasoning.append('Price at lower Bollinger Band')
        
        factors['bollinger_position'] = bb_score
        
        factors['odds_momentum'] = 0
        factors['orderbook_imbalance'] = 0
        
        total = sum([
            factors.get('candle_momentum', 0),
            factors.get('ema_alignment', 0),
            factors.get('rsi_level', 0),
            factors.get('macd_histogram', 0),
            factors.get('atr_context', 0),
            factors.get('bollinger_position', 0)
        ])
        
        return total / 6
    
    def _detect_regime(self, btc_data: Dict) -> tuple:
        klines = btc_data.get('klines', [])
        if len(klines) < 50:
            return MarketRegime.SIDEWAYS, 0
        
        closes = [k['close'] for k in klines]
        highs = [k['high'] for k in klines]
        lows = [k['low'] for k in klines]
        
        recent = closes[-20:]
        if len(recent) < 20:
            return MarketRegime.SIDEWAYS, 0
        
        changes = [abs(recent[i] - recent[i-1]) / recent[i-1] for i in range(1, len(recent))]
        avg_change = sum(changes) / len(changes)
        
        high_change = sum(1 for c in changes if c > 0.005)
        low_change = sum(1 for c in changes if c < 0.001)
        
        if high_change > 12:
            return MarketRegime.WHIPSAW, 80
        
        if len(klines) >= 14:
            adx = self._calculate_adx(highs, lows, closes, period=14)
            if adx and adx > 25:
                trend = closes[-1] - closes[-20]
                trend_pct = (trend / closes[-20]) * 100
                
                if trend_pct > 1.5:
                    return MarketRegime.TRENDING_UP, min(95, 50 + adx)
                elif trend_pct < -1.5:
                    return MarketRegime.TRENDING_DOWN, min(95, 50 + adx)
                elif abs(trend_pct) > 0.8:
                    return MarketRegime.BREAKOUT, 60 + adx
        
        if low_change > 15:
            return MarketRegime.SIDEWAYS, 70
        
        if len(klines) >= 20:
            upper, middle, lower = calculate_bollinger_bands(closes)
            if upper and lower:
                bb_width = (upper - lower) / middle
                if bb_width < 0.02:
                    return MarketRegime.CONSOLIDATION, 65
        
        trend = closes[-1] - closes[-20]
        trend_pct = (trend / closes[-20]) * 100
        
        if abs(trend_pct) > 2:
            return MarketRegime.TRENDING_UP if trend_pct > 0 else MarketRegime.TRENDING_DOWN, abs(trend_pct) * 20
        
        return MarketRegime.SIDEWAYS, 50
    
    def _calculate_adx(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
        """Calculate Average Directional Index for trend strength"""
        if len(closes) < period + 1:
            return None
        
        try:
            plus_dm = []
            minus_dm = []
            tr = []
            
            for i in range(1, len(closes)):
                high_diff = highs[i] - highs[i-1]
                low_diff = lows[i-1] - lows[i]
                
                plus_dm.append(high_diff if high_diff > low_diff else 0)
                minus_dm.append(low_diff if low_diff > high_diff else 0)
                
                tr.append(max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i-1]),
                    abs(lows[i] - closes[i-1])
                ))
            
            if len(tr) < period:
                return None
            
            period_plus_dm = sum(plus_dm[-period:]) / period
            period_minus_dm = sum(minus_dm[-period:]) / period
            period_tr = sum(tr[-period:]) / period
            
            if period_tr == 0:
                return None
            
            plus_di = (period_plus_dm / period_tr) * 100
            minus_di = (period_minus_dm / period_tr) * 100
            
            di_diff = abs(plus_di - minus_di)
            di_sum = plus_di + minus_di
            
            if di_sum == 0:
                return None
            
            dx = (di_diff / di_sum) * 100
            return dx
        except:
            return None
    
    def _analyze_risk_factors(self, time_remaining: int, btc_data: Dict, 
                                market_data: Dict, regime: str, factors: Dict, 
                                reasoning: List) -> float:
        penalty = 0
        
        # Phase 2: Late-entry penalty (15% reduction if <60 seconds)
        if time_remaining < 60:
            late_penalty = 15
            penalty -= late_penalty
            reasoning.append(f'WARNING: Late entry (<60s) - {late_penalty}% penalty')
        
        if regime == MarketRegime.WHIPSAW:
            penalty -= 20
            reasoning.append('WARNING: Whipsaw regime detected')
        
        # Phase 2: Spread/Liquidity filter - check via Polymarket
        market_id = market_data.get('market_id')
        if market_id and market_id != 'unknown':
            from app.services.polymarket_service import polymarket_service
            no_trade, reason = polymarket_service.should_no_trade(
                market_id, 
                spread_threshold=self.spread_warning,
                volume_threshold=self.min_liquidity
            )
            if no_trade:
                penalty -= 30
                reasoning.append(f'WARNING: {reason}')
        
        return penalty
    
    def _create_no_trade_signal(self, reason: str, time_remaining: int = 0) -> Dict:
        return {
            'direction': SignalDirection.NO_TRADE,
            'confidence': 0,
            'tier': ConfidenceTier.WEAK,
            'reasoning': [f'NO TRADE: {reason}'],
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'market_id': 'unknown',
            'price_to_beat': None,
            'live_price': None,
            'regime': 'unknown',
            'time_remaining': time_remaining,
            'up_probability': 0.5,
            'factors': {},
            'ml_prediction': None
        }
    
    def _get_ml_prediction(self, btc_data: Dict, market_data: Dict, regime: str, time_remaining: int) -> Dict:
        """Get ML prediction for current market
        
        GRACEFUL FALLBACK: If ML model not trained, return neutral to use rule-based only
        """
        try:
            if not ml_prediction_engine._is_trained:
                logger.info('ML model not trained, using rule-based only')
                return {'probability': 0.5, 'direction': 'UNKNOWN', 'confidence': 0, 'fallback': True}
            
            features = {
                'price': btc_data.get('price', 0),
                'price_change_1m': btc_data.get('change_1m', 0),
                'price_change_5m': btc_data.get('change_5m', 0),
                'price_change_15m': btc_data.get('change_15m', 0),
                'rsi_14': btc_data.get('rsi', 50),
                'volatility_5m': btc_data.get('atr', 0),
                'volatility_15m': btc_data.get('atr', 0),
                'ma_ratio_5_20': 1.0,
                'ma_ratio_20_50': 1.0,
                'market_probability_up': market_data.get('up_probability', 0.5),
                'signal_confidence': 50,
                'regime': regime,
                'time_remaining': time_remaining,
                'momentum_5': btc_data.get('momentum', 0),
                'momentum_15': btc_data.get('momentum', 0),
                'trend_strength': 50,
            }
            
            prediction = ml_prediction_engine.predict(features)
            prediction['fallback'] = False
            
            ml_feature_engine.add_price(btc_data.get('price', 0))
            
            return prediction
        except Exception as e:
            logger.warning(f'ML prediction failed: {e}')
            return {'probability': 0.5, 'direction': 'UNKNOWN', 'confidence': 0}
    
    def _blend_signals(self, tech_direction: str, tech_confidence: int, ml_prediction: Dict, technical_confidence: int = None) -> tuple:
        """Blend technical and ML signals"""
        if not ml_prediction or ml_prediction.get('direction') == 'UNKNOWN':
            return tech_direction, tech_confidence, False
        
        ml_direction = ml_prediction.get('direction')
        ml_confidence = ml_prediction.get('confidence', 0)
        ml_probability = ml_prediction.get('probability', 0.5)
        
        technical_confidence = technical_confidence or tech_confidence
        
        if ml_direction == tech_direction and ml_confidence >= 60:
            blend_score = ML_BLEND_WEIGHT * ml_confidence + TECHNICAL_BLEND_WEIGHT * technical_confidence
            return ml_direction, int(blend_score), True
        elif ml_direction != tech_direction and ml_confidence >= 75:
            return ml_direction, ml_confidence, True
        else:
            return tech_direction, tech_confidence, False
    
    def _get_confidence_tier(self, confidence: int) -> str:
        if confidence >= 90:
            return ConfidenceTier.ELITE
        elif confidence >= 80:
            return ConfidenceTier.STRONG
        elif confidence >= 70:
            return ConfidenceTier.MODERATE
        else:
            return ConfidenceTier.WEAK


def weighted_sum(factors: Dict, weights: Dict) -> float:
    total = 0
    total_weight = 0
    
    for key, weight in weights.items():
        if key in factors:
            total += (factors[key] / 100) * weight
            total_weight += weight
    
    return total / total_weight if total_weight > 0 else 0


def get_strategy_service():
    global _strategy_service
    if _strategy_service is None:
        _strategy_service = StrategyService()
    return _strategy_service


_strategy_service = None

def strategy_service():
    return get_strategy_service()