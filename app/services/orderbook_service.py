import logging
import json
from typing import Dict, List, Optional
from datetime import datetime, timezone
from threading import Lock

from app.services.polymarket_service import polymarket_service

logger = logging.getLogger(__name__)

WHALE_THRESHOLD = 1000  # $1000+ orders considered whales


class OrderbookAnalyzer:
    """Analyze Polymarket orderbook for microstructure insights"""
    
    def __init__(self):
        self._cache = {}
        self._cache_time = None
        self._lock = Lock()
        self._whale_alerts = []
    
    def analyze_orderbook(self, token_id: str) -> Dict:
        """Get comprehensive orderbook analysis"""
        with self._lock:
            try:
                book = polymarket_service.get_orderbook(token_id)
                
                bids = book.get('bids', [])
                asks = book.get('asks', [])
                
                if not bids or not asks:
                    return self._default_analysis()
                
                wall_imbalance = self._calculate_wall_imbalance(bids, asks)
                spread_analysis = self._analyze_spread(bids, asks)
                large_orders = self._detect_large_orders(bids, asks)
                pressure_score = self._calculate_pressure_score(
                    wall_imbalance, spread_analysis, large_orders
                )
                
                result = {
                    'token_id': token_id,
                    'wall_imbalance': wall_imbalance,
                    'spread_pct': spread_analysis['spread_pct'],
                    'spread_tier': spread_analysis['tier'],
                    'large_orders': large_orders,
                    'pressure_score': pressure_score,
                    'pressure_label': self._pressure_label(pressure_score),
                    'bid_volume': sum(float(b.get('size', 0)) for b in bids[:10]),
                    'ask_volume': sum(float(a.get('size', 0)) for a in asks[:10]),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
                if large_orders:
                    self._whale_alerts.append({
                        'token_id': token_id,
                        'orders': large_orders,
                        'timestamp': result['timestamp']
                    })
                    if len(self._whale_alerts) > 20:
                        self._whale_alerts = self._whale_alerts[-20:]
                
                self._cache = result
                self._cache_time = datetime.now(timezone.utc)
                
                return result
            
            except Exception as e:
                logger.warning(f'Orderbook analysis error: {e}')
                return self._default_analysis()
    
    def _default_analysis(self) -> Dict:
        return {
            'wall_imbalance': 0,
            'spread_pct': 0,
            'spread_tier': 'unknown',
            'large_orders': [],
            'pressure_score': 0,
            'pressure_label': 'neutral',
            'bid_volume': 0,
            'ask_volume': 0,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def _calculate_wall_imbalance(self, bids: List, asks: List) -> float:
        """Calculate buy vs sell wall imbalance (-1 to +1)"""
        bid_volume = sum(float(b.get('size', 0)) for b in bids[:5])
        ask_volume = sum(float(a.get('size', 0)) for a in asks[:5])
        
        total = bid_volume + ask_volume
        if total == 0:
            return 0
        
        return (bid_volume - ask_volume) / total
    
    def _analyze_spread(self, bids: List, asks: List) -> Dict:
        """Analyze bid-ask spread"""
        try:
            best_bid = float(bids[0].get('price', 0))
            best_ask = float(asks[0].get('price', 0))
            
            if best_bid == 0:
                return {'spread_pct': 0, 'tier': 'unknown'}
            
            spread_pct = ((best_ask - best_bid) / best_bid) * 100
            
            if spread_pct < 0.5:
                tier = 'tight'
            elif spread_pct < 1.5:
                tier = 'normal'
            elif spread_pct < 3.0:
                tier = 'wide'
            else:
                tier = 'very_wide'
            
            return {'spread_pct': spread_pct, 'tier': tier}
        
        except Exception as e:
            return {'spread_pct': 0, 'tier': 'unknown'}
    
    def _detect_large_orders(self, bids: List, asks: List) -> List[Dict]:
        """Detect large orders (potential whales)"""
        large_orders = []
        
        for order in bids[:10]:
            size = float(order.get('size', 0))
            if size * float(order.get('price', 0)) >= WHALE_THRESHOLD:
                large_orders.append({
                    'side': 'bid',
                    'price': float(order.get('price', 0)),
                    'size': size,
                    'value': size * float(order.get('price', 0))
                })
        
        for order in asks[:10]:
            size = float(order.get('size', 0))
            if size * float(order.get('price', 0)) >= WHALE_THRESHOLD:
                large_orders.append({
                    'side': 'ask',
                    'price': float(order.get('price', 0)),
                    'size': size,
                    'value': size * float(order.get('price', 0))
                })
        
        return large_orders
    
    def _calculate_pressure_score(
        self, wall_imbalance: float, spread: Dict, large_orders: List
    ) -> float:
        """Calculate overall orderbook pressure (-1 to +1)"""
        score = wall_imbalance * 0.5
        
        if spread['tier'] == 'tight':
            score += 0.25
        elif spread['tier'] == 'wide':
            score -= 0.25
        elif spread['tier'] == 'very_wide':
            score -= 0.5
        
        whale_bias = 0
        for order in large_orders:
            if order['side'] == 'bid':
                whale_bias += 0.1
            else:
                whale_bias -= 0.1
        
        score += max(-0.3, min(0.3, whale_bias))
        
        return max(-1.0, min(1.0, score))
    
    def _pressure_label(self, score: float) -> str:
        """Convert score to label"""
        if score >= 0.5:
            return 'strong_buy'
        elif score >= 0.2:
            return 'moderate_buy'
        elif score >= -0.2:
            return 'neutral'
        elif score >= -0.5:
            return 'moderate_sell'
        else:
            return 'strong_sell'
    
    def get_recent_whale_alerts(self, limit: int = 10) -> List[Dict]:
        """Get recent whale alerts"""
        return self._whale_alerts[-limit:]
    
    def get_cached_analysis(self) -> Optional[Dict]:
        """Get cached analysis"""
        if self._cache and self._cache_time:
            age = (datetime.now(timezone.utc) - self._cache_time).total_seconds()
            if age < 30:
                return self._cache
        return None


class OrderbookService:
    """Main orderbook service with caching"""
    
    def __init__(self):
        self._analyzer = OrderbookAnalyzer()
    
    def get_analysis(self, market_id: str = None) -> Dict:
        """Get orderbook analysis for market"""
        if market_id:
            token_id = polymarket_service.get_token_id_for_market(market_id)
        else:
            market = polymarket_service.get_current_btc_5min_market()
            token_id = polymarket_service.get_token_id_for_market(market.get('id')) if market else None
        
        if not token_id:
            return self._analyzer._default_analysis()
        
        cached = self._analyzer.get_cached_analysis()
        if cached and cached.get('token_id') == token_id:
            return cached
        
        return self._analyzer.analyze_orderbook(token_id)
    
    def get_pressure_gauge(self, market_id: str = None) -> Dict:
        """Get pressure gauge for UI"""
        analysis = self.get_analysis(market_id)
        
        return {
            'score': analysis.get('pressure_score', 0),
            'label': analysis.get('pressure_label', 'neutral'),
            'wall_imbalance': analysis.get('wall_imbalance', 0),
            'spread': analysis.get('spread_tier', 'unknown'),
            'whales': len(analysis.get('large_orders', []))
        }


orderbook_service = OrderbookService()