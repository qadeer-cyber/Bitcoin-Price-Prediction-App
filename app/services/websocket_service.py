import logging
import json
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Set

from flask import Flask

logger = logging.getLogger(__name__)


class RealtimeDashboard:
    """Real-time dashboard updates using SSE (Server-Sent Events) for browser compatibility"""
    
    def __init__(self):
        self._subscribers: Dict[str, Set[Callable]] = defaultdict(set)
        self._polling_thread = None
        self._running = False
        self._update_interval = 5
        self._last_market_data = {}
        self._last_signal = {}
        self._last_price = {}
    
    def subscribe(self, channel: str, callback: Callable) -> None:
        """Subscribe to a channel for real-time updates
        
        Channels:
        - market: Current market data
        - signal: Signal updates
        - price: Price updates
        - all: All updates
        """
        self._subscribers[channel].add(callback)
        logger.info(f'Subscribed to {channel}')
    
    def unsubscribe(self, channel: str, callback: Callable) -> None:
        """Unsubscribe from a channel"""
        if callback in self._subscribers[channel]:
            self._subscribers[channel].remove(callback)
            logger.info(f'Unsubscribed from {channel}')
    
    def publish(self, channel: str, data: Dict) -> None:
        """Publish data to a channel"""
        for callback in self._subscribers[channel]:
            try:
                callback(data)
            except Exception as e:
                logger.warning(f'Publish error: {e}')
        
        for callback in self._subscribers['all']:
            try:
                callback({channel: data})
            except Exception as e:
                logger.warning(f'Publish to all error: {e}')
    
    def start_polling(self, interval: int = 5) -> None:
        """Start polling for updates"""
        if self._running:
            return
        
        self._running = True
        self._update_interval = interval
        self._polling_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._polling_thread.start()
        
        logger.info(f'Real-time polling started (interval: {interval}s)')
    
    def stop_polling(self) -> None:
        """Stop polling"""
        self._running = False
        if self._polling_thread:
            self._polling_thread.join(timeout=5)
        logger.info('Real-time polling stopped')
    
    def _poll_loop(self) -> None:
        """Background polling loop"""
        while self._running:
            try:
                self._fetch_updates()
            except Exception as e:
                logger.warning(f'Polling error: {e}')
            
            time.sleep(self._update_interval)
    
    def _fetch_updates(self) -> None:
        """Fetch latest updates from services"""
        from app.services.polymarket_service import polymarket_service
        from app.services.chainlink_service import binance_service
        from app.services.strategy_service import strategy_service
        
        market_data = {}
        try:
            market = polymarket_service.get_current_btc_5min_market()
            if market:
                market_data = {
                    'market_id': market.get('id'),
                    'question': market.get('question'),
                    'condition_id': market.get('conditionId'),
                    'volume': market.get('volume'),
                    'outcomePrices': market.get('outcomePrices'),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
                if market.get('id') != self._last_market_data.get('market_id'):
                    self.publish('market', market_data)
                    self._last_market_data = market_data
        except Exception as e:
            logger.warning(f'Market fetch error: {e}')
        
        price_data = {}
        try:
            btc_price = binance_service.get_btc_usdt_price()
            if btc_price:
                price_data = {
                    'btc_price': btc_price,
                    'source': 'binance',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
                if btc_price != self._last_price.get('btc_price'):
                    self.publish('price', price_data)
                    self._last_price = price_data
        except Exception as e:
            logger.warning(f'Price fetch error: {e}')
        
        signal_data = {}
        try:
            from app.services.strategy_service import strategy_service
            signal = strategy_service.generate_signal()
            if signal:
                signal_data = {
                    'signal': signal,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
                if signal.get('signal') != self._last_signal.get('signal'):
                    self.publish('signal', signal_data)
                    self._last_signal = signal_data
        except Exception as e:
            logger.warning(f'Signal fetch error: {e}')
    
    def get_status(self) -> Dict:
        """Get current status"""
        return {
            'running': self._running,
            'subscribers': {ch: len(subs) for ch, subs in self._subscribers.items()},
            'interval': self._update_interval,
            'last_market': self._last_market_data.get('market_id'),
            'last_signal': self._last_signal.get('signal', {}).get('signal'),
            'last_price': self._last_price.get('btc_price')
        }


realtime_dashboard = RealtimeDashboard()


class SSEPublisher:
    """Server-Sent Events publisher for Flask"""
    
    def __init__(self):
        self._connections: Dict[str, List] = defaultdict(list)
    
    def add_connection(self, channel: str, queue) -> None:
        """Add SSE connection"""
        self._connections[channel].append(queue)
    
    def remove_connection(self, channel: str, queue) -> None:
        """Remove SSE connection"""
        if queue in self._connections[channel]:
            self._connections[channel].remove(queue)
    
    def broadcast(self, channel: str, data: Dict) -> None:
        """Broadcast to all connections on a channel"""
        message = f"data: {json.dumps(data)}\n\n"
        for queue in self._connections[channel]:
            try:
                queue.put(message)
            except Exception as e:
                logger.warning(f'Broadcast error: {e}')
    
    def publish(self, channel: str, event: str, data: Dict) -> None:
        """Publish event to channel"""
        message = f"event: {event}\ndata: {json.dumps(data)}\n\n"
        for queue in self._connections[channel]:
            try:
                queue.put(message)
            except Exception as e:
                logger.warning(f'Publish error: {e}')


sse_publisher = SSEPublisher()


def get_realtime_service():
    global _realtime_service
    if _realtime_service is None:
        _realtime_service = RealtimeDashboard()
    return _realtime_service


def init_realtime_service(interval: int = 5) -> RealtimeDashboard:
    """Initialize and start the realtime service"""
    svc = get_realtime_service()
    svc.start_polling(interval)
    return svc


_realtime_service = None


sse_publisher = SSEPublisher()