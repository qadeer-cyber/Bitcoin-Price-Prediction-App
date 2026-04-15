import requests
import re
import logging
import json
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Callable
from threading import Thread
from queue import Queue, Empty
import os

logger = logging.getLogger(__name__)

class PolymarketService:
    GAMMA_API = 'https://gamma-api.polymarket.com'
    CLOB_API = 'https://clob.polymarket.com'
    WS_URL = 'wss://ws-subscriptions-clob.polymarket.com'
    
    def __init__(self, api_key: Optional[str] = None):
        self.session = requests.Session()
        self.session.headers.update({'Accept': 'application/json'})
        
        self._api_key = api_key
        if api_key:
            self.session.headers.update({'x-api-key': api_key})
        
        self._ws_enabled = False
        self._ws_thread = None
        self._ws_client = None
        self._orderbook_callbacks = []
        self._last_orderbook = {}
        self._price_cache = {}
        self._chainlink_fallbacks = [
            self._get_polymarket_resolution_price,
            self._get_binance_ws_price,
            self._get_coingecko_price,
        ]
    
    def _get_polymarket_resolution_price(self) -> Optional[float]:
        """Get price from Polymarket Gamma API - Primary source"""
        try:
            market = self.get_current_btc_5min_market()
            if not market:
                return None
            
            resolution_price = market.get('resolutionPrice')
            if resolution_price:
                return float(resolution_price)
            
            outcome_prices = market.get('outcomePrices', [])
            if isinstance(outcome_prices, str):
                try:
                    outcome_prices = json.loads(outcome_prices)
                except:
                    outcome_prices = []
            
            if outcome_prices:
                price = float(outcome_prices[0])
                return price * 100
                
        except Exception as e:
            logger.warning(f'Polymarket resolution price unavailable: {e}')
        return None
    
    def _get_binance_ws_price(self) -> Optional[float]:
        """Get price from Binance WebSocket - Secondary source"""
        try:
            url = 'https://api.binance.com/api/v3/ticker/price'
            params = {'symbol': 'BTCUSDT'}
            response = self.session.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            return float(data.get('price', 0))
        except Exception as e:
            logger.warning(f'Binance price unavailable: {e}')
            return None
    
    def _get_coingecko_price(self) -> Optional[float]:
        """Get price from CoinGecko API - Tertiary source (rate-limited)"""
        try:
            url = 'https://api.coingecko.com/api/v3/simple/price'
            params = {
                'ids': 'bitcoin',
                'vs_currencies': 'usd'
            }
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return float(data.get('bitcoin', {}).get('usd', 0))
        except Exception as e:
            logger.warning(f'CoinGecko price unavailable: {e}')
            return None
    
    def get_btc_price_with_fallback(self) -> Optional[float]:
        """Primary price getter with fallback chain"""
        for fallback_func in self._chainlink_fallbacks:
            price = fallback_func()
            if price and price > 0:
                return price
        return None
    
    def validate_market_price(self, market_data: Dict, current_price: float) -> bool:
        """Validate that Price to Beat matches event metadata"""
        try:
            rules = market_data.get('description', '') or market_data.get('rules', '')
            
            price_pattern = r'\$([0-9,]+(?:\.[0-9]+)?)'
            match = re.search(price_pattern, rules)
            if match:
                reference_price = float(match.group(1).replace(',', ''))
                tolerance = reference_price * 0.01
                return abs(current_price - reference_price) <= tolerance
            
            return True
        except Exception as e:
            logger.warning(f'Market validation failed: {e}')
            return True
    
    def search_btc_5min_markets(self, limit: int = 10) -> List[Dict]:
        try:
            url = f'{self.GAMMA_API}/markets'
            params = {
                'limit': limit,
                'closed': 'false',
                'active': 'true'
            }
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            markets = response.json()
            
            btc_markets = []
            for market in markets:
                question = market.get('question', '')
                if 'btc' in question.lower() and 'up or down' in question.lower() and '5 minute' in question.lower():
                    btc_markets.append(market)
            
            return btc_markets
        except Exception as e:
            logger.error(f'Error searching BTC 5min markets: {e}')
            return []
    
    def get_current_btc_5min_market(self) -> Optional[Dict]:
        markets = self.search_btc_5min_markets(limit=20)
        
        now = datetime.now(timezone.utc)
        
        for market in markets:
            start_time, end_time = self._parse_window_times(market.get('question', ''))
            
            if start_time and end_time:
                if start_time <= now < end_time:
                    return market
                elif now < start_time:
                    continue
        
        return markets[0] if markets else None
    
    def get_market_details(self, market_id: str) -> Optional[Dict]:
        try:
            url = f'{self.GAMMA_API}/markets/{market_id}'
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f'Error getting market details for {market_id}: {e}')
            return None
    
    def get_market_prices(self, token_ids: List[str]) -> Dict[str, float]:
        if not token_ids:
            return {}
        
        try:
            url = f'{self.CLOB_API}/prices'
            params = {'token': token_ids[0]}
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            prices = {}
            for item in data:
                token = item.get('token_id')
                price = item.get('price')
                if token and price:
                    prices[token] = float(price)
            
            return prices
        except Exception as e:
            logger.error(f'Error getting market prices: {e}')
            return {}
    
    def get_orderbook(self, token_id: str) -> Dict:
        try:
            url = f'{self.CLOB_API}/book'
            params = {'token_id': token_id}
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f'Error getting orderbook for {token_id}: {e}')
            return {'bids': [], 'asks': []}
    
    def get_orderbook_spread(self, token_id: str) -> Optional[float]:
        """Calculate bid-ask spread percentage"""
        try:
            book = self.get_orderbook(token_id)
            bids = book.get('bids', [])
            asks = book.get('asks', [])
            
            if not bids or not asks:
                return None
            
            best_bid = float(bids[0].get('price', 0))
            best_ask = float(asks[0].get('price', 0))
            
            if best_bid == 0:
                return None
            
            spread = ((best_ask - best_bid) / best_bid) * 100
            return spread
        except Exception as e:
            logger.warning(f'Error calculating spread: {e}')
            return None
    
    def get_orderbook_imbalance(self, token_id: str) -> float:
        """Calculate orderbook imbalance: positive = more buying pressure"""
        try:
            book = self.get_orderbook(token_id)
            bids = book.get('bids', [])
            asks = book.get('asks', [])
            
            bid_volume = sum(float(b.get('size', 0)) for b in bids[:5])
            ask_volume = sum(float(a.get('size', 0)) for a in asks[:5])
            
            if bid_volume + ask_volume == 0:
                return 0
            
            return ((bid_volume - ask_volume) / (bid_volume + ask_volume)) * 100
        except Exception as e:
            logger.warning(f'Error calculating imbalance: {e}')
            return 0
    
    def get_market_volume_24h(self, market_id: str) -> float:
        """Get 24h trading volume for market"""
        try:
            details = self.get_market_details(market_id)
            volume = details.get('volume', 0) or details.get('tradeVolume', 0)
            return float(volume) if volume else 0
        except Exception as e:
            logger.warning(f'Error getting volume: {e}')
            return 0
    
    def get_market_price_history(self, token_id: str, hours: int = 1) -> List[Dict]:
        try:
            url = f'{self.CLOB_API}/prices-history'
            params = {
                'token_id': token_id,
                'duration': f'{hours}h'
            }
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f'Error getting price history for {token_id}: {e}')
            return []
    
    def get_price_momentum(self, token_id: str, minutes: int = 5) -> float:
        """Calculate price momentum over N minutes"""
        try:
            history = self.get_market_price_history(token_id, hours=minutes/60 + 1)
            if len(history) < 2:
                return 0
            
            first_price = float(history[0].get('price', 0))
            last_price = float(history[-1].get('price', 0))
            
            if first_price == 0:
                return 0
            
            return ((last_price - first_price) / first_price) * 100
        except Exception as e:
            logger.warning(f'Error calculating momentum: {e}')
            return 0
    
    def _parse_window_times(self, title: str) -> tuple:
        if not title:
            return None, None
        
        try:
            pattern = r'(\w+\s+\d+,\s+\d+:\d+[AP]M)\s*-\s*(\d+:\d+[AP]M)\s*ET'
            match = re.search(pattern, title)
            
            if match:
                start_str = match.group(1)
                end_str = match.group(2)
                
                current_year = datetime.now().year
                
                full_start = f'{start_str}, {current_year}'
                start_dt = datetime.strptime(full_start, '%B %d, %Y, %I:%M%p').replace(tzinfo=timezone.utc)
                
                if 'AM' in end_str and 'PM' in start_str:
                    pass
                elif 'PM' in end_str and 'AM' in start_str:
                    start_dt = start_dt + timedelta(days=1)
                
                end_dt = start_dt + timedelta(minutes=5)
                
                return start_dt, end_dt
        except Exception as e:
            logger.warning(f'Failed to parse window times from title: {e}')
        
        now = datetime.now(timezone.utc)
        start = now.replace(second=0, microsecond=0)
        end = start + timedelta(minutes=5)
        
        return start, end
    
    def is_market_resolved(self, market_id: str) -> bool:
        try:
            market = self.get_market_details(market_id)
            if market:
                return market.get('closed', False) or market.get('resolved', False)
        except:
            pass
        return False
    
    def get_resolved_outcome(self, market_id: str) -> Optional[str]:
        try:
            market = self.get_market_details(market_id)
            if market and market.get('closed'):
                outcome = market.get('outcome', '')
                if outcome:
                    return 'UP' if outcome.lower() == 'yes' else 'DOWN'
        except:
            pass
        return None
    
    def get_token_id_for_market(self, market_id: str) -> Optional[str]:
        """Extract CLOB token ID from market data"""
        try:
            market = self.get_market_details(market_id)
            if market:
                token_ids = market.get('clobTokenIds', [])
                if token_ids:
                    return token_ids[0]
        except Exception as e:
            logger.warning(f'Error getting token ID: {e}')
        return None
    
    def should_no_trade(self, market_id: str, spread_threshold: float = 2.0, volume_threshold: float = 5000) -> tuple:
        """Check if conditions warrant NO_TRADE
        
        Returns: (should_no_trade: bool, reason: str)
        """
        try:
            token_id = self.get_token_id_for_market(market_id)
            
            if token_id:
                spread = self.get_orderbook_spread(token_id)
                if spread and spread > spread_threshold:
                    return True, f'Spread too wide: {spread:.1f}%'
                
                imbalance = abs(self.get_orderbook_imbalance(token_id))
                if imbalance > 80:
                    return True, f'Extreme orderbook imbalance: {imbalance:.0f}%'
            
            volume = self.get_market_volume_24h(market_id)
            if volume < volume_threshold:
                return True, f'Low volume: ${volume:,.0f}'
            
            return False, ''
            
        except Exception as e:
            logger.warning(f'Error checking NO_TRADE conditions: {e}')
            return False, ''
    
    def enable_websocket(self, market_id: str, callback: Callable[[Dict], None]):
        """Enable WebSocket for real-time orderbook updates"""
        try:
            import websocket
            
            self._orderbook_callbacks.append(callback)
            self._ws_enabled = True
            self._ws_market_id = market_id
            
            ws_url = f"{self.WS_URL}?market={market_id}"
            
            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    self._last_orderbook = data
                    
                    for cb in self._orderbook_callbacks:
                        try:
                            cb(data)
                        except Exception as e:
                            logger.warning(f'Orderbook callback error: {e}')
                except Exception as e:
                    logger.warning(f'WS message parse error: {e}')
            
            def on_error(ws, error):
                logger.warning(f'WS error: {error}')
                self._ws_enabled = False
            
            def on_close(ws, close_status_code, close_msg):
                logger.info(f'WS closed: {close_status_code} {close_msg}')
                self._ws_enabled = False
            
            def on_open(ws):
                logger.info('WebSocket connected')
                ws.send(json.dumps({
                    'type': 'subscribe',
                    'channel': 'orderbook',
                    'market': market_id
                }))
            
            self._ws_thread = Thread(
                target=self._run_websocket,
                args=(ws_url, on_message, on_error, on_close, on_open)
            )
            self._ws_thread.daemon = True
            self._ws_thread.start()
            
            logger.info(f'WebSocket enabled for {market_id}')
            
        except ImportError:
            logger.warning('websocket-client not installed, using REST polling')
        except Exception as e:
            logger.warning(f'WebSocket setup failed: {e}')
    
    def _run_websocket(self, url, on_message, on_error, on_close, on_open):
        try:
            import websocket
            ws = websocket.WebSocketApp(
                url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            ws.run_forever(ping_interval=30)
        except Exception as e:
            logger.warning(f'WebSocket thread error: {e}')
    
    def disable_websocket(self):
        """Disable WebSocket connection"""
        self._ws_enabled = False
        self._orderbook_callbacks.clear()
        logger.info('WebSocket disabled')
    
    def get_latest_orderbook(self) -> Dict:
        """Get latest orderbook from cache or REST"""
        if self._last_orderbook:
            return self._last_orderbook
        return {'bids': [], 'asks': []}

    def fetch_resolved_markets(self, condition_id: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict]:
        """Fetch actual resolved market data from Polymarket Gamma API
        
        Args:
            condition_id: Optional condition ID to filter markets
            start_date: Start date in ISO format (YYYY-MM-DDTHH:MM:SSZ)
            end_date: End date in ISO format (YYYY-MM-DDTHH:MM:SSZ)
            
        Returns:
            List of resolved market dicts with outcome, resolution_price, volume, timestamps
        """
        try:
            url = f'{self.GAMMA_API}/markets'
            params = {
                'resolved': 'true',
                'closed': 'true'
            }
            
            if condition_id:
                params['conditionId'] = condition_id
            if start_date:
                params['startDate'] = start_date
            if end_date:
                params['endDate'] = end_date
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            markets = response.json()
            
            btc_markets = []
            for market in markets:
                question = market.get('question', '') or market.get('description', '')
                if 'btc' in question.lower() and ('up or down' in question.lower() or '5 minute' in question.lower()):
                    resolved_market = {
                        'market_id': market.get('id'),
                        'condition_id': market.get('conditionId'),
                        'question': question,
                        'outcome': market.get('outcome'),
                        'resolution_price': market.get('resolutionPrice'),
                        'resolution_timestamp': market.get('endDate') or market.get('closedAt'),
                        'volume': market.get('volume') or market.get('tradeVolume'),
                        'created_at': market.get('createdAt'),
                        'start_date': market.get('startDate'),
                        'end_date': market.get('endDate'),
                    }
                    btc_markets.append(resolved_market)
            
            return btc_markets
            
        except Exception as e:
            logger.error(f'Error fetching resolved markets: {e}')
            return []

    def get_btc_condition_id(self) -> Optional[str]:
        """Get the condition ID for BTC 5-minute markets"""
        try:
            markets = self.search_btc_5min_markets(limit=5)
            if markets:
                return markets[0].get('conditionId')
        except Exception as e:
            logger.warning(f'Error getting condition ID: {e}')
        return None

    def get_historical_resolved_markets(self, days: int = 7) -> List[Dict]:
        """Get resolved BTC 5-minute markets for the past N days
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of resolved market data with outcomes
        """
        try:
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            
            return self.fetch_resolved_markets(
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat()
            )
        except Exception as e:
            logger.error(f'Error getting historical markets: {e}')
            return []


def get_polymarket_service(api_key: Optional[str] = None) -> PolymarketService:
    global _polymarket_service_instance
    if _polymarket_service_instance is None:
        _polymarket_service_instance = PolymarketService(api_key)
    return _polymarket_service_instance


_polymarket_service_instance = None
polymarket_service = PolymarketService()