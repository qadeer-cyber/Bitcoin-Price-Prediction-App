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
import math

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
        
        self._fallback_market = None
        self._fallback_btc_price = None
        self._api_unreachable = False
        
        self._window_open_price = None
        self._window_start_time = None
        self._current_window_key = None
    
    def _get_binance_live_price(self) -> Optional[float]:
        """Get live BTC price from Binance REST API"""
        try:
            url = 'https://api.binance.com/api/v3/ticker/price'
            response = self.session.get(url, params={'symbol': 'BTCUSDT'}, timeout=5)
            response.raise_for_status()
            data = response.json()
            return float(data.get('price', 0))
        except Exception as e:
            logger.warning(f'Binance API error: {e}')
            return None
    
    def _extract_price_to_beat(self, market: Dict) -> Optional[float]:
        """Extract Price to Beat from market description/rules"""
        try:
            description = market.get('description', '') or market.get('rules', '') or ''
            price_pattern = r'\$([0-9,]+(?:\.[0-9]+)?)'
            match = re.search(price_pattern, description)
            if match:
                return float(match.group(1).replace(',', ''))
            
            outcome_prices = market.get('outcomePrices', [])
            if isinstance(outcome_prices, str):
                try:
                    outcome_prices = json.loads(outcome_prices)
                except:
                    outcome_prices = []
            
            if outcome_prices and len(outcome_prices) > 0:
                return float(outcome_prices[0]) * 100
        except Exception as e:
            logger.warning(f'Price to beat extraction error: {e}')
        return None
    
    def _get_market_probabilities(self, market: Dict) -> tuple:
        """Get UP/DOWN probabilities from market or CLOB"""
        try:
            outcome_prices = market.get('outcomePrices', [])
            if isinstance(outcome_prices, str):
                try:
                    outcome_prices = json.loads(outcome_prices)
                except:
                    outcome_prices = []
            
            if outcome_prices and len(outcome_prices) >= 2:
                up_prob = float(outcome_prices[0])
                down_prob = float(outcome_prices[1])
                return up_prob, down_prob
            
            token_ids = market.get('clobTokenIds', [])
            if token_ids:
                prices = self.get_market_prices(token_ids)
                if prices:
                    up_price = prices.get(token_ids[0], 0.5)
                    return up_price, 1 - up_price
        except Exception as e:
            logger.warning(f'Probability extraction error: {e}')
        
        return 0.5, 0.5
    
    def _calculate_seconds_remaining(self, market: Dict) -> int:
        """Calculate seconds remaining until market end"""
        try:
            end_date_str = market.get('endDate') or market.get('end_date_iso')
            if end_date_str:
                if 'Z' in end_date_str:
                    end_date_str = end_date_str.replace('Z', '+00:00')
                end_dt = datetime.fromisoformat(end_date_str.replace('+00:00', ''))
            else:
                end_dt = datetime.now(timezone.utc) + timedelta(minutes=5)
            
            now = datetime.now(timezone.utc)
            remaining = int((end_dt - now).total_seconds())
            return max(0, remaining)
        except Exception as e:
            logger.warning(f'Seconds remaining error: {e}')
            return 300
    
    def _generate_fallback_market(self) -> Dict:
        """Generate safe simulated market when API is unreachable"""
        if self._fallback_market:
            return self._fallback_market
        
        import time
        now = datetime.now(timezone.utc)
        
        minutes = now.minute
        window_start = now.replace(second=0, microsecond=0)
        window_end = window_start + timedelta(minutes=5)
        
        if minutes % 5 >= 0:
            window_start = now.replace(second=0, microsecond=0)
            window_end = window_start + timedelta(minutes=5)
        
        btc_price = self._fallback_btc_price or 42000.0
        
        self._fallback_market = {
            'id': 'fallback-btc-5min-' + str(int(time.time())),
            'conditionId': 'fallback-condition',
            'question': f'Will BTC be above or below ${btc_price:,.0f} at {window_end.strftime("%H:%M")} ET?',
            'description': f'Will BTC be above or below ${btc_price:,.0f} in the next 5 minutes?',
            'volume': 10000,
            'outcomePrices': ['0.50', '0.50'],
            'endDate': window_end.isoformat(),
            'startDate': window_start.isoformat(),
            'closed': False,
            'active': True
        }
        return self._fallback_market
    
    def get_live_current_market(self) -> Dict:
        """Get live current market with all data - primary method for API"""
        market = self._get_current_btc_5min_market_live()
        
        if not market:
            self._api_unreachable = True
            market = self._generate_fallback_market()
        
        btc_price = self._get_binance_live_price()
        if not btc_price:
            btc_price = self._get_binance_ws_price()
        if not btc_price:
            btc_price = self._fallback_btc_price or 42000.0
            self._fallback_btc_price = btc_price
        else:
            self._fallback_btc_price = btc_price
        
        price_to_beat = self._extract_price_to_beat(market)
        if not price_to_beat:
            price_to_beat = btc_price
        
        up_prob, down_prob = self._get_market_probabilities(market)
        
        seconds_remaining = self._calculate_seconds_remaining(market)
        
        return {
            'market_id': market.get('id') or market.get('conditionId') or 'unknown',
            'condition_id': market.get('conditionId') or 'unknown',
            'event_title': market.get('question', '') or market.get('description', ''),
            'price_to_beat': price_to_beat,
            'live_price': btc_price,
            'up_probability': up_prob,
            'down_probability': down_prob,
            'seconds_remaining': seconds_remaining,
            'window_start': market.get('startDate'),
            'window_end': market.get('endDate'),
            'volume': market.get('volume', 0),
            'status': 'live' if seconds_remaining > 0 else 'resolved',
            'data_source': 'api' if not self._api_unreachable else 'fallback',
            'is_synthetic': market.get('is_synthetic', False)
        }
    
    def _get_current_btc_5min_market_live(self) -> Optional[Dict]:
        """Get current live BTC 5-minute market from API"""
        try:
            url = f'{self.GAMMA_API}/markets'
            params = {
                'limit': 50,
                'closed': 'false',
                'active': 'true'
            }
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            markets = response.json()
            
            now = datetime.now(timezone.utc)
            btc_markets = []
            
            for market in markets:
                question = market.get('question', '') or market.get('description', '')
                question_lower = question.lower()
                
                if 'btc' in question_lower or 'bitcoin' in question_lower:
                    if ('up or down' in question_lower or 'above or below' in question_lower or '5 minute' in question_lower or '5-min' in question_lower):
                        end_date_str = market.get('endDate')
                        if end_date_str:
                            try:
                                if 'Z' in end_date_str:
                                    end_date_str = end_date_str.replace('Z', '+00:00')
                                end_dt = datetime.fromisoformat(end_date_str.replace('+00:00', ''))
                                if end_dt > now:
                                    btc_markets.append(market)
                            except:
                                btc_markets.append(market)
                        else:
                            btc_markets.append(market)
            
            if btc_markets:
                btc_markets[0]['is_synthetic'] = False
                return btc_markets[0]
            
            return self.get_synthetic_5m_window()
            
        except Exception as e:
            logger.warning(f'Error fetching live market: {e}')
            self._api_unreachable = True
            return self.get_synthetic_5m_window()
    
    def _get_polymarket_resolution_price(self) -> Optional[float]:
        """Get price from Polymarket Gamma API - Primary source"""
        try:
            market = self.get_current_btc_5min_market()
            if not market or market.get('is_synthetic'):
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
    
    def get_synthetic_5m_window(self) -> Optional[Dict]:
        """Generate synthetic 5-minute window when Polymarket markets unavailable"""
        now = datetime.now(timezone.utc)
        
        window_start = now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0)
        window_end = window_start + timedelta(minutes=5)
        window_key = window_start.replace(tzinfo=timezone.utc).timestamp()
        
        if self._current_window_key != window_key:
            self._current_window_key = window_key
            self._window_open_price = self.get_btc_price_with_fallback()
            self._window_start_time = window_start
        
        seconds_remaining = max(0, (window_end - now).total_seconds())
        
        return {
            'id': f'synthetic_{int(window_key)}',
            'question': f'Will BTC be > ${self._window_open_price:.2f} at {window_end.strftime("%H:%M")} UTC?',
            'conditionId': f'synthetic_{int(window_key)}',
            'slug': f'synthetic-btc-5m-{int(window_key)}',
            'description': f'5-minute window: {window_start.strftime("%H:%M")}-{window_end.strftime("%H:%M")} UTC. Price to beat: ${self._window_open_price:.2f}',
            'outcomePrices': json.dumps([str(min(1.0, max(0.0, (self._window_open_price or 50000) / 100000))), '0.5']),
            'volume': '0',
            'liquidity': '0',
            'active': True,
            'closed': False,
            'endDate': window_end.isoformat(),
            'startDate': window_start.isoformat(),
            'price_to_beat': self._window_open_price,
            'window_start': window_start,
            'window_end': window_end,
            'window_key': window_key,
            'seconds_remaining': seconds_remaining,
            'is_synthetic': True,
            'resolutionSource': 'Binance REST API',
            'groupItemTitle': 'Synthetic BTC 5min',
            'volume24hr': '0',
            'volume1wk': '0',
            'volume1mo': '0',
            'volumeClob': '0',
            'liquidityClob': '0',
            'lastTradePrice': 0.5,
            'bestBid': 0.5,
            'bestAsk': 0.5,
            'spread': 0.0,
            'oneDayPriceChange': 0.0,
            'oneWeekPriceChange': 0.0,
            'oneMonthPriceChange': 0.0
        }
    
    def get_current_btc_5min_market(self) -> Optional[Dict]:
        markets = self.search_btc_5min_markets(limit=20)
        
        now = datetime.now(timezone.utc)
        
        for market in markets:
            start_time, end_time = self._parse_window_times(market.get('question', ''))
            
            if start_time and end_time:
                if start_time <= now < end_time:
                    market['is_synthetic'] = False
                    return market
                elif now < start_time:
                    continue
        
        if markets:
            markets[0]['is_synthetic'] = False
            return markets[0]
        
        return self.get_synthetic_5m_window()
    
    def get_market_details(self, market_id: str) -> Optional[Dict]:
        if market_id and market_id.startswith('synthetic_'):
            return self.get_synthetic_5m_window()
        
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