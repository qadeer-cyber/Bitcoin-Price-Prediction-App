import requests
import logging
from typing import Optional, List, Dict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class ChainlinkService:
    CHAINLINK_DATA_API = 'https://data-api.chainlink.io'
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'Accept': 'application/json'})
    
    def get_btc_usd_price(self) -> Optional[float]:
        try:
            url = f'{self.CHAINLINK_DATA_API}/prices/latest'
            params = {
                'feed': 'btc-usd'
            }
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'data' in data:
                return float(data['data']['price'])
        except Exception as e:
            logger.warning(f'Chainlink price fetch failed: {e}')
        
        return None
    
    def get_btc_usd_historical(self, limit: int = 100) -> List[Dict]:
        try:
            url = f'{self.CHAINLINK_DATA_API}/prices/historical'
            params = {
                'feed': 'btc-usd',
                'limit': limit
            }
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return data.get('data', [])
        except Exception as e:
            logger.warning(f'Chainlink historical fetch failed: {e}')
        
        return []

chainlink_service = ChainlinkService()


class BinanceService:
    API_URL = 'https://api.binance.com/api/v3'
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'Accept': 'application/json'})
    
    def get_btc_usdt_price(self) -> Optional[float]:
        try:
            url = f'{self.API_URL}/ticker/price'
            params = {'symbol': 'BTCUSDT'}
            response = self.session.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            return float(data.get('price', 0))
        except Exception as e:
            logger.error(f'Binance price fetch failed: {e}')
            return None
    
    def get_btc_klines(self, interval: str = '1m', limit: int = 100) -> List[Dict]:
        try:
            url = f'{self.API_URL}/klines'
            params = {
                'symbol': 'BTCUSDT',
                'interval': interval,
                'limit': limit
            }
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            klines = []
            for k in data:
                klines.append({
                    'open_time': datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5]),
                    'close_time': datetime.fromtimestamp(k[6] / 1000, tz=timezone.utc)
                })
            
            return klines
        except Exception as e:
            logger.error(f'Binance klines fetch failed: {e}')
            return []
    
    def get_24h_ticker(self) -> Optional[Dict]:
        try:
            url = f'{self.API_URL}/ticker/24hr'
            params = {'symbol': 'BTCUSDT'}
            response = self.session.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            return {
                'last_price': float(data.get('lastPrice', 0)),
                'price_change': float(data.get('priceChange', 0)),
                'price_change_percent': float(data.get('priceChangePercent', 0)),
                'high_price': float(data.get('highPrice', 0)),
                'low_price': float(data.get('lowPrice', 0)),
                'volume': float(data.get('volume', 0)),
                'quote_volume': float(data.get('quoteVolume', 0))
            }
        except Exception as e:
            logger.error(f'Binance 24h ticker fetch failed: {e}')
            return None

binance_service = BinanceService()