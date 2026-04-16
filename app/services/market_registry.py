import logging
from typing import Dict, List, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

MARKET_ASSETS = {
    'BTC': {
        'name': 'Bitcoin',
        'symbol': 'BTC',
        'markets': ['5 minute', '1 hour', 'daily'],
        'enabled': True
    },
    'ETH': {
        'name': 'Ethereum',
        'symbol': 'ETH',
        'markets': ['5 minute', '1 hour', 'daily'],
        'enabled': True
    },
    'SOL': {
        'name': 'Solana',
        'symbol': 'SOL',
        'markets': ['5 minute', '1 hour'],
        'enabled': True
    },
    'DOGE': {
        'name': 'Dogecoin',
        'symbol': 'DOGE',
        'markets': ['5 minute', '1 hour'],
        'enabled': True
    }
}


class MarketRegistry:
    """Multi-market asset registry"""
    
    def __init__(self):
        self._active_markets = {}
        self._subscribed_users = {}
    
    def get_available_markets(self) -> List[Dict]:
        """Get all available markets"""
        markets = []
        
        for asset, config in MARKET_ASSETS.items():
            if config['enabled']:
                for market_type in config['markets']:
                    markets.append({
                        'asset': asset,
                        'name': config['name'],
                        'type': market_type,
                        'market_id': f'{asset}_{market_type}'.replace(' ', '_')
                    })
        
        return markets
    
    def discover_markets(self, asset: str) -> List[Dict]:
        """Discover active markets for an asset"""
        if asset not in MARKET_ASSETS:
            return []
        
        config = MARKET_ASSETS[asset]
        if not config['enabled']:
            return []
        
        markets = []
        for market_type in config['markets']:
            markets.append({
                'asset': asset,
                'type': market_type,
                'market_id': f'{asset}_{market_type}'.replace(' ', '_')
            })
        
        return markets
    
    def subscribe_user(
        self,
        user_id: str,
        market_ids: List[str]
    ) -> Dict:
        """Subscribe user to markets"""
        if user_id not in self._subscribed_users:
            self._subscribed_users[user_id] = set()
        
        self._subscribed_users[user_id].update(market_ids)
        
        return {
            'user_id': user_id,
            'subscribed': list(self._subscribed_users[user_id])
        }
    
    def unsubscribe_user(
        self,
        user_id: str,
        market_ids: List[str] = None
    ) -> Dict:
        """Unsubscribe user from markets"""
        if user_id not in self._subscribed_users:
            return {'user_id': user_id, 'subscribed': []}
        
        if market_ids:
            self._subscribed_users[user_id].difference_update(market_ids)
        else:
            self._subscribed_users[user_id].clear()
        
        return {
            'user_id': user_id,
            'subscribed': list(self._subscribed_users[user_id])
        }
    
    def get_user_subscriptions(self, user_id: str) -> List[str]:
        """Get user's market subscriptions"""
        return list(self._subscribed_users.get(user_id, set()))
    
    def set_market_active(self, market_id: str, active: bool) -> None:
        """Mark market as active/inactive"""
        self._active_markets[market_id] = {
            'active': active,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def is_market_active(self, market_id: str) -> bool:
        """Check if market is active"""
        return self._active_markets.get(market_id, {}).get('active', False)
    
    def get_asset_config(self, asset: str) -> Optional[Dict]:
        """Get asset configuration"""
        return MARKET_ASSETS.get(asset)
    
    def enable_asset(self, asset: str, enabled: bool) -> bool:
        """Enable/disable an asset"""
        if asset in MARKET_ASSETS:
            MARKET_ASSETS[asset]['enabled'] = enabled
            return True
        return False


market_registry = MarketRegistry()