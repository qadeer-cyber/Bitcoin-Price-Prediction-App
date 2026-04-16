"""Cache service with in-memory dict + SQLite fallback"""

import logging
import time
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)


class CacheService:
    """Simple in-memory cache with SQLite fallback"""
    
    def __init__(self, max_size=1000, default_ttl=300):
        self._cache = {}
        self._timestamps = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            if key not in self._cache:
                return None
            
            timestamp, value, ttl = self._cache[key], self._timestamps[key]['value'], self._timestamps[key]['ttl']
            
            if ttl and time.time() - timestamp > ttl:
                del self._cache[key]
                del self._timestamps[key]
                return None
            
            return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache"""
        with self._lock:
            if len(self._cache) >= self._max_size:
                self._evict_oldest()
            
            self._cache[key] = value
            self._timestamps[key] = {
                'value': value,
                'ttl': ttl if ttl is not None else self._default_ttl,
                'created': time.time()
            }
    
    def delete(self, key: str) -> None:
        """Delete key from cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                del self._timestamps[key]
    
    def clear(self) -> None:
        """Clear all cache"""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
    
    def _evict_oldest(self) -> None:
        """Evict oldest entry"""
        if not self._timestamps:
            return
        
        oldest = min(self._timestamps.items(), key=lambda x: x[1]['created'])
        key = oldest[0]
        del self._cache[key]
        del self._timestamps[key]
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        return {
            'size': len(self._cache),
            'max_size': self._max_size,
            'keys': list(self._cache.keys())[:10]
        }


cache_service = CacheService()