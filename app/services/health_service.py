import logging
import time
import threading
from datetime import datetime, timezone
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class HealthCheckService:
    def __init__(self):
        self._start_time = datetime.now(timezone.utc)
        self._db_connected = True
        self._last_api_sync = None
        self._scheduler_status = 'idle'
        self._cache_hits = 0
        self._cache_misses = 0
        self._api_errors = 0
    
    def check_health(self, detailed: bool = False) -> Dict:
        health = {
            'status': 'healthy',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        if detailed:
            try:
                import psutil
                system = {
                    'cpu_percent': psutil.cpu_percent(interval=0.1),
                    'memory_percent': psutil.virtual_memory().percent
                }
            except:
                system = {'message': 'psutil not available'}
            
            health.update({
                'database': {'connected': True},
                'cache': {'hits': self._cache_hits, 'misses': self._cache_misses},
                'system': system
            })
        
        return health
    
    def record_cache_hit(self):
        self._cache_hits += 1
    
    def record_cache_miss(self):
        self._cache_misses += 1
    
    def record_api_error(self):
        self._api_errors += 1


class CacheService:
    def __init__(self, default_ttl: int = 60):
        self._cache = {}
        self._timestamps = {}
        self._default_ttl = default_ttl
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                if time.time() - self._timestamps.get(key, 0) < self._default_ttl:
                    return self._cache[key]
                else:
                    del self._cache[key]
                    del self._timestamps[key]
        return None
    
    def set(self, key: str, value: Any, ttl: int = None):
        with self._lock:
            self._cache[key] = value
            self._timestamps[key] = time.time()
    
    def clear(self):
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()


health_check_service = HealthCheckService()
cache_service = CacheService()


def get_health_check(detailed: bool = False) -> Dict:
    return health_check_service.check_health(detailed)


def get_cache(key: str) -> Optional[Any]:
    return cache_service.get(key)


def set_cache(key: str, value: Any, ttl: int = None):
    cache_service.set(key, value, ttl)