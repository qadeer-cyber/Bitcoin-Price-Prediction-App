import logging
import time
import threading
import os
from datetime import datetime, timezone
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

MAX_DB_SIZE_MB = 100


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
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'uptime_seconds': (datetime.now(timezone.utc) - self._start_time).total_seconds()
        }
        
        if detailed:
            try:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                system = {
                    'cpu_percent': psutil.cpu_percent(interval=0.1),
                    'memory_percent': psutil.virtual_memory().percent,
                    'process_memory_mb': memory_info.rss / 1024 / 1024,
                    'thread_count': threading.active_count()
                }
            except:
                system = {'message': 'psutil not available'}
            
            try:
                db_size = self._get_db_size_mb()
                db_health = {
                    'connected': True,
                    'size_mb': db_size,
                    'size_warning': db_size > MAX_DB_SIZE_MB
                }
            except:
                db_health = {'connected': False, 'error': 'Could not determine size'}
            
            health.update({
                'database': db_health,
                'cache': {
                    'hits': self._cache_hits,
                    'misses': self._cache_misses,
                    'hit_rate': self._cache_hits / max(1, self._cache_hits + self._cache_misses)
                },
                'api_errors': self._api_errors,
                'system': system
            })
            
            health['alerts'] = self._generate_alerts(db_size if detailed else 0)
        
        return health
    
    def _get_db_size_mb(self) -> float:
        """Get database file size in MB"""
        try:
            db_path = os.environ.get('DATABASE_URL', 'sqlite:///polysignal.db')
            if db_path.startswith('sqlite:///'):
                path = db_path.replace('sqlite:///', '')
                if os.path.exists(path):
                    return os.path.getsize(path) / 1024 / 1024
        except:
            pass
        return 0
    
    def _generate_alerts(self, db_size: float) -> list:
        """Generate health alerts"""
        alerts = []
        
        if db_size > MAX_DB_SIZE_MB:
            alerts.append({
                'level': 'warning',
                'message': f'Database size ({db_size:.0f}MB) exceeds recommended limit'
            })
        
        if self._api_errors > 10:
            alerts.append({
                'level': 'error',
                'message': f'High API error count: {self._api_errors}'
            })
        
        return alerts
    
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