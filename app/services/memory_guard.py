"""Memory and CPU monitoring for low-RAM environments"""

import logging
import psutil

logger = logging.getLogger(__name__)

# Thresholds
MEMORY_WARNING_THRESHOLD = 85
CPU_WARNING_THRESHOLD = 90


class MemoryGuard:
    """Monitor system resources and pause heavy jobs when needed"""
    
    def __init__(self, memory_threshold=85, cpu_threshold=90):
        self.memory_threshold = memory_threshold
        self.cpu_threshold = cpu_threshold
        self._heavy_job_paused = False
    
    def check_memory(self) -> bool:
        """Check if memory is below threshold"""
        try:
            percent = psutil.virtual_memory().percent
            if percent >= self.memory_threshold:
                if not self._heavy_job_paused:
                    logger.warning(f'Memory high: {percent}%, pausing heavy jobs')
                    self._heavy_job_paused = True
                return False
            else:
                if self._heavy_job_paused:
                    logger.info('Memory normalized, resuming heavy jobs')
                    self._heavy_job_paused = False
                return True
        except:
            return True
    
    def check_cpu(self) -> bool:
        """Check if CPU is below threshold"""
        try:
            percent = psutil.cpu_percent(interval=1)
            return percent < self.cpu_threshold
        except:
            return True
    
    def can_run_heavy_job(self) -> bool:
        """Check if heavy job can run"""
        return self.check_memory() and self.check_cpu()
    
    def get_status(self) -> dict:
        """Get current status"""
        try:
            return {
                'memory_percent': psutil.virtual_memory().percent,
                'cpu_percent': psutil.cpu_percent(interval=0.5),
                'can_run_heavy': self.can_run_heavy_job(),
                'heavy_paused': self._heavy_job_paused
            }
        except:
            return {'error': 'psutil unavailable'}


memory_guard = MemoryGuard()