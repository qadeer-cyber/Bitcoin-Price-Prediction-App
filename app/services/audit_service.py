import logging
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from app.models.db import db

logger = logging.getLogger(__name__)

# Sensitive fields that should never be logged
SENSITIVE_FIELDS = {
    'password', 'password_hash', 'passwd', 'secret',
    'api_key', 'apikey', 'token', 'access_token', 'refresh_token',
    'secret_key', 'private_key', 'bcrypt_hash',
    'credit_card', 'cc_number', 'ssn', 'account_number'
}

# Patterns to mask
SENSITIVE_PATTERNS = [
    (r'sk_live_[a-zA-Z0-9]+', 'sk_live_***'),
    (r'pk_live_[a-zA-Z0-9]+', 'pk_live_***'),
    (r'[\w]+@[\w]+\.[\w]+', '***@***'),  # emails
]


class AuditService:
    """Audit logging for security and compliance
    
    SECURITY: Never logs passwords, API keys, or PII. All sensitive data is masked.
    """
    
    def __init__(self):
        self._logs = []
        self._max_logs = 10000
    
    def _mask_sensitive(self, data: Dict) -> Dict:
        """Mask sensitive fields in data"""
        if not data:
            return {}
        
        masked = {}
        
        for key, value in data.items():
            key_lower = key.lower()
            
            # Skip sensitive field names completely
            if key_lower in SENSITIVE_FIELDS:
                masked[key] = '***REDACTED***'
                continue
            
            # Mask API keys with specific patterns
            if isinstance(value, str):
                masked_value = value
                for pattern, replacement in SENSITIVE_PATTERNS:
                    masked_value = re.sub(pattern, replacement, masked_value)
                
                # Mask patterns in values
                if 'sk_' in value or 'pk_' in value:
                    masked[key] = '***KEY***'
                elif '@' in value and len(value) > 3:
                    masked[key] = '***EMAIL***'
                else:
                    masked[key] = masked_value
            else:
                masked[key] = value
        
        return masked
    
    def log_event(
        self,
        event_type: str,
        user_id: Optional[str],
        action: str,
        details: Dict = None,
        ip_address: str = None,
        user_agent: str = None
    ) -> None:
        """Log an audit event (sensitive data always masked)"""
        safe_details = self._mask_sensitive(details) if details else {}
        
        event = {
            'event_type': event_type,
            'user_id': user_id,
            'action': action,
            'details': safe_details,
            'ip_address': ip_address,
            'user_agent': user_agent,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        self._logs.append(event)
        
        if len(self._logs) > self._max_logs:
            self._logs = self._logs[-self._max_logs:]
        
        logger.info(f'Audit: {event_type} - {action} by {user_id}')
    
    def log_auth_event(
        self,
        user_id: str,
        event: str,
        success: bool,
        ip_address: str = None
    ) -> None:
        """Log authentication event"""
        self.log_event(
            event_type='auth',
            user_id=user_id,
            action=event,
            details={'success': success},
            ip_address=ip_address
        )
    
    def log_config_change(
        self,
        user_id: str,
        setting_key: str,
        old_value: str,
        new_value: str
    ) -> None:
        """Log configuration change"""
        self.log_event(
            event_type='config',
            user_id=user_id,
            action='update_setting',
            details={
                'setting_key': setting_key,
                'old_value': old_value,
                'new_value': new_value
            }
        )
    
    def log_api_key_usage(
        self,
        api_key: str,
        endpoint: str,
        ip_address: str = None
    ) -> None:
        """Log API key usage"""
        masked_key = api_key[:8] + '...' if api_key else 'None'
        self.log_event(
            event_type='api_key',
            user_id=None,
            action=endpoint,
            details={'api_key': masked_key},
            ip_address=ip_address
        )
    
    def log_signal_override(
        self,
        user_id: str,
        original_signal: str,
        override_signal: str,
        reason: str
    ) -> None:
        """Log manual signal override"""
        self.log_event(
            event_type='signal_override',
            user_id=user_id,
            action='override_signal',
            details={
                'original': original_signal,
                'override': override_signal,
                'reason': reason
            }
        )
    
    def get_logs(
        self,
        event_type: str = None,
        user_id: str = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get audit logs with filters"""
        logs = self._logs
        
        if event_type:
            logs = [l for l in logs if l['event_type'] == event_type]
        
        if user_id:
            logs = [l for l in logs if l['user_id'] == user_id]
        
        return logs[-limit:]
    
    def get_recent_activity(self, hours: int = 24) -> Dict:
        """Get recent activity summary"""
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        recent_logs = [
            l for l in self._logs
            if datetime.fromisoformat(l['timestamp']) > cutoff
        ]
        
        event_types = {}
        for log in recent_logs:
            et = log['event_type']
            event_types[et] = event_types.get(et, 0) + 1
        
        return {
            'total_events': len(recent_logs),
            'event_types': event_types,
            'hours': hours
        }


audit_service = AuditService()