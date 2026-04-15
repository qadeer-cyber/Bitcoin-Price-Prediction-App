import logging
import requests
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class AlertService:
    def __init__(self):
        self._telegram_webhook = None
        self._discord_webhook = None
        self._email_config = None
        self._alert_enabled = True
    
    def configure(self, telegram_webhook: str = None, discord_webhook: str = None,
                  email_config: dict = None):
        """Configure alert destinations"""
        self._telegram_webhook = telegram_webhook
        self._discord_webhook = discord_webhook
        self._email_config = email_config
    
    def set_enabled(self, enabled: bool):
        """Enable/disable all alerts"""
        self._alert_enabled = enabled
    
    def send_signal_alert(self, signal: dict) -> bool:
        """Send signal alert to configured destinations"""
        if not self._alert_enabled:
            return False
        
        direction = signal.get('direction', 'UNKNOWN')
        confidence = signal.get('confidence', 0)
        market_id = signal.get('market_id', 'unknown')[:16]
        reasoning = signal.get('reasoning', '')[:100]
        price = signal.get('live_price', 0)
        
        emoji = '🟢' if direction == 'UP' else ('🔴' if direction == 'DOWN' else '⚪')
        
        message = f"""🔔 NEW SIGNAL: {emoji} {direction}
• Confidence: {confidence}%
• Market: {market_id}...
• Price: ${price:,.2f}
• Reasoning: {reasoning}"""
        
        results = []
        
        if self._telegram_webhook:
            results.append(self._send_telegram(message))
        
        if self._discord_webhook:
            results.append(self._send_discord(message))
        
        return any(results)
    
    def send_market_alert(self, alert_type: str, message: str) -> bool:
        """Send general market alert"""
        if not self._alert_enabled:
            return False
        
        emoji_map = {
            'market_up': '📈',
            'market_down': '📉',
            'error': '⚠️',
            'info': 'ℹ️'
        }
        
        emoji = emoji_map.get(alert_type, 'ℹ️')
        full_message = f"{emoji} {alert_type.upper()}: {message}"
        
        results = []
        
        if self._telegram_webhook:
            results.append(self._send_telegram(full_message))
        
        if self._discord_webhook:
            results.append(self._send_discord(full_message))
        
        return any(results)
    
    def _send_telegram(self, message: str) -> bool:
        """Send message via Telegram webhook"""
        if not self._telegram_webhook:
            return False
        
        try:
            payload = {'text': message, 'parse_mode': 'Markdown'}
            response = requests.post(self._telegram_webhook, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f'Telegram alert error: {e}')
            return False
    
    def _send_discord(self, message: str) -> bool:
        """Send message via Discord webhook"""
        if not self._discord_webhook:
            return False
        
        try:
            payload = {'content': message}
            response = requests.post(self._discord_webhook, json=payload, timeout=10)
            return response.status_code in [200, 204]
        except Exception as e:
            logger.error(f'Discord alert error: {e}')
            return False


alert_service = AlertService()


def send_signal_alert(signal: dict) -> bool:
    return alert_service.send_signal_alert(signal)


def send_market_alert(alert_type: str, message: str) -> bool:
    return alert_service.send_market_alert(alert_type, message)