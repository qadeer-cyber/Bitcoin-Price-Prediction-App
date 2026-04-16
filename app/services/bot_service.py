import logging
import os
import json
from datetime import datetime, timezone
from threading import Thread
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning('python-telegram-bot not installed')


class TelegramBotService:
    """Interactive Telegram Bot for Polysignal
    
    SECURITY: Requires TELEGRAM_CHAT_ID to be set for authorized access
    """
    
    def __init__(self):
        self._token = os.environ.get('TELEGRAM_BOT_TOKEN')
        self._chat_id = os.environ.get('TELEGRAM_CHAT_ID')
        self._enabled = bool(self._token and self._chat_id)
        self._app = None
        self._running = False
        self._user_states = {}
        self._alerts_enabled = {}
        self._feedback = {}
        
        self._authorized_chats = set()
        if self._chat_id:
            self._authorized_chats.add(str(self._chat_id))
        
        self._startup_messages = {}
    
    def is_enabled(self) -> bool:
        return self._enabled and TELEGRAM_AVAILABLE
    
    def _is_authorized(self, chat_id: str) -> bool:
        """Check if chat_id is authorized"""
        return str(chat_id) in self._authorized_chats
    
    def start(self) -> bool:
        """Start the bot in background thread"""
        if not self.is_enabled():
            logger.info('Telegram bot not enabled')
            return False
        
        if self._running:
            return True
        
        try:
            self._app = Application.builder().token(self._token).build()
            
            self._app.add_handler(CommandHandler('start', self._cmd_start))
            self._app.add_handler(CommandHandler('help', self._cmd_help))
            self._app.add_handler(CommandHandler('signal', self._cmd_signal))
            self._app.add_handler(CommandHandler('stats', self._cmd_stats))
            self._app.add_handler(CommandHandler('toggle', self._cmd_toggle))
            self._app.add_handler(CommandHandler('settings', self._cmd_settings))
            self._app.add_handler(CallbackQueryHandler(self._handle_callback))
            
            self._app.run_polling(allowed_updates=['message', 'callback_query'])
            self._running = True
            
            logger.info('Telegram bot started')
            return True
        
        except Exception as e:
            logger.warning(f'Failed to start Telegram bot: {e}')
            return False
    
    def stop(self) -> None:
        """Stop the bot"""
        if self._app and self._running:
            try:
                self._app.stop()
                self._running = False
                logger.info('Telegram bot stopped')
            except Exception as e:
                logger.warning(f'Error stopping bot: {e}')
    
    async def _check_auth(self, update) -> bool:
        """Check if user is authorized"""
        chat_id = str(update.effective_chat.id)
        if not self._is_authorized(chat_id):
            await update.message.reply_text('❌ Unauthorized. Contact admin for access.')
            logger.warning(f'Unauthorized access attempt from {chat_id}')
            return False
        return True
    
    async def _cmd_start(self, update, context):
        """Handle /start command"""
        if not await self._check_auth(update):
            return
        await update.message.reply_text(
            '🪙 *Polysignal BTC* - Real-time Binary Trading Signals\n\n'
            '_Track Polymarket BTC 5-minute markets with precision._\n\n'
            'Commands:\n'
            '/signal - Get current signal\n'
            '/stats - View performance stats\n'
            '/toggle - Turn alerts on/off\n'
            '/settings - Manage preferences\n'
            '/help - Show help',
            parse_mode='Markdown'
        )
    
    async def _cmd_help(self, update, context):
        """Handle /help command"""
        if not await self._check_auth(update):
            return
        await update.message.reply_text(
            '📖 *Help*\n\n'
            '*Commands:*\n'
            '/signal - Get current market signal\n'
            '/stats - Today\'s win rate & PnL\n'
            '/toggle - Enable/disable alerts\n'
            '/settings - Manage notifications\n\n'
            '*Alerts:*\n'
            'When enabled, receive alerts for new signals\n'
            'Use Agree/Disagree buttons to provide feedback',
            parse_mode='Markdown'
        )
    
    async def _cmd_signal(self, update, context):
        """Handle /signal command - get current signal"""
        if not await self._check_auth(update):
            return
        try:
            from app.services.strategy_service import strategy_service
            from app.services.polymarket_service import polymarket_service
            from app.services.orderbook_service import orderbook_service
            
            market = polymarket_service.get_current_btc_5min_market()
            if not market:
                await update.message.reply_text('No active market')
                return
            
            signal = strategy_service.generate_signal(market)
            
            direction = signal.get('direction', 'UNKNOWN')
            confidence = signal.get('confidence', 0)
            tier = signal.get('tier', 'unknown')
            
            emoji = '🟢' if direction == 'UP' else '🔴' if direction == 'DOWN' else '⚪'
            
            message = f'{emoji} *Signal: {direction}*\n\n'
            message += f'Confidence: {confidence}% ({tier})\n'
            
            if signal.get('reasoning'):
                message += '\n_Reasoning:_\n'
                for r in signal.get('reasoning', [])[:3]:
                    message += f'• {r}\n'
            
            pressure = orderbook_service.get_pressure_gauge()
            if pressure:
                message += f'\nOrderbook: {pressure.get("label", "neutral")}'
            
            keyboard = [
                [
                    InlineKeyboardButton('👍 Agree', callback_data='agree'),
                    InlineKeyboardButton('👎 Disagree', callback_data='disagree')
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                message,
                parse_mode='Markdown',
                reply_markup=reply_markup
            )
        
        except Exception as e:
            logger.error(f'Signal command error: {e}')
            await update.message.reply_text('Error getting signal')
    
    async def _cmd_stats(self, update, context):
        """Handle /stats command"""
        if not await self._check_auth(update):
            return
        try:
            from app.services.analytics_service import analytics_service
            
            today = analytics_service.get_today_summary()
            
            win_rate = today.get('win_rate', 0)
            signals = today.get('signals_generated', 0)
            correct = today.get('correct', 0)
            
            message = f'📊 *Today\'s Stats*\n\n'
            message += f'Signals: {signals}\n'
            message += f'Correct: {correct}\n'
            message += f'Win Rate: {win_rate:.1f}%\n'
            
            await update.message.reply_text(message, parse_mode='Markdown')
        
        except Exception as e:
            logger.error(f'Stats command error: {e}')
            await update.message.reply_text('Error getting stats')
    
    async def _cmd_toggle(self, update, context):
        """Handle /toggle command"""
        if not await self._check_auth(update):
            return
        user_id = str(update.effective_user.id)
        
        self._alerts_enabled[user_id] = not self._alerts_enabled.get(user_id, True)
        
        status = 'enabled' if self._alerts_enabled[user_id] else 'disabled'
        
        await update.message.reply_text(
            f'Alerts {status}!'
        )
    
    async def _cmd_settings(self, update, context):
        """Handle /settings command"""
        if not await self._check_auth(update):
            return
        keyboard = [
            [
                InlineKeyboardButton('Telegram', callback_data='set_telegram'),
                InlineKeyboardButton('Discord', callback_data='set_discord')
            ],
            [
                InlineKeyboardButton('All', callback_data='set_all'),
                InlineKeyboardButton('Premium Only', callback_data='set_premium')
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            '⚙️ *Settings*\n\nSelect notification preferences:',
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
    
    async def _handle_callback(self, update, context):
        """Handle inline button callbacks"""
        query = update.callback_query
        await query.answer()
        
        user_id = str(query.from_user.id)
        data = query.data
        
        if data == 'agree':
            self._feedback[user_id] = self._feedback.get(user_id, 0) + 1
            await query.edit_message_text('Thanks for feedback! 👍')
        elif data == 'disagree':
            await query.edit_message_text('Thanks for feedback! 👎')
        elif data.startswith('set_'):
            await query.edit_message_text('Settings updated!')
    
    def send_alert(self, signal: Dict) -> bool:
        """Send signal alert to enabled users"""
        if not self._running:
            return False
        
        try:
            message = self._format_signal_alert(signal)
            
            for user_id, enabled in self._alerts_enabled.items():
                if enabled:
                    self._app.bot.send_message(chat_id=user_id, text=message)
            
            return True
        
        except Exception as e:
            logger.warning(f'Failed to send alert: {e}')
            return False
    
    def _format_signal_alert(self, signal: Dict) -> str:
        """Format signal for alert"""
        direction = signal.get('direction', 'UNKNOWN')
        confidence = signal.get('confidence', 0)
        
        emoji = '🟢' if direction == 'UP' else '🔴' if direction == 'DOWN' else '⚪'
        
        message = f'{emoji} *New Signal: {direction}* ({confidence}%)\n\n'
        
        if signal.get('reasoning'):
            message += signal.get('reasoning', [])[0]
        
        return message
    
    def get_user_preferences(self, user_id: str) -> Dict:
        """Get user preferences"""
        return {
            'alerts_enabled': self._alerts_enabled.get(user_id, True),
            'feedback_count': self._feedback.get(user_id, 0)
        }


class BotRunner:
    """Background bot runner"""
    
    def __init__(self):
        self._bot = None
    
    def start(self) -> None:
        """Start bot in background thread"""
        if not TELEGRAM_AVAILABLE:
            logger.info('Bot not available - skipping')
            return
        
        self._bot = TelegramBotService()
        
        if not self._bot.is_enabled():
            logger.info('Bot not configured - skipping')
            return
        
        thread = Thread(target=self._run_bot, daemon=True)
        thread.start()
    
    def _run_bot(self) -> None:
        """Run bot"""
        self._bot.start()


telegram_bot = TelegramBotService()
bot_runner = BotRunner()