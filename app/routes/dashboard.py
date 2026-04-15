import logging
from flask import Blueprint, render_template, jsonify, request
from datetime import datetime, timezone

from app.services.polymarket_service import polymarket_service
from app.services.chainlink_service import binance_service
from app.services.strategy_service import strategy_service
from app.services.analytics_service import analytics_service
from app.utils.time_utils import get_market_window_times

logger = logging.getLogger(__name__)

dashboard_bp = Blueprint('dashboard', __name__)


@dashboard_bp.route('/')
def index():
    try:
        market = polymarket_service.get_current_btc_5min_market()
        
        btc_price = binance_service.get_btc_usdt_price()
        window_times = get_market_window_times()
        
        if market:
            outcome_prices = market.get('outcomePrices', [])
            if isinstance(outcome_prices, str):
                import json
                try:
                    outcome_prices = json.loads(outcome_prices)
                except:
                    outcome_prices = [0.5, 0.5]
            
            up_prob = outcome_prices[0] if len(outcome_prices) > 0 else 0.5
        else:
            up_prob = 0.5
        
        market_data = {
            'market_id': market.get('conditionId') if market else 'unknown',
            'event_title': market.get('question') if market else 'Loading...',
            'price_to_beat': btc_price,
            'up_probability': up_prob,
            'window_start': window_times['window_start'],
            'window_end': window_times['window_end'],
            'time_remaining': window_times['remaining_seconds']
        }
        
        signal = strategy_service().generate_signal(market_data)
        
        today_stats = analytics_service.get_today_stats()
        
        recent_signals = analytics_service.get_signal_history(limit=10)
        
        return render_template(
            'dashboard.html',
            market=market_data,
            signal=signal,
            today_stats=today_stats,
            recent_signals=recent_signals,
            btc_price=btc_price
        )
    
    except Exception as e:
        logger.error(f'Error in dashboard: {e}')
        return render_template(
            'dashboard.html',
            market={},
            signal={'direction': 'NO_TRADE', 'confidence': 0, 'reasoning': ['Error loading data']},
            today_stats={},
            recent_signals=[],
            btc_price=None
        )


@dashboard_bp.route('/history')
def history():
    try:
        signals = analytics_service.get_signal_history(limit=100)
        
        return render_template('history.html', signals=signals)
    
    except Exception as e:
        logger.error(f'Error in history: {e}')
        return render_template('history.html', signals=[])


@dashboard_bp.route('/analytics')
def analytics():
    try:
        today_stats = analytics_service.get_today_stats()
        confidence_buckets = analytics_service.get_accuracy_by_confidence_bucket()
        hourly = analytics_service.get_accuracy_by_hour()
        regime = analytics_service.get_accuracy_by_regime()
        up_down = analytics_service.get_up_vs_down_performance()
        rolling = analytics_service.get_rolling_accuracy(10)
        pnl = analytics_service.get_simulated_pnl()
        
        return render_template(
            'analytics.html',
            today_stats=today_stats,
            confidence_buckets=confidence_buckets,
            hourly=hourly,
            regime=regime,
            up_down=up_down,
            rolling=rolling,
            pnl=pnl
        )
    
    except Exception as e:
        logger.error(f'Error in analytics: {e}')
        return render_template('analytics.html', today_stats={})


@dashboard_bp.route('/settings')
def settings():
    try:
        from app.models.db import Settings
        
        settings = {
            'ema_short': Settings.get_value('ema_short', 9),
            'ema_medium': Settings.get_value('ema_medium', 20),
            'ema_long': Settings.get_value('ema_long', 50),
            'rsi_period': Settings.get_value('rsi_period', 14),
            'rsi_overbought': Settings.get_value('rsi_overbought', 70),
            'rsi_oversold': Settings.get_value('rsi_oversold', 30),
            'macd_fast': Settings.get_value('macd_fast', 12),
            'macd_slow': Settings.get_value('macd_slow', 26),
            'macd_signal': Settings.get_value('macd_signal', 9),
            'atr_period': Settings.get_value('atr_period', 14),
            'confidence_threshold': Settings.get_value('confidence_threshold', 50),
            'no_trade_aggressiveness': Settings.get_value('no_trade_aggressiveness', 60),
            'spread_warning': Settings.get_value('spread_warning', 5),
            'min_liquidity': Settings.get_value('min_liquidity', 1000),
            'final_minute_penalty': Settings.get_value('final_minute_penalty', 20),
            'refresh_interval': Settings.get_value('refresh_interval', 5)
        }
        
        return render_template('settings.html', settings=settings)
    
    except Exception as e:
        logger.error(f'Error in settings: {e}')
        return render_template('settings.html', settings={})