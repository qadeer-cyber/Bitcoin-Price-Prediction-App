import logging
import json
from flask import Blueprint, jsonify, request
from datetime import datetime, timezone, timedelta
from typing import Optional

from app.services.polymarket_service import polymarket_service
from app.services.chainlink_service import binance_service
from app.services.strategy_service import strategy_service
from app.services.settlement_service import settlement_service
from app.services.analytics_service import analytics_service
from app.services.backtest_service import EnhancedBacktestService
from app.models.db import db, SignalLog, MarketSnapshot, Settings
from app.utils.time_utils import get_market_window_times, format_time_remaining
from app.utils.formatters import format_price, format_percentage

logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__)


@api_bp.route('/market/current', methods=['GET'])
def get_current_market():
    try:
        market = polymarket_service.get_current_btc_5min_market()
        
        if not market:
            return jsonify({'error': 'No active market found'}), 404
        
        btc_price = binance_service.get_btc_usdt_price()
        
        window_times = get_market_window_times()
        
        market_id = market.get('conditionId') or market.get('id', 'unknown')
        
        outcome_prices = market.get('outcomePrices', [])
        if isinstance(outcome_prices, str):
            import json
            try:
                outcome_prices = json.loads(outcome_prices)
            except:
                outcome_prices = [0.5, 0.5]
        
        up_prob = outcome_prices[0] if len(outcome_prices) > 0 else 0.5
        down_prob = 1 - up_prob
        
        snapshot = MarketSnapshot(
            market_id=market_id,
            event_title=market.get('question', ''),
            window_start=window_times['window_start'],
            window_end=window_times['window_end'],
            price_to_beat=btc_price,
            live_price=btc_price,
            up_probability=up_prob,
            down_probability=down_prob,
            status='live',
            volume_usd=market.get('volume')
        )
        db.session.add(snapshot)
        db.session.commit()
        
        return jsonify({
            'market_id': market_id,
            'event_title': market.get('question', ''),
            'window_start': window_times['window_start'].isoformat(),
            'window_end': window_times['window_end'].isoformat(),
            'price_to_beat': btc_price,
            'live_price': btc_price,
            'up_probability': up_prob,
            'down_probability': down_prob,
            'time_remaining': window_times['remaining_seconds'],
            'status': 'live'
        })
    
    except Exception as e:
        logger.error(f'Error getting current market: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/market/history', methods=['GET'])
def get_market_history():
    try:
        limit = request.args.get('limit', 20, type=int)
        
        snapshots = MarketSnapshot.query.order_by(
            MarketSnapshot.snapshot_time.desc()
        ).limit(limit).all()
        
        return jsonify({
            'markets': [s.to_dict() for s in snapshots]
        })
    
    except Exception as e:
        logger.error(f'Error getting market history: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/signal/current', methods=['GET'])
def get_current_signal():
    try:
        market_data = {}
        
        market = polymarket_service.get_current_btc_5min_market()
        
        if market:
            btc_price = binance_service.get_btc_usdt_price()
            window_times = get_market_window_times()
            
            outcome_prices = market.get('outcomePrices', [])
            if isinstance(outcome_prices, str):
                import json
                try:
                    outcome_prices = json.loads(outcome_prices)
                except:
                    outcome_prices = [0.5, 0.5]
            
            up_prob = outcome_prices[0] if len(outcome_prices) > 0 else 0.5
            
            market_data = {
                'market_id': market.get('conditionId') or market.get('id', 'unknown'),
                'event_title': market.get('question', ''),
                'price_to_beat': btc_price,
                'up_probability': up_prob,
                'window_start': window_times['window_start'],
                'window_end': window_times['window_end'],
                'time_remaining': window_times['remaining_seconds']
            }
        
        signal = strategy_service().generate_signal(market_data)
        
        if signal['direction'] != 'NO_TRADE':
            existing = SignalLog.query.filter(
                SignalLog.market_id == signal['market_id'],
                SignalLog.timestamp >= datetime.now(timezone.utc) - timedelta(seconds=10)
            ).first()
            
            if not existing:
                signal_log = SignalLog(
                    market_id=signal['market_id'],
                    signal_direction=signal['direction'],
                    confidence=signal['confidence'],
                    confidence_tier=signal['tier'],
                    price_to_beat=signal['price_to_beat'],
                    live_price_at_signal=signal['live_price'],
                    reasoning='; '.join(signal['reasoning']),
                    regime=signal['regime'],
                    market_probability_up=signal.get('up_probability'),
                    time_remaining_at_signal=signal.get('time_remaining', 0)
                )
                db.session.add(signal_log)
                db.session.commit()
        
        settlement_service.check_pending_markets()
        
        return jsonify(signal)
    
    except Exception as e:
        logger.error(f'Error getting current signal: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/signals/history', methods=['GET'])
def get_signals_history():
    try:
        limit = request.args.get('limit', 50, type=int)
        direction = request.args.get('direction')
        min_conf = request.args.get('min_confidence', 0, type=int)
        correct = request.args.get('correct')
        
        correct_filter = None
        if correct == 'true':
            correct_filter = True
        elif correct == 'false':
            correct_filter = False
        
        signals = analytics_service.get_signal_history(
            limit=limit,
            direction=direction,
            min_confidence=min_conf,
            correct=correct_filter
        )
        
        return jsonify({'signals': signals})
    
    except Exception as e:
        logger.error(f'Error getting signals history: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/analytics/today', methods=['GET'])
def get_today_analytics():
    try:
        stats = analytics_service.get_today_stats()
        
        return jsonify(stats)
    
    except Exception as e:
        logger.error(f'Error getting today analytics: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/analytics/rolling', methods=['GET'])
def get_rolling_analytics():
    try:
        window = request.args.get('window', 10, type=int)
        
        rolling = analytics_service.get_rolling_accuracy(window)
        
        return jsonify({'rolling': rolling})
    
    except Exception as e:
        logger.error(f'Error getting rolling analytics: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/analytics/confidence-buckets', methods=['GET'])
def get_confidence_bucket_analytics():
    try:
        buckets = analytics_service.get_accuracy_by_confidence_bucket()
        
        return jsonify(buckets)
    
    except Exception as e:
        logger.error(f'Error getting confidence bucket analytics: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/analytics/hourly', methods=['GET'])
def get_hourly_analytics():
    try:
        hourly = analytics_service.get_accuracy_by_hour()
        
        return jsonify(hourly)
    
    except Exception as e:
        logger.error(f'Error getting hourly analytics: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/analytics/regime', methods=['GET'])
def get_regime_analytics():
    try:
        regime = analytics_service.get_accuracy_by_regime()
        
        return jsonify(regime)
    
    except Exception as e:
        logger.error(f'Error getting regime analytics: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/analytics/updown', methods=['GET'])
def get_up_down_analytics():
    try:
        up_down = analytics_service.get_up_vs_down_performance()
        
        return jsonify(up_down)
    
    except Exception as e:
        logger.error(f'Error getting up/down analytics: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/analytics/confusion-matrix', methods=['GET'])
def get_confusion_matrix():
    try:
        matrix = analytics_service.get_confusion_matrix()
        
        return jsonify(matrix)
    
    except Exception as e:
        logger.error(f'Error getting confusion matrix: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/analytics/pnl', methods=['GET'])
def get_pnl_analytics():
    try:
        stake = request.args.get('stake', 1.0, type=float)
        
        pnl = analytics_service.get_simulated_pnl(stake)
        
        return jsonify(pnl)
    
    except Exception as e:
        logger.error(f'Error getting PnL analytics: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/settings', methods=['GET'])
def get_settings():
    try:
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
        
        return jsonify(settings)
    
    except Exception as e:
        logger.error(f'Error getting settings: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/settings', methods=['POST'])
def update_settings():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        allowed_keys = [
            'ema_short', 'ema_medium', 'ema_long',
            'rsi_period', 'rsi_overbought', 'rsi_oversold',
            'macd_fast', 'macd_slow', 'macd_signal',
            'atr_period', 'confidence_threshold', 'no_trade_aggressiveness',
            'spread_warning', 'min_liquidity', 'final_minute_penalty',
            'refresh_interval'
        ]
        
        for key, value in data.items():
            if key in allowed_keys:
                Settings.set_value(key, value)
        
        strategy_service().refresh_settings()
        
        return jsonify({'status': 'success', 'message': 'Settings updated'})
    
    except Exception as e:
        logger.error(f'Error updating settings: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/backtest/run', methods=['POST'])
def run_backtest():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        use_synthetic = data.get('use_synthetic', True)
        initial_balance = data.get('initial_balance', 10000)
        stake_per_trade = data.get('stake_per_trade', 100)
        
        if not start_date or not end_date:
            return jsonify({'error': 'start_date and end_date required'}), 400
        
        backtest_svc = EnhancedBacktestService()
        result = backtest_svc.run_backtest(
            start_date=start_date,
            end_date=end_date,
            use_synthetic=use_synthetic,
            initial_balance=initial_balance,
            stake_per_trade=stake_per_trade
        )
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f'Error running backtest: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat()
    })


@api_bp.route('/health/detailed', methods=['GET'])
def detailed_health_check():
    try:
        from app.services.health_service import health_check_service, get_cache
        from app.services.backtest_service import BacktestRun
        
        detailed = request.args.get('detailed', 'false').lower() == 'true'
        
        health = health_check_service.check_health(detailed=detailed)
        
        if detailed:
            try:
                db_ok = db.session.execute(db.text('SELECT 1')).scalar is not None
                health['database']['connected'] = db_ok
            except:
                health['database']['connected'] = False
            
            try:
                last_run = BacktestRun.query.order_by(BacktestRun.created_at.desc()).first()
                if last_run:
                    health['backtest']['last_run'] = last_run.created_at.isoformat()
            except:
                pass
        
        return jsonify(health)
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@api_bp.route('/backtest/runs', methods=['GET'])
def get_backtest_runs():
    try:
        from app.services.backtest_service import enhanced_backtest_service
        
        limit = request.args.get('limit', 10, type=int)
        runs = enhanced_backtest_service.get_backtest_runs(limit)
        
        return jsonify({'runs': runs})
    
    except Exception as e:
        logger.error(f'Error getting backtest runs: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/market/history/resolved', methods=['GET'])
def get_resolved_markets():
    """Fetch actual resolved Polymarket BTC 5-minute markets
    
    Query params:
        days: Number of days to look back (default: 7)
    """
    try:
        days = request.args.get('days', 7, type=int)
        
        markets = polymarket_service.get_historical_resolved_markets(days=days)
        
        return jsonify({
            'count': len(markets),
            'markets': markets
        })
    
    except Exception as e:
        logger.error(f'Error getting resolved markets: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/market/history/condition/<condition_id>', methods=['GET'])
def get_markets_by_condition(condition_id):
    """Fetch resolved markets by condition ID
    
    Query params:
        start_date: Start date (ISO format)
        end_date: End date (ISO format)
    """
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        markets = polymarket_service.fetch_resolved_markets(
            condition_id=condition_id,
            start_date=start_date,
            end_date=end_date
        )
        
        return jsonify({
            'condition_id': condition_id,
            'count': len(markets),
            'markets': markets
        })
    
    except Exception as e:
        logger.error(f'Error getting markets by condition: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/optimizer/grid', methods=['POST'])
def run_grid_search():
    """Run grid search optimization
    
    Body:
        start_date: str
        end_date: str
        params_to_optimize: List[str] (optional)
        metric: str (win_rate, expectancy, roi, max_drawdown)
        use_synthetic: bool
    """
    try:
        from app.services.optimizer_service import strategy_optimizer
        
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        if not start_date or not end_date:
            return jsonify({'error': 'start_date and end_date required'}), 400
        
        result = strategy_optimizer.grid_search(
            start_date=start_date,
            end_date=end_date,
            params_to_optimize=data.get('params_to_optimize'),
            metric=data.get('metric', 'win_rate'),
            use_synthetic=data.get('use_synthetic', True),
            initial_balance=data.get('initial_balance', 10000),
            stake_per_trade=data.get('stake_per_trade', 100)
        )
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f'Error running grid search: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/optimizer/quick', methods=['POST'])
def run_quick_optimize():
    """Run quick optimization
    
    Body:
        start_date: str
        end_date: str
    """
    try:
        from app.services.optimizer_service import strategy_optimizer
        
        data = request.get_json() or {}
        
        start_date = data.get('start_date', (datetime.now(timezone.utc) - timedelta(days=3)).strftime('%Y-%m-%d'))
        end_date = data.get('end_date', datetime.now(timezone.utc).strftime('%Y-%m-%d'))
        
        result = strategy_optimizer.quick_optimize(start_date, end_date)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f'Error running quick optimization: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/analytics/ml/features', methods=['GET'])
def get_ml_features():
    """Get current ML features
    
    Query params:
        include_history: bool (optional)
    """
    try:
        from app.services.ml_service import ml_feature_engine
        
        include_history = request.args.get('include_history', 'false').lower() == 'true'
        
        features = ml_feature_engine.get_features()
        
        result = {'features': features}
        
        if include_history:
            result['price_history'] = list(ml_feature_engine._price_history)
            result['signal_history_count'] = len(ml_feature_engine._signal_history)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f'Error getting ML features: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/ml/train', methods=['POST'])
def train_ml_model():
    """Train the ML model
    
    Body:
        min_samples: int (optional, default: 100)
    """
    try:
        from app.services.ml_service import ml_prediction_engine
        
        data = request.get_json() or {}
        min_samples = data.get('min_samples', 100)
        
        result = ml_prediction_engine.train_model(min_samples=min_samples)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f'Error training ML model: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/ml/status', methods=['GET'])
def get_ml_status():
    """Get ML model status"""
    try:
        from app.services.ml_service import ml_prediction_engine
        
        status = ml_prediction_engine.get_status()
        
        return jsonify(status)
    
    except Exception as e:
        logger.error(f'Error getting ML status: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/ml/predict', methods=['POST'])
def get_ml_predict():
    """Get ML prediction for current market
    
    If no features provided, uses current market data
    """
    try:
        from app.services.ml_service import ml_prediction_engine
        
        data = request.get_json() or {}
        
        if data:
            prediction = ml_prediction_engine.predict(data)
        else:
            from app.services.chainlink_service import binance_service
            from app.services.polymarket_service import polymarket_service
            
            btc_price = binance_service.get_btc_usdt_price()
            market = polymarket_service.get_current_btc_5min_market()
            
            features = {
                'price': btc_price or 0,
                'price_change_1m': 0,
                'price_change_5m': 0,
                'price_change_15m': 0,
                'rsi_14': 50,
                'volatility_5m': 0,
                'volatility_15m': 0,
                'ma_ratio_5_20': 1.0,
                'ma_ratio_20_50': 1.0,
                'market_probability_up': market.get('up_probability', 0.5) if market else 0.5,
                'signal_confidence': 50,
                'regime': 'unknown',
                'time_remaining': 300,
                'momentum_5': 0,
                'momentum_15': 0,
                'trend_strength': 50,
            }
            
            prediction = ml_prediction_engine.predict(features)
        
        return jsonify(prediction)
    
    except Exception as e:
        logger.error(f'Error getting ML prediction: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/analytics/regime', methods=['GET'])
def get_regime_analysis():
    """Get market regime analysis"""
    try:
        from app.services.ml_service import analytics_engine
        
        days = request.args.get('days', 7, type=int)
        
        result = analytics_engine.get_market_regime_analysis(days)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f'Error getting regime analysis: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/analytics/confidence', methods=['GET'])
def get_confidence_analysis():
    """Get confidence tier analysis"""
    try:
        from app.services.ml_service import analytics_engine
        
        days = request.args.get('days', 30, type=int)
        
        result = analytics_engine.get_confidence_analysis(days)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f'Error getting confidence analysis: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/analytics/timing', methods=['GET'])
def get_timing_analysis():
    """Get timing analysis"""
    try:
        from app.services.ml_service import analytics_engine
        
        days = request.args.get('days', 30, type=int)
        
        result = analytics_engine.get_timing_analysis(days)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f'Error getting timing analysis: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/analytics/rolling', methods=['GET'])
def get_rolling_performance():
    """Get rolling performance
    
    Query params:
        window_hours: int (default: 24)
    """
    try:
        from app.services.ml_service import analytics_engine
        
        window_hours = request.args.get('window_hours', 24, type=int)
        
        result = analytics_engine.get_rolling_performance(window_hours)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f'Error getting rolling performance: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/alerts/configure', methods=['POST'])
def configure_alerts():
    try:
        from app.services.alert_service import alert_service
        
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        telegram_webhook = data.get('telegram_webhook')
        discord_webhook = data.get('discord_webhook')
        email_config = data.get('email_config')
        enabled = data.get('enabled', True)
        
        alert_service.configure(
            telegram_webhook=telegram_webhook,
            discord_webhook=discord_webhook,
            email_config=email_config
        )
        alert_service.set_enabled(enabled)
        
        return jsonify({'status': 'success', 'message': 'Alerts configured'})
    
    except Exception as e:
        logger.error(f'Error configuring alerts: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/alerts/test', methods=['POST'])
def test_alert():
    try:
        from app.services.alert_service import send_market_alert
        
        message = request.get_json().get('message', 'Test alert') if request.get_json() else 'Test alert'
        send_market_alert('info', message)
        
        return jsonify({'status': 'success', 'message': 'Test alert sent'})
    
    except Exception as e:
        logger.error(f'Error sending test alert: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/cache/clear', methods=['POST'])
def clear_cache():
    try:
        from app.services.health_service import cache_service
        
        cache_service.clear()
        
        return jsonify({'status': 'success', 'message': 'Cache cleared'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/orderbook/analyze', methods=['GET'])
def analyze_orderbook():
    """Get orderbook analysis
    
    Query params:
        market_id: str (optional) - if not provided, uses current market
    """
    try:
        from app.services.orderbook_service import orderbook_service
        
        market_id = request.args.get('market_id')
        
        result = orderbook_service.get_analysis(market_id)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f'Error analyzing orderbook: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/orderbook/pressure', methods=['GET'])
def get_pressure_gauge():
    """Get orderbook pressure gauge for UI
    
    Returns simplified pressure data for dashboard gauge display
    """
    try:
        from app.services.orderbook_service import orderbook_service
        
        market_id = request.args.get('market_id')
        
        result = orderbook_service.get_pressure_gauge(market_id)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f'Error getting pressure gauge: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/orderbook/whales', methods=['GET'])
def get_whale_alerts():
    """Get recent whale order alerts
    
    Query params:
        limit: int (default: 10)
    """
    try:
        from app.services.orderbook_service import orderbook_service
        
        limit = request.args.get('limit', 10, type=int)
        
        alerts = orderbook_service._analyzer.get_recent_whale_alerts(limit)
        
        return jsonify({'alerts': alerts, 'count': len(alerts)})
    
    except Exception as e:
        logger.error(f'Error getting whale alerts: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/risk/limits', methods=['GET'])
def get_risk_limits():
    """Get current risk limits"""
    try:
        from app.services.risk_service import risk_engine
        
        return jsonify(risk_engine.get_risk_limits())
    
    except Exception as e:
        logger.error(f'Error getting risk limits: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/risk/position_size', methods=['POST'])
def calculate_position_size():
    """Calculate position size
    
    Body:
        method: str (kelly, fixed_fractional, volatility_adjusted)
        account_balance: float
        win_rate: float (0-1)
        avg_win_loss_ratio: float
        volatility: float (optional)
    """
    try:
        from app.services.risk_service import risk_engine
        
        data = request.get_json() or {}
        
        method = data.get('method', 'fixed_fractional')
        account_balance = data.get('account_balance', 10000)
        win_rate = data.get('win_rate', 0.5)
        avg_win_loss_ratio = data.get('avg_win_loss_ratio', 1.5)
        volatility = data.get('volatility', 0.02)
        
        size = risk_engine.calculate_position_size(
            method, account_balance, win_rate, avg_win_loss_ratio, volatility
        )
        
        return jsonify({
            'position_size': size,
            'method': method,
            'account_balance': account_balance
        })
    
    except Exception as e:
        logger.error(f'Error calculating position size: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/risk/circuit_breaker', methods=['GET'])
def check_circuit_breaker():
    """Check circuit breaker status
    
    Query params:
        daily_pnl: float
        account_balance: float
    """
    try:
        from app.services.risk_service import risk_engine
        
        daily_pnl = request.args.get('daily_pnl', 0, type=float)
        account_balance = request.args.get('account_balance', 10000, type=float)
        
        result = risk_engine.check_circuit_breaker(daily_pnl, account_balance)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f'Error checking circuit breaker: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/portfolio/add_trade', methods=['POST'])
def add_portfolio_trade():
    """Add trade to portfolio
    
    Body:
        direction: str (UP or DOWN)
        size: float
        entry_price: float
        exit_price: float
        fees: float (optional)
    """
    try:
        from app.services.risk_service import portfolio_tracker
        
        data = request.get_json() or {}
        
        trade = portfolio_tracker.add_trade(
            direction=data['direction'],
            size=data['size'],
            entry_price=data['entry_price'],
            exit_price=data['exit_price'],
            fees=data.get('fees', 0)
        )
        
        return jsonify(trade)
    
    except Exception as e:
        logger.error(f'Error adding trade: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/portfolio/performance', methods=['GET'])
def get_portfolio_performance():
    """Get portfolio performance metrics"""
    try:
        from app.services.risk_service import portfolio_tracker
        
        return jsonify(portfolio_tracker.get_performance())
    
    except Exception as e:
        logger.error(f'Error getting performance: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/portfolio/equity', methods=['GET'])
def get_portfolio_equity():
    """Get equity curve
    
    Query params:
        days: int (optional)
    """
    try:
        from app.services.risk_service import portfolio_tracker
        
        days = request.args.get('days', type=int)
        
        equity = portfolio_tracker.get_equity_curve(days)
        
        return jsonify({'equity': equity, 'count': len(equity)})
    
    except Exception as e:
        logger.error(f'Error getting equity: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/portfolio/reset', methods=['POST'])
def reset_portfolio():
    """Reset portfolio"""
    try:
        from app.services.risk_service import portfolio_tracker
        
        portfolio_tracker.reset()
        
        return jsonify({'status': 'reset'})
    
    except Exception as e:
        logger.error(f'Error resetting portfolio: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/markets/available', methods=['GET'])
def get_available_markets():
    """Get all available markets"""
    try:
        from app.services.market_registry import market_registry
        
        return jsonify({'markets': market_registry.get_available_markets()})
    
    except Exception as e:
        logger.error(f'Error getting markets: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/markets/subscribe', methods=['POST'])
def subscribe_markets():
    """Subscribe to markets
    
    Body:
        market_ids: List[str]
        user_id: str (optional)
    """
    try:
        from app.services.market_registry import market_registry
        
        data = request.get_json() or {}
        market_ids = data.get('market_ids', [])
        user_id = data.get('user_id', 'default')
        
        result = market_registry.subscribe_user(user_id, market_ids)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f'Error subscribing markets: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/audit/logs', methods=['GET'])
def get_audit_logs():
    """Get audit logs (admin only)
    
    Query params:
        event_type: str (optional)
        user_id: str (optional)
        limit: int (default: 100)
    """
    try:
        from app.services.audit_service import audit_service
        
        event_type = request.args.get('event_type')
        user_id = request.args.get('user_id')
        limit = request.args.get('limit', 100, type=int)
        
        logs = audit_service.get_logs(event_type, user_id, limit)
        
        return jsonify({'logs': logs, 'count': len(logs)})
    
    except Exception as e:
        logger.error(f'Error getting audit logs: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/audit/activity', methods=['GET'])
def get_recent_activity():
    """Get recent activity summary
    
    Query params:
        hours: int (default: 24)
    """
    try:
        from app.services.audit_service import audit_service
        
        hours = request.args.get('hours', 24, type=int)
        
        return jsonify(audit_service.get_recent_activity(hours))
    
    except Exception as e:
        logger.error(f'Error getting activity: {e}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/keys', methods=['POST'])
def create_api_key():
    """Create scoped API key
    
    Body:
        name: str - Key name/description
        permissions: List[str] - e.g., ['read:signals', 'read:analytics']
    
    Note: Requires X-API-Key header or login session
    """
    try:
        import secrets
        import hashlib
        
        from app.routes.auth import get_current_user
        
        user = get_current_user()
        
        data = request.get_json() or {}
        name = data.get('name', 'API Key')
        permissions = data.get('permissions', ['read:signals'])
        
        key_value = secrets.token_hex(32)
        
        # Hash the key for storage (using SHA-256)
        key_hash = hashlib.sha256(key_value.encode()).hexdigest()
        
        from app.models.db import db, Settings
        
        key_suffix = secrets.token_hex(8)
        key = 'api_key_' + key_suffix
        
        key_record = Settings(
            key=key,
            value=json.dumps({
                'hash': key_hash,
                'name': name,
                'permissions': permissions,
                'created_at': datetime.now(timezone.utc).isoformat()
            })
        )
        db.session.add(key_record)
        db.session.commit()
        
        from app.services.audit_service import audit_service
        audit_service.log_event(
            event_type='api_key',
            user_id=user.user_id if user else None,
            action='create',
            details={'name': name, 'permissions': permissions}
        )
        
        return jsonify({
            'api_key': key_value,
            'name': name,
            'permissions': permissions,
            'created_at': datetime.now(timezone.utc).isoformat()
        }), 201
    
    except Exception as e:
        logger.error(f'Error creating API key: {e}')
        return jsonify({'error': str(e)}), 500