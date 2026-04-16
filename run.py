import os
import sys
import logging
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

# Initialize memory guardrail
def check_memory():
    """Check available memory before heavy operations"""
    try:
        import psutil
        return psutil.virtual_memory().percent < 85
    except:
        return True

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from flask_login import LoginManager
from datetime import datetime, timezone

from app.models.db import db, Settings, User
from app.routes import dashboard, api
from app.routes.auth import auth_bp
from app.routes.websocket import ws_bp

# Initialize Sentry (optional)
sentry_dsn = os.environ.get('SENTRY_DSN')
if sentry_dsn:
    sentry_sdk.init(
        dsn=sentry_dsn,
        integrations=[FlaskIntegration()],
        traces_sample_rate=0.1,
        environment=os.environ.get('FLASK_ENV', 'production')
    )
    logger.info('Sentry initialized')

# APScheduler for lightweight background tasks (replaces Celery)
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    SCHEDULER = BackgroundScheduler()
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER = None
    SCHEDULER_AVAILABLE = False

login_manager = LoginManager()
login_manager.login_view = 'dashboard.login'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__, template_folder='app/templates', static_folder='app/static')
    
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'poly-signal-btc-2026-secret')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///polysignal.db')
    pool_size = int(os.environ.get('SQLALCHEMY_POOL_SIZE', '10'))
    max_overflow = int(os.environ.get('SQLALCHEMY_MAX_OVERFLOW', '20'))
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'pool_size': pool_size,
        'max_overflow': max_overflow
    }
    
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    db.init_app(app)
    login_manager.init_app(app)
    
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.filter_by(user_id=user_id).first()
    
    with app.app_context():
        db.create_all()
        
        _init_default_settings()
        _init_telegram_bot()
        _init_scheduler()
    
    # Memory guard: Skip heavy background tasks on low RAM
    if check_memory() and SCHEDULER_AVAILABLE:
        try:
            SCHEDULER.start()
            logger.info('APScheduler started')
        except Exception as e:
            logger.warning(f'Scheduler start failed: {e}')
    
    app.register_blueprint(dashboard.dashboard_bp)
    app.register_blueprint(api.api_bp, url_prefix='/api')
    app.register_blueprint(auth_bp, url_prefix='/api')
    app.register_blueprint(ws_bp, url_prefix='/api')
    
    @app.context_processor
    def inject_now():
        return {'now': datetime.now(timezone.utc)}
    
    return app


def _init_default_settings():
    defaults = {
        'ema_short': 9,
        'ema_medium': 20,
        'ema_long': 50,
        'rsi_period': 14,
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'atr_period': 14,
        'confidence_threshold': 50,
        'no_trade_aggressiveness': 60,
        'spread_warning': 5,
        'min_liquidity': 1000,
        'final_minute_penalty': 20,
        'refresh_interval': 5
    }
    
    for key, value in defaults.items():
        existing = Settings.query.filter_by(key=key).first()
        if existing is None:
            setting = Settings(key=key, value=str(value))
            db.session.add(setting)
    
    db.session.commit()


def _init_telegram_bot():
    """Initialize Telegram bot if configured"""
    try:
        from app.services.bot_service import bot_runner
        bot_runner.start()
    except Exception as e:
        logger.warning(f'Telegram bot init failed: {e}')


def _init_scheduler():
    """Initialize lightweight scheduler (replaces Celery)"""
    if not SCHEDULER_AVAILABLE or not check_memory():
        logger.info('APScheduler skipped (low RAM or not available)')
        return
    
    try:
# Check pending markets every 5 minutes (only if not already running)
        if not SCHEDULER.running:
            SCHEDULER.add_job(
            check_pending,
            'interval',
            minutes=5,
            id='check_pending'
        )
        
        logger.info('APScheduler jobs registered')
    except Exception as e:
        logger.warning(f'Scheduler setup failed: {e}')


app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)