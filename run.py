import os
import sys
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from flask_login import LoginManager
from datetime import datetime, timezone

from app.models.db import db, Settings, User
from app.routes import dashboard, api
from app.routes.auth import auth_bp
from app.routes.websocket import ws_bp

login_manager = LoginManager()
login_manager.login_view = 'dashboard.login'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__, template_folder='app/templates', static_folder='app/static')
    
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'poly-signal-btc-2026-secret')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///polysignal.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    db.init_app(app)
    login_manager.init_app(app)
    
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.filter_by(user_id=user_id).first()
    
    with app.app_context():
        db.create_all()
        
        _init_default_settings()
    
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


app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)