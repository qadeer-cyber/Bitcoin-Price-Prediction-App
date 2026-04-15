from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import uuid
import secrets

db = SQLAlchemy()


class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(36), unique=True, default=lambda: str(uuid.uuid4()))
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    api_key = db.Column(db.String(64), unique=True)
    is_active = db.Column(db.Boolean, default=True)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)
    
    def verify_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)
    
    def generate_api_key(self) -> str:
        self.api_key = secrets.token_hex(32)
        return self.api_key
    
    def get_id(self):
        return self.user_id
    
    def to_dict(self, include_private=False):
        data = {
            'user_id': self.user_id,
            'username': self.username,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
        if include_private:
            data['email'] = self.email
            data['api_key'] = self.api_key
            data['is_admin'] = self.is_admin
            data['last_login'] = self.last_login.isoformat() if self.last_login else None
        return data


class StrategyProfile(db.Model):
    __tablename__ = 'strategy_profiles'
    
    id = db.Column(db.Integer, primary_key=True)
    profile_id = db.Column(db.String(36), unique=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    name = db.Column(db.String(50), nullable=False)
    
    user = db.relationship('User', backref='profiles')
    description = db.Column(db.String(500))
    is_active = db.Column(db.Boolean, default=False)
    
    # Confidence thresholds
    min_confidence_weak = db.Column(db.Integer, default=50)
    min_confidence_moderate = db.Column(db.Integer, default=60)
    min_confidence_strong = db.Column(db.Integer, default=75)
    min_confidence_elite = db.Column(db.Integer, default=90)
    
    # Market filters
    max_spread = db.Column(db.Float, default=2.0)
    min_volume = db.Column(db.Float, default=5000)
    max_orderbook_imbalance = db.Column(db.Float, default=80)
    
    # Regime settings
    allow_trending = db.Column(db.Boolean, default=True)
    allow_sideways = db.Column(db.Boolean, default=True)
    allow_breakout = db.Column(db.Boolean, default=True)
    allow_whipsaw = db.Column(db.Boolean, default=False)
    
    # Timing settings
    max_time_remaining = db.Column(db.Integer, default=300)
    early_exit_enabled = db.Column(db.Boolean, default=False)
    early_exit_threshold = db.Column(db.Float, default=0.80)
    
    # Alert settings
    alerts_enabled = db.Column(db.Boolean, default=True)
    alert_telegram = db.Column(db.Boolean, default=True)
    alert_discord = db.Column(db.Boolean, default=False)
    alert_email = db.Column(db.Boolean, default=False)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'profile_id': self.profile_id,
            'user_id': self.user_id,
            'name': self.name,
            'description': self.description,
            'is_active': self.is_active,
            'thresholds': {
                'min_confidence_weak': self.min_confidence_weak,
                'min_confidence_moderate': self.min_confidence_moderate,
                'min_confidence_strong': self.min_confidence_strong,
                'min_confidence_elite': self.min_confidence_elite,
            },
            'filters': {
                'max_spread': self.max_spread,
                'min_volume': self.min_volume,
                'max_orderbook_imbalance': self.max_orderbook_imbalance,
            },
            'regimes': {
                'allow_trending': self.allow_trending,
                'allow_sideways': self.allow_sideways,
                'allow_breakout': self.allow_breakout,
                'allow_whipsaw': self.allow_whipsaw,
            },
            'timing': {
                'max_time_remaining': self.max_time_remaining,
                'early_exit_enabled': self.early_exit_enabled,
                'early_exit_threshold': self.early_exit_threshold,
            },
            'alerts': {
                'enabled': self.alerts_enabled,
                'telegram': self.alert_telegram,
                'discord': self.alert_discord,
                'email': self.alert_email,
            },
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class MarketSnapshot(db.Model):
    __tablename__ = 'market_snapshots'
    
    id = db.Column(db.Integer, primary_key=True)
    snapshot_id = db.Column(db.String(36), unique=True, default=lambda: str(uuid.uuid4()))
    market_id = db.Column(db.String(100), nullable=False, index=True)
    event_title = db.Column(db.String(500))
    window_start = db.Column(db.DateTime, nullable=False)
    window_end = db.Column(db.DateTime, nullable=False)
    price_to_beat = db.Column(db.Float)
    live_price = db.Column(db.Float)
    up_probability = db.Column(db.Float)
    down_probability = db.Column(db.Float)
    status = db.Column(db.String(20), default='live')
    volume_usd = db.Column(db.Float)
    snapshot_time = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'snapshot_id': self.snapshot_id,
            'market_id': self.market_id,
            'event_title': self.event_title,
            'window_start': self.window_start.isoformat() if self.window_start else None,
            'window_end': self.window_end.isoformat() if self.window_end else None,
            'price_to_beat': self.price_to_beat,
            'live_price': self.live_price,
            'up_probability': self.up_probability,
            'down_probability': self.down_probability,
            'status': self.status,
            'volume_usd': self.volume_usd,
            'snapshot_time': self.snapshot_time.isoformat() if self.snapshot_time else None
        }

class SignalLog(db.Model):
    __tablename__ = 'signal_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    signal_id = db.Column(db.String(36), unique=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('users.user_id'), nullable=True, index=True)
    market_id = db.Column(db.String(100), nullable=False, index=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    signal_direction = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Integer, default=0)
    confidence_tier = db.Column(db.String(20))
    price_to_beat = db.Column(db.Float)
    live_price_at_signal = db.Column(db.Float)
    reasoning = db.Column(db.Text)
    regime = db.Column(db.String(20))
    market_probability_up = db.Column(db.Float)
    resolved_outcome = db.Column(db.String(10))
    is_correct = db.Column(db.Boolean)
    time_remaining_at_signal = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'signal_id': self.signal_id,
            'market_id': self.market_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'signal_direction': self.signal_direction,
            'confidence': self.confidence,
            'confidence_tier': self.confidence_tier,
            'price_to_beat': self.price_to_beat,
            'live_price_at_signal': self.live_price_at_signal,
            'reasoning': self.reasoning,
            'regime': self.regime,
            'market_probability_up': self.market_probability_up,
            'resolved_outcome': self.resolved_outcome,
            'is_correct': self.is_correct,
            'time_remaining_at_signal': self.time_remaining_at_signal,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Settings(db.Model):
    __tablename__ = 'settings'
    
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(50), unique=True, nullable=False)
    value = db.Column(db.String(500))
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    @staticmethod
    def get_value(key, default=None):
        setting = Settings.query.filter_by(key=key).first()
        if setting is None:
            return default
        
        try:
            return int(setting.value)
        except ValueError:
            try:
                return float(setting.value)
            except ValueError:
                return setting.value
    
    @staticmethod
    def set_value(key, value):
        setting = Settings.query.filter_by(key=key).first()
        if setting is None:
            setting = Settings(key=key, value=str(value))
            db.session.add(setting)
        else:
            setting.value = str(value)
        db.session.commit()

class ResolvedMarket(db.Model):
    __tablename__ = 'resolved_markets'
    
    id = db.Column(db.Integer, primary_key=True)
    market_id = db.Column(db.String(100), unique=True, nullable=False)
    event_title = db.Column(db.String(500))
    window_start = db.Column(db.DateTime)
    window_end = db.Column(db.DateTime)
    price_to_beat = db.Column(db.Float)
    final_price = db.Column(db.Float)
    outcome = db.Column(db.String(10))
    resolved_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'market_id': self.market_id,
            'event_title': self.event_title,
            'window_start': self.window_start.isoformat() if self.window_start else None,
            'window_end': self.window_end.isoformat() if self.window_end else None,
            'price_to_beat': self.price_to_beat,
            'final_price': self.final_price,
            'outcome': self.outcome,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }