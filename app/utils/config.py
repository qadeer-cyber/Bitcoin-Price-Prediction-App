import os
from datetime import timedelta

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'poly-signal-btc-2026-secret')
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///polysignal.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Polymarket API
    POLYMARKET_GAMMA_API = 'https://gamma-api.polymarket.com'
    POLYMARKET_CLOB_API = 'https://clob.polymarket.com'
    
    # Binance API (supplementary BTC price)
    BINANCE_API = 'https://api.binance.com/api/v3'
    
    # Signal refresh interval (seconds)
    SIGNAL_REFRESH_INTERVAL = 5
    
    # Default strategy settings
    DEFAULT_EMA_SHORT = 9
    DEFAULT_EMA_MEDIUM = 20
    DEFAULT_EMA_LONG = 50
    DEFAULT_RSI_PERIOD = 14
    DEFAULT_RSI_OVERBOUGHT = 70
    DEFAULT_RSI_OVERSOLD = 30
    DEFAULT_MACD_FAST = 12
    DEFAULT_MACD_SLOW = 26
    DEFAULT_MACD_SIGNAL = 9
    DEFAULT_ATR_PERIOD = 14
    
    # Confidence thresholds
    DEFAULT_CONFIDENCE_THRESHOLD = 50
    DEFAULT_NO_TRADE_AGGRESSIVENESS = 60
    
    # Risk settings
    DEFAULT_SPREAD_WARNING_THRESHOLD = 0.05
    DEFAULT_MIN_LIQUIDITY_WARNING = 1000
    DEFAULT_FINAL_MINUTE_PENALTY = 20
    
    # Market window
    MARKET_WINDOW_MINUTES = 5