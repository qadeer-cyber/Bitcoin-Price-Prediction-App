# PolySignal BTC

Polymarket BTC 5-Minute Market Intelligence Platform

## Overview

PolySignal BTC is a premium Flask web application for analyzing and tracking Polymarket's BTC 5-minute "Up or Down" prediction markets. It provides real-time signal generation, market tracking, and advanced analytics.

## Features

- **Active Market Detection**: Tracks current Polymarket BTC 5-minute markets
- **Signal Generation**: Weighted confluence signal engine using:
  - Polymarket-specific factors (price distance, time remaining, market probability)
  - BTC microstructure (EMA, RSI, MACD, Bollinger Bands)
  - Risk filters (regime detection, late entry warnings)
- **Signal Resolution Tracking**: Automatic evaluation of signal correctness
- **Advanced Analytics**: Daily stats, confidence buckets, hourly heatmap, regime analysis
- **Premium Dashboard**: Dark luxury institutional UI with ApexCharts

## Tech Stack

- Python 3.11+
- Flask 3.x
- SQLAlchemy (SQLite)
- Bootstrap 5 (dark theme)
- ApexCharts

## Prerequisites

```
Python 3.11+
pip
```

## Installation

```bash
# Clone or navigate to project directory
cd Bitcoin-Price-Prediction-App

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Application

```bash
# Start the Flask application
python app.py

# Open browser to http://localhost:5000
```

## Data Sources

- **Resolution**: Chainlink BTC/USD data stream (per Polymarket rules)
- **Supplementary**: Binance BTC/USDT for real-time reference
- **Market Data**: Polymarket Gamma API

## Important Notes

1. **Not a Trading Bot**: This is a decision-support and analytics platform only.
2. **No Guaranteed Accuracy**: Signals are generated based on technical confluence - results will vary.
3. **Market Resolution**: Polymarket resolves based on Chainlink BTC/USD, not exchange close prices.
4. **Supplementary Data**: Binance price is used as supplementary reference only.

## Project Structure

```
/app
├── routes/
│   ├── dashboard.py    # Main pages
│   └── api.py          # JSON API endpoints
├── services/
│   ├── polymarket_service.py    # Market discovery
│   ├── chainlink_service.py     # Price feeds
│   ├── strategy_service.py     # Signal engine
│   ├── analytics_service.py    # Analytics
│   └── settlement_service.py   # Resolution tracking
├── models/
│   └── db.py           # SQLAlchemy models
├── templates/         # Jinja2 templates
├── static/            # CSS/JS
└── utils/             # Helper utilities
app.py                 # Flask application
requirements.txt      # Python dependencies
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/api/market/current` | Current active market |
| `/api/market/history` | Historical markets |
| `/api/signal/current` | Current signal |
| `/api/signals/history` | Signal logs |
| `/api/analytics/today` | Today's stats |
| `/api/analytics/rolling` | Rolling accuracy |
| `/api/settings` | Get/Update settings |
| `/api/health` | Health check |

## Pages

- **Dashboard** (`/`): Active market, current signal, today's performance
- **History** (`/history`): Filterable signal log table
- **Analytics** (`/analytics`): Charts, accuracy breakdown, PnL simulation
- **Settings** (`/settings`): Strategy parameters configuration

## Known Limitations

1. Uses Binance as supplementary BTC price (Chainlink direct API requires specific access)
2. Historical Polymarket 5-minute data may be limited - includes synthetic fallback
3. No actual trading - analytics platform only

## Disclaimer

This software is provided for educational and informational purposes only. It does not constitute financial advice. Past performance does not guarantee future results. Always conduct your own research before making investment decisions.