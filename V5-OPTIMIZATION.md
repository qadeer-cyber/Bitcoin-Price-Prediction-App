# Polysignal BTC V5 - Optimization Summary

## ✅ What Was Kept
- Flask 3.x, SQLAlchemy 3.x, Flask-Login
- Pandas, NumPy for data
- scikit-learn (ML inference only)
- Telegram Bot (optional)
- SQLite database
- All V1-V4 trading features

## 🗑 What Was Removed (for Low-RAM)
- Celery → APScheduler
- Redis → Flask-Caching with SQLite
- PostgreSQL → SQLite (default)
- Alembic migrations
- WebSockets → REST polling
- 2FA (pyotp)
- bcrypt
- Rate limiting (flask-limiter)
- Flower dashboard

## 🔧 New Optimizations
- Memory guard: `check_memory()` before heavy tasks
- APScheduler: Lightweight background jobs
- Simple cache: SQLite-based caching
- Synchronous only: No async overhead

## 📦 requirements-optimized.txt
Use this file for Python 3.9.13 on low-spec Macs:
```bash
pip install -r requirements-optimized.txt
```

## 🚀 Running
```bash
# Development
python run.py

# Production (with gunicorn)
gunicorn -w 2 run:app
```