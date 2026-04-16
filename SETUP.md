# Polysignal BTC V5 - Setup for macOS 10.13

## Exact Setup Commands

### 1. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements-optimized.txt
```

### 3. Run Development Server
```bash
python run.py
```

### 4. Open in Browser
```
http://localhost:5000
```

## Troubleshooting

### Memory Issues
If you see "Memory high, pausing heavy jobs":
- Close other applications
- Reduce browser tabs
- Check Activity Monitor

### Port Already in Use
```bash
lsof -i :5000 | grep python
kill <PID>
```

### First Run - Train ML Model
```bash
curl -X POST http://localhost:5000/api/ml/train \
  -H "Content-Type: application/json" \
  -d '{"min_samples": 50}'
```

## Background Jobs
The app uses APScheduler for:
- Checking pending market resolutions every 5 minutes
- Auto-resumes if memory normalizes

## Performance Tips
- Use Safari 13+ for best compatibility
- Disable browser extensions if slow
- Close unused tabs

## Production Mode (Optional)
```bash
gunicorn -w 2 run:app
```