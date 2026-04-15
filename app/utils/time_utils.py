from datetime import datetime, timezone, timedelta
from dateutil import parser

UTC = timezone.utc

def now_utc():
    return datetime.now(UTC)

def parse_timestamp(ts):
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=UTC)
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, UTC)
    if isinstance(ts, str):
        try:
            return parser.parse(ts).replace(tzinfo=UTC)
        except:
            return None
    return None

def format_datetime(dt):
    if dt is None:
        return '-'
    return dt.strftime('%Y-%m-%d %H:%M:%S')

def format_time_remaining(seconds):
    if seconds <= 0:
        return 'Ended'
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f'{minutes}m {secs}s'

def get_market_window_times():
    now = now_utc()
    current_minute = now.minute
    current_second = now.second
    
    window_start = now.replace(second=0, microsecond=0)
    window_end = window_start + timedelta(minutes=5)
    
    elapsed = current_minute % 5 * 60 + current_second
    remaining = 300 - elapsed
    
    return {
        'window_start': window_start,
        'window_end': window_end,
        'elapsed_seconds': elapsed,
        'remaining_seconds': remaining
    }

def extract_time_from_title(title):
    if not title:
        return None, None
    
    import re
    
    patterns = [
        r'(\d{1,2}:\d{2}[AP]M)\s*-\s*(\d{1,2}:\d{2}[AP]M)\s*ET',
        r'(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\s*ET',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, title)
        if match:
            start_str = match.group(1)
            end_str = match.group(2)
            return start_str, end_str
    
    return None, None

def is_market_likely_active(title, current_time=None):
    start_str, end_str = extract_time_from_title(title)
    if not start_str or not end_str:
        return True
    
    return True

def get_hour_of_day(dt=None):
    if dt is None:
        dt = now_utc()
    return dt.hour