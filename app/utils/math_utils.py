import math
from typing import List, Optional

def calculate_ema(prices: List[float], period: int) -> Optional[float]:
    if len(prices) < period:
        return None
    
    ema = sum(prices[:period]) / period
    multiplier = 2 / (period + 1)
    
    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema
    
    return ema

def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
    if len(prices) < period + 1:
        return None
    
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9):
    if len(prices) < slow:
        return None, None, None
    
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    
    if ema_fast is None or ema_slow is None:
        return None, None, None
    
    macd_line = ema_fast - ema_slow
    
    macd_values = []
    temp_prices = prices[:slow]
    
    for i in range(slow, len(prices)):
        ef = calculate_ema(prices[:i+1], fast)
        es = calculate_ema(prices[:i+1], slow)
        if ef and es:
            macd_values.append(ef - es)
    
    if len(macd_values) < signal:
        signal_line = macd_line
    else:
        signal_line = calculate_ema(macd_values, signal)
    
    histogram = macd_line - signal_line if signal_line else None
    
    return macd_line, signal_line, histogram

def calculate_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    
    trs = []
    for i in range(1, len(closes)):
        high_low = highs[i] - lows[i]
        high_close = abs(highs[i] - closes[i-1])
        low_close = abs(lows[i] - closes[i-1])
        tr = max(high_low, high_close, low_close)
        trs.append(tr)
    
    if len(trs) < period:
        return None
    
    atr = sum(trs[-period:]) / period
    return atr

def calculate_bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2.0):
    if len(prices) < period:
        return None, None, None
    
    recent = prices[-period:]
    sma = sum(recent) / period
    
    variance = sum((p - sma) ** 2 for p in recent) / period
    std = math.sqrt(variance)
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    
    return upper, sma, lower

def calculate_price_momentum(prices: List[float], period: int = 1) -> float:
    if len(prices) < period + 1:
        return 0.0
    
    current = prices[-1]
    past = prices[-period-1]
    
    return ((current - past) / past) * 100 if past != 0 else 0.0

def calculate_candle_strength(opens: List[float], closes: List[float], highs: List[float], lows: List[float]) -> float:
    if len(closes) < 1:
        return 0.0
    
    last_close = closes[-1]
    last_open = opens[-1] if len(opens) >= 1 else last_close
    last_high = highs[-1] if len(highs) >= 1 else last_close
    last_low = lows[-1] if len(lows) >= 1 else last_close
    
    if last_high == last_low:
        return 0.0
    
    body = abs(last_close - last_open) / (last_high - last_low)
    direction = 1 if last_close >= last_open else -1
    
    return direction * body

def normalize_value(value: float, min_val: float, max_val: float) -> float:
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)

def weighted_confluence(factors: dict, weights: dict) -> float:
    total_weight = 0
    weighted_sum = 0
    
    for factor, value in factors.items():
        weight = weights.get(factor, 1.0)
        weighted_sum += value * weight
        total_weight += weight
    
    if total_weight == 0:
        return 0
    
    return weighted_sum / total_weight