def format_price(price: float, decimals: int = 2) -> str:
    if price is None:
        return '-'
    return f'${price:,.{decimals}f}'

def format_percentage(value: float, decimals: int = 1) -> str:
    if value is None:
        return '-'
    return f'{value * 100:.{decimals}f}%'

def format_number(num: float, decimals: int = 0) -> str:
    if num is None:
        return '-'
    if abs(num) >= 1_000_000:
        return f'{num/1_000_000:.{decimals}f}M'
    elif abs(num) >= 1_000:
        return f'{num/1_000:.{decimals}f}K'
    return f'{num:.{decimals}f}'

def format_confidence(confidence: int) -> str:
    if confidence >= 90:
        return f'{confidence} (Elite)'
    elif confidence >= 80:
        return f'{confidence} (Strong)'
    elif confidence >= 70:
        return f'{confidence} (Moderate)'
    elif confidence >= 50:
        return f'{confidence} (Weak)'
    else:
        return 'NO TRADE'

def format_outcome(outcome: str) -> str:
    if outcome is None:
        return 'Pending'
    return outcome.upper()

def format_correctness(is_correct: bool) -> str:
    if is_correct is None:
        return 'Pending'
    return '✓ Correct' if is_correct else '✗ Wrong'

def format_streak(streak: int) -> str:
    if streak == 0:
        return '0'
    sign = '+' if streak > 0 else '-'
    return f'{sign}{abs(streak)}'