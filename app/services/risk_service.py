import logging
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, List
from collections import deque

logger = logging.getLogger(__name__)


class RiskEngine:
    """Professional-grade risk management engine"""
    
    def __init__(self):
        self._max_daily_loss_pct = 0.05
        self._circuit_breaker_triggered = False
        self._max_position_size_pct = 0.25
        self._min_kelly_fraction = 0.01
        self._max_correlation = 0.7
    
    def calculate_position_size(
        self,
        method: str,
        account_balance: float,
        win_rate: float,
        avg_win_loss_ratio: float,
        volatility: float = 0.02
    ) -> float:
        """Calculate position size based on method
        
        Args:
            method: kelly, fixed_fractional, or volatility_adjusted
            account_balance: Current account size
            win_rate: Historical win rate (0-1)
            avg_win_loss_ratio: Average win / average loss
            volatility: Current market volatility
        """
        if method == 'kelly':
            return self._kelly_criterion(account_balance, win_rate, avg_win_loss_ratio)
        elif method == 'volatility_adjusted':
            return self._volatility_adjusted(account_balance, volatility)
        else:
            return self._fixed_fractional(account_balance)
    
    def _kelly_criterion(
        self,
        account_balance: float,
        win_rate: float,
        win_loss_ratio: float
    ) -> float:
        """Kelly Criterion position sizing
        
        f* = (bp - q) / b
        where:
            b = odds received (win/loss ratio)
            p = probability of win
            q = probability of loss (1-p)
        """
        q = 1 - win_rate
        b = win_loss_ratio
        
        if b <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0
        
        kelly_pct = (b * win_rate - q) / b
        
        if kelly_pct <= 0:
            return 0
        
        kelly_pct = min(kelly_pct, self._max_position_size_pct)
        
        kelly_pct = max(kelly_pct, self._min_kelly_fraction)
        
        return account_balance * kelly_pct
    
    def _fixed_fractional(self, account_balance: float, fraction: float = 0.02) -> float:
        """Fixed fractional position sizing"""
        return account_balance * fraction
    
    def _volatility_adjusted(
        self,
        account_balance: float,
        volatility: float,
        target_risk: float = 0.02
    ) -> float:
        """Volatility-adjusted position sizing"""
        if volatility <= 0:
            return self._fixed_fractional(account_balance)
        
        vol_adjusted = target_risk / volatility
        vol_adjusted = min(vol_adjusted, self._max_position_size_pct)
        
        return account_balance * vol_adjusted
    
    def check_circuit_breaker(
        self,
        daily_pnl: float,
        account_balance: float
    ) -> Dict:
        """Check if circuit breaker should trigger
        
        Triggers if daily loss exceeds max_daily_loss_pct
        """
        daily_loss_pct = -daily_pnl / account_balance if account_balance > 0 else 0
        
        if daily_loss_pct >= self._max_daily_loss_pct:
            self._circuit_breaker_triggered = True
            return {
                'triggered': True,
                'daily_loss_pct': daily_loss_pct,
                'threshold': self._max_daily_loss_pct,
                'message': f'Circuit breaker triggered: {daily_loss_pct:.1%} loss'
            }
        
        return {
            'triggered': False,
            'daily_loss_pct': daily_loss_pct
        }
    
    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker at start of new day"""
        self._circuit_breaker_triggered = False
    
    def calculate_var(
        self,
        returns: List[float],
        confidence: float = 0.95
    ) -> float:
        """Calculate Value at Risk
        
        Args:
            returns: List of historical returns
            confidence: Confidence level (e.g., 0.95 for 95%)
        """
        if not returns:
            return 0
        
        sorted_returns = sorted(returns)
        index = int((1 - confidence) * len(sorted_returns))
        
        return abs(sorted_returns[index]) if index < len(sorted_returns) else 0
    
    def check_correlation_risk(
        self,
        positions: List[Dict],
        new_position: Dict
    ) -> Dict:
        """Check correlation risk for new position
        
        Prevents overexposure to correlated markets
        """
        if len(positions) < 1:
            return {'allowed': True, 'correlation': 0}
        
        asset_class = new_position.get('asset', 'BTC')
        
        same_asset_count = sum(
            1 for p in positions 
            if p.get('asset') == asset_class
        )
        
        if same_asset_count >= 3:
            return {
                'allowed': False,
                'reason': f'Max positions in {asset_class} reached',
                'correlation': 1.0
            }
        
        return {'allowed': True, 'correlation': 0}
    
    def get_risk_limits(self) -> Dict:
        """Get current risk limits"""
        return {
            'max_daily_loss_pct': self._max_daily_loss_pct,
            'max_position_size_pct': self._max_position_size_pct,
            'min_kelly_fraction': self._min_kelly_fraction,
            'max_correlation': self._max_correlation,
            'circuit_breaker_triggered': self._circuit_breaker_triggered
        }
    
    def set_limits(
        self,
        max_daily_loss_pct: float = None,
        max_position_size_pct: float = None
    ) -> None:
        """Update risk limits"""
        if max_daily_loss_pct:
            self._max_daily_loss_pct = max_daily_loss_pct
        if max_position_size_pct:
            self._max_position_size_pct = max_position_size_pct


class PortfolioTracker:
    """Track simulated portfolio performance"""
    
    def __init__(self):
        self._trades = []
        self._equity_curve = deque(maxlen=10000)
        self._starting_balance = 10000
        self._current_balance = 10000
    
    def add_trade(
        self,
        direction: str,
        size: float,
        entry_price: float,
        exit_price: float,
        fees: float = 0
    ) -> Dict:
        """Add a trade to portfolio
        
        Returns trade result with PnL
        """
        if direction == 'UP':
            pnl = (exit_price - entry_price) * size
        else:
            pnl = (entry_price - exit_price) * size
        
        pnl -= fees
        
        trade = {
            'direction': direction,
            'size': size,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'fees': fees,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        self._trades.append(trade)
        self._current_balance += pnl
        
        self._equity_curve.append({
            'balance': self._current_balance,
            'timestamp': trade['timestamp']
        })
        
        return trade
    
    def get_equity_curve(self, days: int = None) -> List[Dict]:
        """Get equity curve data"""
        if days:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            return [
                e for e in self._equity_curve
                if datetime.fromisoformat(e['timestamp']) > cutoff
            ]
        return list(self._equity_curve)
    
    def get_performance(self) -> Dict:
        """Get portfolio performance metrics"""
        if not self._trades:
            return {'error': 'No trades'}
        
        total_pnl = sum(t['pnl'] for t in self._trades)
        winning_trades = [t for t in self._trades if t['pnl'] > 0]
        losing_trades = [t for t in self._trades if t['pnl'] <= 0]
        
        wins = len(winning_trades)
        losses = len(losing_trades)
        total = wins + losses
        
        win_rate = (wins / total * 100) if total > 0 else 0
        
        avg_win = (
            sum(t['pnl'] for t in winning_trades) / wins
            if wins > 0 else 0
        )
        avg_loss = (
            abs(sum(t['pnl'] for t in losing_trades) / losses)
            if losses > 0 else 0
        )
        
        returns = [t['pnl'] / self._starting_balance for t in self._trades]
        
        max_dd = self._calculate_max_drawdown()
        sharpe = self._calculate_sharpe_ratio(returns)
        sortino = self._calculate_sortino_ratio(returns)
        
        return {
            'total_trades': total,
            'winning_trades': wins,
            'losing_trades': losses,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss > 0 else 0,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'current_balance': self._current_balance,
            'starting_balance': self._starting_balance,
            'roi': ((self._current_balance - self._starting_balance) / self._starting_balance * 100)
        }
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if not self._equity_curve:
            return 0
        
        balances = [e['balance'] for e in self._equity_curve]
        peak = balances[0]
        max_dd = 0
        
        for balance in balances:
            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        return max_dd * 100
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0
        
        mean_return = sum(returns) / len(returns)
        std_return = (sum((r - mean_return) ** 2 for r in returns) / len(returns)) ** 0.5
        
        if std_return == 0:
            return 0
        
        return (mean_return / std_return) * (252 ** 0.5)
    
    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """Calculate Sortino ratio (uses downside deviation)"""
        if len(returns) < 2:
            return 0
        
        mean_return = sum(returns) / len(returns)
        downside_returns = [r for r in returns if r < 0]
        
        if not downside_returns:
            return 0
        
        downside_std = (
            sum(r ** 2 for r in downside_returns) / len(downside_returns)
        ) ** 0.5
        
        if downside_std == 0:
            return 0
        
        return (mean_return / downside_std) * (252 ** 0.5)
    
    def reset(self) -> None:
        """Reset portfolio"""
        self._trades = []
        self._equity_curve.clear()
        self._current_balance = self._starting_balance


risk_engine = RiskEngine()
portfolio_tracker = PortfolioTracker()