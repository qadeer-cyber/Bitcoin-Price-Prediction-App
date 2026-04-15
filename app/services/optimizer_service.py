import logging
import itertools
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.services.backtest_service import EnhancedBacktestService, enhanced_backtest_service

logger = logging.getLogger(__name__)


class StrategyOptimizer:
    """Grid search optimizer for strategy parameters"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self._results = []
    
    OPTIMIZATION_PARAMS = {
        'min_confidence_weak': [40, 45, 50, 55, 60],
        'min_confidence_moderate': [55, 60, 65, 70, 75],
        'min_confidence_strong': [70, 75, 80, 85, 90],
        'min_confidence_elite': [85, 90, 95],
        'max_spread': [1.0, 1.5, 2.0, 2.5, 3.0],
        'min_volume': [1000, 2500, 5000, 10000],
        'max_orderbook_imbalance': [60, 70, 80, 90],
        'max_time_remaining': [120, 180, 240, 300],
    }
    
    def grid_search(
        self,
        start_date: str,
        end_date: str,
        params_to_optimize: Optional[List[str]] = None,
        metric: str = 'win_rate',
        use_synthetic: bool = True,
        initial_balance: float = 10000,
        stake_per_trade: float = 100
    ) -> Dict:
        """Run grid search optimization
        
        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            params_to_optimize: List of param names to optimize (default: all)
            metric: Metric to optimize (win_rate, expectancy, roi, max_drawdown)
            use_synthetic: Use synthetic data
            initial_balance: Initial balance
            stake_per_trade: Stake per trade
            
        Returns:
            Best params and all results
        """
        if params_to_optimize is None:
            params_to_optimize = ['min_confidence_weak', 'min_confidence_moderate', 'max_spread']
        
        param_grid = {
            k: self.OPTIMIZATION_PARAMS[k]
            for k in params_to_optimize
            if k in self.OPTIMIZATION_PARAMS
        }
        
        combinations = self._generate_combinations(param_grid)
        
        logger.info(f'Starting grid search with {len(combinations)} combinations')
        
        best_result = None
        best_metric_val = float('-inf') if metric in ['win_rate', 'expectancy', 'roi'] else float('inf')
        all_results = []
        
        for combo in combinations:
            result = self._run_single_backtest(
                start_date, end_date, combo,
                use_synthetic, initial_balance, stake_per_trade
            )
            
            metric_val = result.get(metric, 0)
            combo_results = {**combo, **result, 'metric': metric_val}
            all_results.append(combo_results)
            
            is_better = (
                metric in ['win_rate', 'expectancy', 'roi'] and metric_val > best_metric_val
            ) or (
                metric == 'max_drawdown' and metric_val < best_metric_val
            )
            
            if is_better:
                best_result = combo_results
                best_metric_val = metric_val
        
        all_results.sort(
            key=lambda x: x[metric],
            reverse=metric in ['win_rate', 'expectancy', 'roi']
        )
        
        return {
            'best_params': best_result,
            'best_metric': metric,
            'best_value': best_metric_val,
            'total_combinations': len(combinations),
            'results': all_results[:50]
        }
    
    def _generate_combinations(self, param_grid: Dict) -> List[Dict]:
        """Generate all parameter combinations"""
        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def _run_single_backtest(
        self,
        start_date: str,
        end_date: str,
        params: Dict,
        use_synthetic: bool,
        initial_balance: float,
        stake_per_trade: float
    ) -> Dict:
        """Run single backtest with given params"""
        
        market_data = enhanced_backtest_service._generate_synthetic_markets(
            datetime.fromisoformat(start_date),
            datetime.fromisoformat(end_date)
        )
        
        correct = 0
        wrong = 0
        no_trades = 0
        total_pnl = 0
        
        min_conf = params.get('min_confidence_weak', 50)
        max_spread = params.get('max_spread', 2.0)
        min_vol = params.get('min_volume', 5000)
        max_imbalance = params.get('max_orderbook_imbalance', 80)
        
        for market in market_data:
            price_to_beat = market.get('price_to_beat')
            up_prob = market.get('up_probability', 0.5)
            
            if price_to_beat and up_prob:
                confidence = int(min(95, up_prob * 100))
                
                if confidence < min_conf:
                    no_trades += 1
                    continue
                
                if market.get('volume', 0) < min_vol:
                    no_trades += 1
                    continue
                
                simulated_price = market.get('final_price', price_to_beat)
                actual_outcome = 'UP' if simulated_price >= price_to_beat else 'DOWN'
                predicted = 'UP' if up_prob > 0.5 else 'DOWN'
                
                is_correct = predicted == actual_outcome
                
                if is_correct:
                    correct += 1
                    payout = stake_per_trade / up_prob if up_prob > 0.01 else stake_per_trade
                    total_pnl += payout - stake_per_trade
                else:
                    wrong += 1
                    total_pnl -= stake_per_trade
        
        total = correct + wrong
        win_rate = (correct / total * 100) if total > 0 else 0
        roi = ((total_pnl / initial_balance) * 100) if initial_balance > 0 else 0
        
        return {
            'signals_generated': correct + wrong + no_trades,
            'correct': correct,
            'wrong': wrong,
            'no_trades': no_trades,
            'win_rate': win_rate,
            'net_pnl': total_pnl,
            'roi': roi
        }
    
    def quick_optimize(
        self,
        start_date: str,
        end_date: str,
        metric: str = 'win_rate'
    ) -> Dict:
        """Quick optimization with reduced parameter space"""
        params_to_optimize = ['min_confidence_weak', 'max_spread']
        
        return self.grid_search(
            start_date=start_date,
            end_date=end_date,
            params_to_optimize=params_to_optimize,
            metric=metric
        )


strategy_optimizer = StrategyOptimizer()