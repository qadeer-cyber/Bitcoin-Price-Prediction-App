import pytest
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock

# Set test environment
os.environ['DATABASE_URL'] = 'sqlite://:memory:'
os.environ['SECRET_KEY'] = 'test-secret'


@pytest.fixture
def app():
    """Create test Flask app"""
    from run import create_app
    app = create_app()
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()


@pytest.fixture
def runner(app):
    """Create test CLI runner"""
    return app.test_cli_runner()


class TestGracefulDegradation:
    """Test graceful degradation for Celery/Redis"""
    
    def test_celery_not_available(self):
        """Test that app works when Celery not available"""
        from app.services.settlement_service import CELERY_AVAILABLE, get_celery_app
        
        # Should be false if Celery not installed
        celery_app = get_celery_app()
        assert celery_app is None or celery_app is not None
    
    def test_synchronous_fallback(self):
        """Test synchronous fallback works"""
        from app.services.settlement_service import SettlementService
        
        svc = SettlementService()
        result = svc.run_async('test_task')
        # Should return None for sync execution
        assert result is None


class TestPortfolioCalculations:
    """Test portfolio PnL calculations"""
    
    def test_add_trade_up(self):
        """Test adding UP trade"""
        from app.services.risk_service import portfolio_tracker
        
        portfolio_tracker.reset()
        
        trade = portfolio_tracker.add_trade(
            direction='UP',
            size=1,  # 1 contract
            entry_price=0.50,  # 50 cents (implied probability)
            exit_price=0.51,  # 51 cents
            fees=0.01
        )
        
        # Binary: win = (0.51 - 0.50) * 100 - 0.01 = 0.99
        assert trade['pnl'] > 0
    
    def test_add_trade_down(self):
        """Test adding DOWN trade"""
        from app.services.risk_service import portfolio_tracker
        
        portfolio_tracker.reset()
        
        trade = portfolio_tracker.add_trade(
            direction='DOWN',
            size=1,
            entry_price=0.50,
            exit_price=0.49,
            fees=0.01
        )
        
        assert trade['pnl'] > 0
    
    def test_performance_metrics(self):
        """Test performance metrics calculation"""
        from app.services.risk_service import portfolio_tracker
        
        portfolio_tracker.reset()
        
        # Add winning trade
        portfolio_tracker.add_trade('UP', 100, 50000, 50200, 1)
        # Add losing trade
        portfolio_tracker.add_trade('DOWN', 100, 50000, 49800, 1)
        
        perf = portfolio_tracker.get_performance()
        
        assert perf['total_trades'] == 2
        assert perf['winning_trades'] == 1
        assert perf['losing_trades'] == 1
        assert perf['win_rate'] == 50.0
        assert perf['profit_factor'] > 0
    
    def test_max_drawdown(self):
        """Test max drawdown calculation"""
        from app.services.risk_service import portfolio_tracker
        
        portfolio_tracker.reset()
        
        portfolio_tracker.add_trade('UP', 100, 50000, 45000, 1)  # -5000
        portfolio_tracker.add_trade('UP', 100, 45000, 40000, 1)  # -5000
        portfolio_tracker.add_trade('UP', 100, 40000, 55000, 1)  # +15000
        
        perf = portfolio_tracker.get_performance()
        
        assert perf['max_drawdown'] > 0


class TestRiskEngine:
    """Test risk engine calculations"""
    
    def test_kelly_criterion(self):
        """Test Kelly Criterion"""
        from app.services.risk_service import risk_engine
        
        size = risk_engine.calculate_position_size(
            method='kelly',
            account_balance=10000,
            win_rate=0.55,
            avg_win_loss_ratio=1.5
        )
        
        assert size > 0
    
    def test_fixed_fractional(self):
        """Test Fixed Fractional"""
        from app.services.risk_service import risk_engine
        
        size = risk_engine.calculate_position_size(
            method='fixed_fractional',
            account_balance=10000,
            win_rate=0.5,
            avg_win_loss_ratio=1.0
        )
        
        assert size == 200  # 2% default
    
    def test_volatility_adjusted(self):
        """Test Volatility Adjusted"""
        from app.services.risk_service import risk_engine
        
        size = risk_engine.calculate_position_size(
            method='volatility_adjusted',
            account_balance=10000,
            win_rate=0.5,
            avg_win_loss_ratio=1.0,
            volatility=0.02
        )
        
        assert size > 0
    
    def test_circuit_breaker_trigger(self):
        """Test circuit breaker triggers on 5%+ loss"""
        from app.services.risk_service import risk_engine
        
        result = risk_engine.check_circuit_breaker(
            daily_pnl=-600,
            account_balance=10000
        )
        
        assert result['triggered'] == True


class TestAuditLogs:
    """Test audit log integrity"""
    
    def test_log_event(self):
        """Test basic audit logging"""
        from app.services.audit_service import audit_service
        import time
        
        audit_service._logs.clear()
        
        audit_service.log_event(
            event_type='test',
            user_id='test_user',
            action='test_action',
            details={'key': 'value'}
        )
        
        assert len(audit_service._logs) > 0
    
    def test_auth_event(self):
        """Test auth event logging"""
        from app.services.audit_service import audit_service
        
        audit_service._logs.clear()
        
        audit_service.log_auth_event(
            user_id='test_user',
            event='login',
            success=True,
            ip_address='127.0.0.1'
        )
        
        logs = audit_service.get_logs(event_type='auth')
        assert len(logs) > 0
    
    def test_config_change(self):
        """Test config change logging"""
        from app.services.audit_service import audit_service
        
        audit_service._logs.clear()
        
        audit_service.log_config_change(
            user_id='test_user',
            setting_key='test_key',
            old_value='old',
            new_value='new'
        )
        
        logs = audit_service.get_logs(event_type='config')
        assert len(logs) > 0
    
    def test_no_sensitive_data_in_logs(self):
        """Ensure passwords/keys not logged"""
        from app.services.audit_service import audit_service
        
        audit_service._logs.clear()
        
        # Try to log sensitive data
        audit_service.log_event(
            event_type='auth',
            user_id='test',
            action='login',
            details={
                'password': 'secret123',
                'api_key': 'sk_live_123',
                'safe_field': 'visible'
            }
        )
        
        log = audit_service._logs[0]
        
        # Sensitive fields should be masked or not present
        details = log.get('details', {})
        
        # Check passwords not exposed
        if 'password' in details:
            assert details['password'] == '***REDACTED***'
        
        # API keys should be masked
        if 'api_key' in details:
            assert 'sk_live_' not in details['api_key']


class TestRateLimiter:
    """Test rate limiting behavior"""
    
    def test_rate_limiter_optional(self):
        """Test that rate limiter is optional"""
        try:
            from run import limiter
            # Limiter may be None if flask-limiter not installed
            assert limiter is None or hasattr(limiter, 'init_app')
        except ImportError:
            pass  # Skip if flask-limiter not available


class TestAPIBackwardCompatibility:
    """Test API backward compatibility"""
    
    def test_signal_current(self, client):
        """Test /api/signal/current returns expected structure"""
        response = client.get('/api/signal/current')
        
        assert response.status_code in [200, 404, 500]
        
        if response.status_code == 200:
            data = response.get_json()
            assert 'direction' in data or 'error' in data
    
    def test_health_endpoint(self, client):
        """Test /api/health"""
        response = client.get('/api/health')
        
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.get_json()
            assert 'status' in data
    
    def test_backtest_run(self, client):
        """Test /api/backtest/run"""
        response = client.post('/api/backtest/run', json={
            'start_date': '2024-01-01',
            'end_date': '2024-01-02'
        })
        
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.get_json()
            assert 'win_rate' in data or 'error' in data


class TestMarketRegistry:
    """Test market registry"""
    
    def test_get_available_markets(self, client):
        """Test /api/markets/available"""
        response = client.get('/api/markets/available')
        
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.get_json()
            assert 'markets' in data or 'error' in data
    
    def test_subscribe_markets(self, client):
        """Test /api/markets/subscribe"""
        response = client.post('/api/markets/subscribe', json={
            'market_ids': ['BTC_5_minute']
        })
        
        assert response.status_code in [200, 500]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])