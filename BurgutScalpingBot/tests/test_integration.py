import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import json
import os


class TestBotIntegration:
    """
    To'liq bot integration testlari - barcha modullarni birgalikda test qilish
    """
    
    @pytest.fixture
    def mock_config(self):
        return {
            'trading': {
                'pair': 'BTC/USDT',
                'timeframe': '1m',
                'position_size': 0.01,
                'max_positions': 3
            },
            'risk': {
                'max_drawdown': 0.05,
                'stop_loss': 0.02,
                'take_profit': 0.03
            },
            'ai': {
                'model_path': 'models/scalping_model.pkl',
                'confidence_threshold': 0.7
            }
        }
    
    @pytest.fixture
    def mock_market_data(self):
        """Mock market data generator"""
        return {
            'ohlcv': [
                [1640995200000, 47000, 47100, 46900, 47050, 1.5],
                [1640995260000, 47050, 47150, 46950, 47100, 1.8],
                [1640995320000, 47100, 47200, 47000, 47150, 2.1],
                [1640995380000, 47150, 47250, 47050, 47200, 1.9],
                [1640995440000, 47200, 47300, 47100, 47250, 2.3]
            ],
            'orderbook': {
                'bids': [[47200, 1.5], [47190, 2.1], [47180, 1.8]],
                'asks': [[47210, 1.2], [47220, 1.9], [47230, 2.2]]
            },
            'trades': [
                {'price': 47205, 'amount': 0.5, 'side': 'buy', 'timestamp': 1640995500000},
                {'price': 47195, 'amount': 0.8, 'side': 'sell', 'timestamp': 1640995505000}
            ]
        }
    
    @pytest.mark.asyncio
    async def test_full_trading_cycle(self, mock_config, mock_market_data):
        """To'liq trading cycle testi"""
        with patch.multiple(
            'trading.strategy_manager.StrategyManager',
            get_market_data=AsyncMock(return_value=mock_market_data),
            place_order=AsyncMock(return_value={'id': 'order123', 'status': 'filled'}),
            get_balance=AsyncMock(return_value={'BTC': 0.5, 'USDT': 1000})
        ):
            # Bot initialization
            from trading.strategy_manager import StrategyManager
            from ai.model_monitor import AIModelMonitor
            from monitoring.telegram_reporter import TelegramReporter
            
            strategy_manager = StrategyManager(mock_config)
            ai_monitor = AIModelMonitor()
            telegram_reporter = TelegramReporter()
            
            # Test complete trading cycle
            signal = await strategy_manager.generate_signal(mock_market_data)
            
            assert signal is not None
            assert 'action' in signal
            assert signal['action'] in ['BUY', 'SELL', 'HOLD']
            
            if signal['action'] in ['BUY', 'SELL']:
                # Execute trade
                result = await strategy_manager.execute_trade(signal)
                assert result['status'] == 'success'
                assert 'order_id' in result
                
                # Check AI model performance
                ai_health = ai_monitor.check_model_health()
                assert ai_health['status'] == 'healthy'
                
                # Verify telegram reporting
                await telegram_reporter.send_trade_report(result)
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, mock_config):
        """Xatolik holatlari uchun integration test"""
        with patch.multiple(
            'trading.strategy_manager.StrategyManager',
            get_market_data=AsyncMock(side_effect=Exception("API Error")),
            place_order=AsyncMock(side_effect=Exception("Order Failed"))
        ):
            from trading.strategy_manager import StrategyManager
            from utils.health_checker import HealthChecker
            
            strategy_manager = StrategyManager(mock_config)
            health_checker = HealthChecker()
            
            # Test error handling
            with pytest.raises(Exception):
                await strategy_manager.get_market_data()
            
            # Check health status after error
            health_status = health_checker.check_all_systems()
            assert health_status['overall_health'] == 'degraded'
            assert 'api_connection' in health_status['failed_checks']
    
    @pytest.mark.asyncio
    async def test_rate_limiting_integration(self, mock_config):
        """Rate limiting integration test"""
        from utils.rate_limiter import RateLimiter
        from trading.strategy_manager import StrategyManager
        
        rate_limiter = RateLimiter(max_calls=5, window=10)
        strategy_manager = StrategyManager(mock_config)
        
        # Test rate limiting
        for i in range(10):
            allowed, message = rate_limiter.check_rate_limit('test_api')
            if i < 5:
                assert allowed == True
            else:
                assert allowed == False
                assert 'rate limit exceeded' in message.lower()
    
    @pytest.mark.asyncio
    async def test_ai_model_integration(self, mock_config, mock_market_data):
        """AI model integration test"""
        with patch('ai.model_monitor.AIModelMonitor.load_model') as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = [0.8, 0.2]  # Buy signal
            mock_load.return_value = mock_model
            
            from ai.model_monitor import AIModelMonitor
            from trading.strategy_manager import StrategyManager
            
            ai_monitor = AIModelMonitor()
            strategy_manager = StrategyManager(mock_config)
            
            # Test AI prediction
            prediction = ai_monitor.predict_market_direction(mock_market_data)
            assert prediction['confidence'] > 0.7
            assert prediction['action'] in ['BUY', 'SELL', 'HOLD']
            
            # Test model health monitoring
            health = ai_monitor.check_model_health()
            assert health['status'] == 'healthy'
    
    @pytest.mark.asyncio
    async def test_telegram_integration(self, mock_config):
        """Telegram integration test"""
        with patch('telegram.Bot') as mock_bot:
            mock_bot.send_message = AsyncMock()
            
            from monitoring.telegram_reporter import TelegramReporter
            
            reporter = TelegramReporter()
            
            # Test different message types
            await reporter.send_trade_report({
                'action': 'BUY',
                'pair': 'BTC/USDT',
                'price': 47200,
                'amount': 0.01,
                'profit': 25.50
            })
            
            await reporter.send_health_report({
                'status': 'healthy',
                'uptime': '2h 30m',
                'trades_today': 15
            })
            
            await reporter.send_critical_alert('High drawdown detected: 4.5%')
            
            # Verify calls
            assert mock_bot.send_message.call_count >= 3
    
    @pytest.mark.asyncio
    async def test_backtest_integration(self, mock_config):
        """Backtest integration test"""
        from backtest.advanced_backtester import AdvancedBacktester
        from trading.strategy_manager import StrategyManager
        
        backtester = AdvancedBacktester()
        strategy_manager = StrategyManager(mock_config)
        
        # Mock historical data
        historical_data = [
            {'timestamp': 1640995200000, 'open': 47000, 'high': 47100, 'low': 46900, 'close': 47050, 'volume': 1.5},
            {'timestamp': 1640995260000, 'open': 47050, 'high': 47150, 'low': 46950, 'close': 47100, 'volume': 1.8},
            {'timestamp': 1640995320000, 'open': 47100, 'high': 47200, 'low': 47000, 'close': 47150, 'volume': 2.1}
        ]
        
        # Run backtest
        results = backtester.run_comprehensive_backtest(strategy_manager, historical_data)
        
        assert 'total_return' in results
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results
        assert 'win_rate' in results
        assert isinstance(results['total_return'], (int, float))
    
    @pytest.mark.asyncio
    async def test_risk_management_integration(self, mock_config):
        """Risk management integration test"""
        from trading.strategy_manager import StrategyManager
        from utils.health_checker import HealthChecker
        
        strategy_manager = StrategyManager(mock_config)
        health_checker = HealthChecker()
        
        # Test position size calculation
        position_size = strategy_manager.calculate_position_size(
            account_balance=1000,
            risk_percent=0.02,
            entry_price=47200,
            stop_loss_price=46200
        )
        
        assert position_size > 0
        assert position_size <= 1000 * 0.02  # Max 2% risk
        
        # Test stop loss trigger
        current_positions = [
            {'id': 'pos1', 'side': 'long', 'entry_price': 47000, 'amount': 0.01, 'stop_loss': 46000}
        ]
        
        stop_loss_triggered = strategy_manager.check_stop_loss(current_positions, current_price=45800)
        assert len(stop_loss_triggered) > 0
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, mock_config):
        """Performance monitoring integration test"""
        from utils.advanced_logger import TradingLogger
        from monitoring.telegram_reporter import TelegramReporter
        
        logger = TradingLogger()
        reporter = TelegramReporter()
        
        # Log sample trades
        sample_trades = [
            {'action': 'BUY', 'price': 47000, 'amount': 0.01, 'profit': 15.5, 'timestamp': datetime.now()},
            {'action': 'SELL', 'price': 47200, 'amount': 0.01, 'profit': -8.2, 'timestamp': datetime.now()},
            {'action': 'BUY', 'price': 47100, 'amount': 0.01, 'profit': 22.8, 'timestamp': datetime.now()}
        ]
        
        for trade in sample_trades:
            logger.log_trade_decision(trade, 'executed', 'AI signal confirmed')
        
        # Calculate performance metrics
        metrics = logger.calculate_performance_metrics()
        
        assert 'total_trades' in metrics
        assert 'win_rate' in metrics
        assert 'average_profit' in metrics
        assert metrics['total_trades'] == 3
        
        # Test performance reporting
        await reporter.send_performance_report(metrics)
    
    @pytest.mark.asyncio
    async def test_config_validation_integration(self, mock_config):
        """Configuration validation integration test"""
        from config.validators import ConfigValidator
        
        validator = ConfigValidator()
        
        # Test valid config
        validation_result = validator.validate_config(mock_config)
        assert validation_result['valid'] == True
        
        # Test invalid config
        invalid_config = mock_config.copy()
        invalid_config['risk']['max_drawdown'] = 1.5  # Invalid: >100%
        
        validation_result = validator.validate_config(invalid_config)
        assert validation_result['valid'] == False
        assert 'max_drawdown' in validation_result['errors']
    
    @pytest.mark.asyncio
    async def test_system_recovery_integration(self, mock_config):
        """System recovery integration test"""
        from utils.health_checker import HealthChecker
        from trading.strategy_manager import StrategyManager
        
        health_checker = HealthChecker()
        strategy_manager = StrategyManager(mock_config)
        
        # Simulate system failure
        health_checker.record_failure('database_connection', 'Connection timeout')
        health_checker.record_failure('api_connection', 'Rate limit exceeded')
        
        # Check system health
        health_status = health_checker.check_all_systems()
        assert health_status['overall_health'] == 'critical'
        
        # Test recovery procedures
        recovery_actions = health_checker.get_recovery_actions()
        assert len(recovery_actions) > 0
        assert any('restart' in action.lower() for action in recovery_actions)
    
    def test_logging_integration(self, mock_config):
        """Logging integration test"""
        from utils.advanced_logger import TradingLogger
        
        logger = TradingLogger()
        
        # Test different log levels
        logger.log_info("Bot started successfully")
        logger.log_warning("High volatility detected")
        logger.log_error("Failed to connect to exchange")
        logger.log_critical("Emergency shutdown triggered")
        
        # Test trade logging
        trade_data = {
            'action': 'BUY',
            'pair': 'BTC/USDT',
            'price': 47200,
            'amount': 0.01,
            'confidence': 0.85
        }
        
        logger.log_trade_decision(trade_data, 'executed', 'Strong bullish signal')
        
        # Verify logs
        assert len(logger.trade_history) > 0
        assert logger.trade_history[-1]['signal']['action'] == 'BUY'
    
    @pytest.mark.asyncio
    async def test_full_system_stress_test(self, mock_config):
        """To'liq sistema stress test"""
        from trading.strategy_manager import StrategyManager
        from utils.rate_limiter import RateLimiter
        from utils.health_checker import HealthChecker
        
        strategy_manager = StrategyManager(mock_config)
        rate_limiter = RateLimiter(max_calls=100, window=60)
        health_checker = HealthChecker()
        
        # Simulate high load
        tasks = []
        for i in range(50):
            task = asyncio.create_task(self._simulate_trading_activity(strategy_manager, rate_limiter))
            tasks.append(task)
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check system health after stress
        health_status = health_checker.check_all_systems()
        
        # Verify system handled stress
        successful_tasks = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_tasks) > 0
        
        # System should still be operational
        assert health_status['overall_health'] in ['healthy', 'degraded']
    
    async def _simulate_trading_activity(self, strategy_manager, rate_limiter):
        """Simulate trading activity for stress test"""
        try:
            # Check rate limit
            allowed, _ = rate_limiter.check_rate_limit('trading_api')
            if not allowed:
                return False
            
            # Simulate market data fetch
            await asyncio.sleep(0.1)  # Simulate API call
            
            # Simulate signal generation
            signal = {'action': 'BUY', 'confidence': 0.8}
            
            # Simulate trade execution
            await asyncio.sleep(0.05)  # Simulate order placement
            
            return True
        except Exception as e:
            return False


class TestSystemStability:
    """Sistema barqarorligi testlari"""
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self):
        """Memory leak detection test"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Simulate long-running operations
        for i in range(1000):
            data = {'timestamp': datetime.now(), 'data': list(range(100))}
            # Simulate processing
            await asyncio.sleep(0.001)
        
        # Force garbage collection
        gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (<10MB)
        assert memory_increase < 10 * 1024 * 1024, f"Memory leak detected: {memory_increase} bytes"
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Concurrent operations test"""
        from utils.advanced_logger import TradingLogger
        
        logger = TradingLogger()
        
        # Test concurrent logging
        async def log_worker(worker_id):
            for i in range(100):
                logger.log_info(f"Worker {worker_id} - Message {i}")
                await asyncio.sleep(0.001)
        
        # Start multiple workers
        tasks = [log_worker(i) for i in range(5)]
        await asyncio.gather(*tasks)
        
        # Verify all messages logged
        assert len(logger.get_recent_logs()) >= 500
    
    def test_configuration_edge_cases(self):
        """Configuration edge cases test"""
        from config.validators import ConfigValidator
        
        validator = ConfigValidator()
        
        # Test extreme values
        edge_cases = [
            {'risk': {'max_drawdown': 0.001}},  # Very low
            {'risk': {'max_drawdown': 0.99}},   # Very high
            {'trading': {'position_size': 0.00001}},  # Very small
            {'ai': {'confidence_threshold': 0.99}},   # Very high confidence
        ]
        
        for config in edge_cases:
            result = validator.validate_config(config)
            # Should handle edge cases gracefully
            assert isinstance(result, dict)
            assert 'valid' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
