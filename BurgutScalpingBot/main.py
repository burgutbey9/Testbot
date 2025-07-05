#!/usr/bin/env python3
"""
BurgutScalpingBot - Main Entry Point
DEX AI Scalping Bot with comprehensive error handling and monitoring
"""

import asyncio
import signal
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional
import logging
from contextlib import asynccontextmanager

# Internal imports
from config.settings import Settings
from config.validators import ConfigValidator
from utils.advanced_logger import TradingLogger
from utils.rate_limiter import RateLimiter
from utils.health_checker import HealthChecker
from monitoring.telegram_reporter import TelegramReporter
from monitoring.metrics_collector import MetricsCollector
from trading.strategy_manager import StrategyManager
from trading.order_manager import OrderManager
from trading.position_manager import PositionManager
from ai.model_monitor import AIModelMonitor
from ai.sentiment_analyzer import AISentimentAnalyzer
from ai.rl_agent import RLAgent
from risk_management.risk_manager import RiskManager
from backtest.advanced_backtester import AdvancedBacktester


class BurgutScalpingBot:
    """Main bot class with comprehensive error handling and monitoring"""
    
    def __init__(self):
        # Initialize configuration
        self.settings = Settings()
        self.config_validator = ConfigValidator()
        
        # Initialize logging
        self.logger = TradingLogger()
        self.main_logger = logging.getLogger(__name__)
        
        # Initialize core components
        self.rate_limiter = RateLimiter()
        self.health_checker = HealthChecker()
        self.telegram_reporter = TelegramReporter()
        self.metrics_collector = MetricsCollector()
        
        # Initialize trading components
        self.strategy_manager = StrategyManager()
        self.order_manager = OrderManager()
        self.position_manager = PositionManager()
        self.risk_manager = RiskManager()
        
        # Initialize AI components
        self.ai_monitor = AIModelMonitor()
        self.sentiment_analyzer = AISentimentAnalyzer()
        self.rl_agent = RLAgent()
        
        # Initialize backtest engine
        self.backtester = AdvancedBacktester()
        
        # Bot state
        self.is_running = False
        self.emergency_stop = False
        self.last_heartbeat = datetime.now()
        
        # Performance tracking
        self.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'start_time': datetime.now()
        }
        
        # Circuit breaker states
        self.circuit_breaker = {
            'consecutive_losses': 0,
            'max_consecutive_losses': 5,
            'daily_loss_limit': 0.05,  # 5% daily loss limit
            'is_tripped': False
        }
    
    async def initialize(self) -> bool:
        """Initialize bot with comprehensive validation"""
        try:
            self.main_logger.info("🚀 BurgutScalpingBot başlatilmoqda...")
            
            # 1. Validate configuration
            validation_result = self.config_validator.validate_all_configs()
            if not validation_result.is_valid:
                self.main_logger.error(f"❌ Config validation failed: {validation_result.errors}")
                return False
            
            # 2. Initialize all components
            components_init = await self._initialize_components()
            if not components_init:
                self.main_logger.error("❌ Component initialization failed")
                return False
            
            # 3. Run health checks
            health_status = await self.health_checker.comprehensive_health_check()
            if not health_status.is_healthy:
                self.main_logger.error(f"❌ Health check failed: {health_status.issues}")
                return False
            
            # 4. Start monitoring systems
            await self._start_monitoring()
            
            # 5. Send startup notification
            await self.telegram_reporter.send_startup_notification({
                'bot_name': 'BurgutScalpingBot',
                'version': '2.0.0',
                'start_time': datetime.now().isoformat(),
                'config_status': 'validated',
                'health_status': 'healthy'
            })
            
            self.main_logger.info("✅ Bot successfully initialized")
            return True
            
        except Exception as e:
            self.main_logger.error(f"❌ Bot initialization failed: {str(e)}")
            await self.telegram_reporter.send_critical_alert(f"Bot initialization failed: {str(e)}")
            return False
    
    async def _initialize_components(self) -> bool:
        """Initialize all bot components"""
        try:
            # Initialize in dependency order
            components = [
                ('rate_limiter', self.rate_limiter),
                ('health_checker', self.health_checker),
                ('telegram_reporter', self.telegram_reporter),
                ('metrics_collector', self.metrics_collector),
                ('risk_manager', self.risk_manager),
                ('order_manager', self.order_manager),
                ('position_manager', self.position_manager),
                ('sentiment_analyzer', self.sentiment_analyzer),
                ('ai_monitor', self.ai_monitor),
                ('rl_agent', self.rl_agent),
                ('strategy_manager', self.strategy_manager),
                ('backtester', self.backtester)
            ]
            
            for name, component in components:
                try:
                    if hasattr(component, 'initialize'):
                        await component.initialize()
                    self.main_logger.info(f"✅ {name} initialized")
                except Exception as e:
                    self.main_logger.error(f"❌ {name} initialization failed: {str(e)}")
                    return False
            
            return True
            
        except Exception as e:
            self.main_logger.error(f"Component initialization error: {str(e)}")
            return False
    
    async def _start_monitoring(self):
        """Start monitoring systems"""
        # Start background tasks
        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._metrics_collection_loop())
        asyncio.create_task(self._health_monitoring_loop())
        asyncio.create_task(self._ai_model_monitoring_loop())
        asyncio.create_task(self._performance_reporting_loop())
    
    async def run(self):
        """Main bot execution loop"""
        try:
            self.is_running = True
            self.main_logger.info("🎯 Trading loop started")
            
            while self.is_running and not self.emergency_stop:
                try:
                    # Update heartbeat
                    self.last_heartbeat = datetime.now()
                    
                    # Check circuit breaker
                    if self.circuit_breaker['is_tripped']:
                        self.main_logger.warning("🔴 Circuit breaker tripped, waiting...")
                        await asyncio.sleep(30)
                        continue
                    
                    # Rate limiting check
                    can_trade, rate_msg = self.rate_limiter.check_rate_limit('main_trading')
                    if not can_trade:
                        self.main_logger.warning(f"⏰ Rate limit: {rate_msg}")
                        await asyncio.sleep(5)
                        continue
                    
                    # Execute trading cycle
                    await self._execute_trading_cycle()
                    
                    # Sleep between cycles
                    await asyncio.sleep(self.settings.TRADING_CYCLE_INTERVAL)
                    
                except Exception as e:
                    self.main_logger.error(f"❌ Trading cycle error: {str(e)}")
                    await self.telegram_reporter.send_critical_alert(f"Trading cycle error: {str(e)}")
                    await asyncio.sleep(10)
                    
        except Exception as e:
            self.main_logger.error(f"❌ Main loop error: {str(e)}")
            await self.emergency_shutdown(f"Main loop error: {str(e)}")
    
    async def _execute_trading_cycle(self):
        """Execute single trading cycle"""
        try:
            # 1. Collect market data
            market_data = await self._collect_market_data()
            if not market_data:
                return
            
            # 2. Analyze sentiment
            sentiment_data = await self.sentiment_analyzer.analyze_market_sentiment(market_data)
            
            # 3. Get AI predictions
            ai_predictions = await self.rl_agent.get_predictions(market_data, sentiment_data)
            
            # 4. Check AI model health
            model_health = await self.ai_monitor.check_model_health(ai_predictions)
            if not model_health.is_healthy:
                self.main_logger.warning(f"⚠️ AI model health issue: {model_health.issues}")
                # Fallback to simpler strategy
                ai_predictions = await self.strategy_manager.get_fallback_signals(market_data)
            
            # 5. Generate trading signals
            signals = await self.strategy_manager.generate_signals(
                market_data, sentiment_data, ai_predictions
            )
            
            # 6. Risk management check
            risk_approved_signals = await self.risk_manager.evaluate_signals(signals)
            
            # 7. Execute approved trades
            if risk_approved_signals:
                await self._execute_trades(risk_approved_signals)
            
            # 8. Update positions
            await self.position_manager.update_positions()
            
            # 9. Collect metrics
            await self.metrics_collector.collect_cycle_metrics({
                'signals_generated': len(signals),
                'signals_approved': len(risk_approved_signals),
                'sentiment_score': sentiment_data.get('composite_score', 0),
                'ai_confidence': ai_predictions.get('confidence', 0)
            })
            
        except Exception as e:
            self.main_logger.error(f"Trading cycle execution error: {str(e)}")
            raise
    
    async def _collect_market_data(self) -> Optional[Dict[str, Any]]:
        """Collect market data from multiple sources"""
        try:
            # Implementation would collect from DEX APIs
            market_data = {
                'timestamp': datetime.now().isoformat(),
                'price': 50000.0,  # Example data
                'volume': 1000000.0,
                'order_book': {'bids': [], 'asks': []},
                'recent_trades': []
            }
            
            return market_data
            
        except Exception as e:
            self.main_logger.error(f"Market data collection error: {str(e)}")
            return None
    
    async def _execute_trades(self, signals: list):
        """Execute approved trading signals"""
        for signal in signals:
            try:
                # Execute trade through order manager
                result = await self.order_manager.execute_order(signal)
                
                if result.success:
                    self.performance_stats['total_trades'] += 1
                    self.logger.log_trade_decision(signal, result, "Signal executed successfully")
                    
                    await self.telegram_reporter.send_trade_notification({
                        'signal': signal,
                        'result': result,
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    self.main_logger.warning(f"⚠️ Trade execution failed: {result.error}")
                    
            except Exception as e:
                self.main_logger.error(f"Trade execution error: {str(e)}")
                await self.telegram_reporter.send_critical_alert(f"Trade execution error: {str(e)}")
    
    async def _heartbeat_loop(self):
        """Heartbeat monitoring loop"""
        while self.is_running:
            try:
                await self.health_checker.send_heartbeat()
                await asyncio.sleep(60)  # Every minute
            except Exception as e:
                self.main_logger.error(f"Heartbeat error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _metrics_collection_loop(self):
        """Metrics collection loop"""
        while self.is_running:
            try:
                await self.metrics_collector.collect_system_metrics()
                await asyncio.sleep(300)  # Every 5 minutes
            except Exception as e:
                self.main_logger.error(f"Metrics collection error: {str(e)}")
                await asyncio.sleep(300)
    
    async def _health_monitoring_loop(self):
        """Health monitoring loop"""
        while self.is_running:
            try:
                health_status = await self.health_checker.comprehensive_health_check()
                if not health_status.is_healthy:
                    await self.telegram_reporter.send_health_alert(health_status)
                
                await asyncio.sleep(900)  # Every 15 minutes
            except Exception as e:
                self.main_logger.error(f"Health monitoring error: {str(e)}")
                await asyncio.sleep(900)
    
    async def _ai_model_monitoring_loop(self):
        """AI model monitoring loop"""
        while self.is_running:
            try:
                model_status = await self.ai_monitor.comprehensive_model_check()
                if model_status.requires_retraining:
                    await self.telegram_reporter.send_critical_alert(
                        "AI model requires retraining - performance degraded"
                    )
                
                await asyncio.sleep(1800)  # Every 30 minutes
            except Exception as e:
                self.main_logger.error(f"AI monitoring error: {str(e)}")
                await asyncio.sleep(1800)
    
    async def _performance_reporting_loop(self):
        """Performance reporting loop"""
        while self.is_running:
            try:
                performance_report = await self._generate_performance_report()
                await self.telegram_reporter.send_performance_report(performance_report)
                await asyncio.sleep(3600)  # Every hour
            except Exception as e:
                self.main_logger.error(f"Performance reporting error: {str(e)}")
                await asyncio.sleep(3600)
    
    async def _generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        current_time = datetime.now()
        runtime = current_time - self.performance_stats['start_time']
        
        return {
            'timestamp': current_time.isoformat(),
            'runtime_hours': runtime.total_seconds() / 3600,
            'total_trades': self.performance_stats['total_trades'],
            'winning_trades': self.performance_stats['winning_trades'],
            'win_rate': (self.performance_stats['winning_trades'] / max(self.performance_stats['total_trades'], 1)) * 100,
            'total_pnl': self.performance_stats['total_pnl'],
            'max_drawdown': self.performance_stats['max_drawdown'],
            'avg_trades_per_hour': self.performance_stats['total_trades'] / max(runtime.total_seconds() / 3600, 1)
        }
    
    async def emergency_shutdown(self, reason: str):
        """Emergency shutdown with cleanup"""
        self.main_logger.critical(f"🚨 EMERGENCY SHUTDOWN: {reason}")
        
        try:
            # Set emergency stop flag
            self.emergency_stop = True
            
            # Close all positions
            await self.position_manager.close_all_positions()
            
            # Cancel all orders
            await self.order_manager.cancel_all_orders()
            
            # Send emergency notification
            await self.telegram_reporter.send_emergency_alert({
                'reason': reason,
                'timestamp': datetime.now().isoformat(),
                'positions_closed': True,
                'orders_cancelled': True
            })
            
            # Stop bot
            self.is_running = False
            
        except Exception as e:
            self.main_logger.error(f"Emergency shutdown error: {str(e)}")
    
    async def graceful_shutdown(self):
        """Graceful shutdown with cleanup"""
        self.main_logger.info("🔄 Graceful shutdown initiated")
        
        try:
            # Stop new trading
            self.is_running = False
            
            # Wait for current trades to complete
            await asyncio.sleep(5)
            
            # Close positions if required
            if self.settings.CLOSE_POSITIONS_ON_SHUTDOWN:
                await self.position_manager.close_all_positions()
            
            # Send shutdown notification
            await self.telegram_reporter.send_shutdown_notification({
                'timestamp': datetime.now().isoformat(),
                'final_performance': await self._generate_performance_report()
            })
            
            self.main_logger.info("✅ Graceful shutdown completed")
            
        except Exception as e:
            self.main_logger.error(f"Graceful shutdown error: {str(e)}")


async def main():
    """Main entry point"""
    bot = None
    
    try:
        # Initialize bot
        bot = BurgutScalpingBot()
        
        # Setup signal handlers
        def signal_handler(signum, frame):
            if bot:
                asyncio.create_task(bot.graceful_shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Initialize bot
        if not await bot.initialize():
            print("❌ Bot initialization failed")
            sys.exit(1)
        
        # Run bot
        await bot.run()
        
    except KeyboardInterrupt:
        print("\n🔄 Keyboard interrupt received")
        if bot:
            await bot.graceful_shutdown()
    
    except Exception as e:
        print(f"❌ Critical error: {str(e)}")
        if bot:
            await bot.emergency_shutdown(f"Critical error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # Set up basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run bot
    asyncio.run(main())
