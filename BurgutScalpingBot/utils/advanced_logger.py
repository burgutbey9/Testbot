import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import asyncio
from pathlib import Path

class TradingLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.trade_history = []
        self.performance_metrics = {}
        self.error_count = 0
        self.last_heartbeat = datetime.now()
        
        self.setup_loggers()
        self.setup_file_rotation()
    
    def setup_loggers(self):
        """Configure multiple loggers for different purposes"""
        # Main trading logger
        self.trading_logger = logging.getLogger('trading')
        self.trading_logger.setLevel(logging.INFO)
        
        # Error logger
        self.error_logger = logging.getLogger('error')
        self.error_logger.setLevel(logging.ERROR)
        
        # Performance logger
        self.performance_logger = logging.getLogger('performance')
        self.performance_logger.setLevel(logging.INFO)
        
        # Create handlers
        self.create_file_handlers()
        self.create_console_handler()
    
    def create_file_handlers(self):
        """Create file handlers for different log types"""
        # Trading log handler
        trading_handler = logging.FileHandler(
            self.log_dir / f"trading_{datetime.now().strftime('%Y%m%d')}.log"
        )
        trading_handler.setLevel(logging.INFO)
        trading_formatter = logging.Formatter(
            '%(asctime)s - TRADING - %(levelname)s - %(message)s'
        )
        trading_handler.setFormatter(trading_formatter)
        self.trading_logger.addHandler(trading_handler)
        
        # Error log handler
        error_handler = logging.FileHandler(
            self.log_dir / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
        )
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter(
            '%(asctime)s - ERROR - %(levelname)s - %(message)s - %(pathname)s:%(lineno)d'
        )
        error_handler.setFormatter(error_formatter)
        self.error_logger.addHandler(error_handler)
        
        # Performance log handler
        performance_handler = logging.FileHandler(
            self.log_dir / f"performance_{datetime.now().strftime('%Y%m%d')}.log"
        )
        performance_handler.setLevel(logging.INFO)
        performance_formatter = logging.Formatter(
            '%(asctime)s - PERFORMANCE - %(message)s'
        )
        performance_handler.setFormatter(performance_formatter)
        self.performance_logger.addHandler(performance_handler)
    
    def create_console_handler(self):
        """Create console handler for real-time monitoring"""
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        # Add to all loggers
        self.trading_logger.addHandler(console_handler)
        self.error_logger.addHandler(console_handler)
        self.performance_logger.addHandler(console_handler)
    
    def setup_file_rotation(self):
        """Setup log file rotation to prevent disk space issues"""
        # Delete logs older than 30 days
        cutoff_date = datetime.now() - timedelta(days=30)
        for log_file in self.log_dir.glob("*.log"):
            if log_file.stat().st_mtime < cutoff_date.timestamp():
                try:
                    log_file.unlink()
                except Exception as e:
                    print(f"Failed to delete old log file {log_file}: {e}")
    
    def log_trade_decision(self, signal: Dict[str, Any], decision: str, reasoning: str):
        """Log trading decision with full context"""
        trade_log = {
            'timestamp': datetime.now().isoformat(),
            'signal': {
                'symbol': signal.get('symbol', 'UNKNOWN'),
                'side': signal.get('side', 'UNKNOWN'),
                'price': signal.get('price', 0),
                'size': signal.get('size', 0),
                'confidence': signal.get('confidence', 0),
                'source': signal.get('source', 'UNKNOWN')
            },
            'decision': decision,
            'reasoning': reasoning,
            'ai_confidence': getattr(signal, 'ai_confidence', 0),
            'market_conditions': self.get_market_snapshot(),
            'bot_health': self.get_bot_health()
        }
        
        self.trade_history.append(trade_log)
        self.trading_logger.info(f"Trade Decision: {json.dumps(trade_log, indent=2)}")
        
        # Keep only last 1000 trades in memory
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]
    
    def log_trade_execution(self, trade_id: str, execution_data: Dict[str, Any]):
        """Log trade execution details"""
        execution_log = {
            'timestamp': datetime.now().isoformat(),
            'trade_id': trade_id,
            'execution_data': execution_data,
            'slippage': execution_data.get('slippage', 0),
            'fees': execution_data.get('fees', 0),
            'latency_ms': execution_data.get('latency_ms', 0)
        }
        
        self.trading_logger.info(f"Trade Execution: {json.dumps(execution_log, indent=2)}")
    
    def log_performance_metric(self, metric_name: str, value: float, additional_data: Optional[Dict] = None):
        """Log performance metrics with alerting"""
        metric_log = {
            'timestamp': datetime.now().isoformat(),
            'metric_name': metric_name,
            'value': value,
            'additional_data': additional_data or {}
        }
        
        self.performance_metrics[metric_name] = metric_log
        self.performance_logger.info(f"Performance Metric: {json.dumps(metric_log)}")
        
        # Critical metric alerting
        self.check_critical_metrics(metric_name, value)
    
    def check_critical_metrics(self, metric_name: str, value: float):
        """Check if metrics exceed critical thresholds"""
        critical_thresholds = {
            'drawdown': 0.05,  # 5% drawdown
            'daily_loss': 0.03,  # 3% daily loss
            'error_rate': 0.1,  # 10% error rate
            'api_latency': 5000,  # 5 seconds
            'failed_trades': 5  # 5 failed trades
        }
        
        if metric_name in critical_thresholds:
            threshold = critical_thresholds[metric_name]
            if value > threshold:
                self.log_critical_alert(f"Critical {metric_name}: {value} exceeds threshold {threshold}")
    
    def log_error(self, error_msg: str, error_type: str = "GENERAL", exception: Exception = None):
        """Log errors with detailed context"""
        self.error_count += 1
        
        error_log = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'error_message': error_msg,
            'error_count': self.error_count,
            'exception_type': type(exception).__name__ if exception else None,
            'exception_details': str(exception) if exception else None,
            'bot_health': self.get_bot_health()
        }
        
        self.error_logger.error(f"Error: {json.dumps(error_log, indent=2)}")
        
        # Reset error count daily
        if self.error_count > 50:  # Too many errors
            self.log_critical_alert(f"High error count: {self.error_count}")
    
    def log_critical_alert(self, message: str):
        """Log critical alerts that need immediate attention"""
        alert_log = {
            'timestamp': datetime.now().isoformat(),
            'alert_type': 'CRITICAL',
            'message': message,
            'bot_health': self.get_bot_health(),
            'recent_errors': self.get_recent_errors()
        }
        
        self.error_logger.critical(f"CRITICAL ALERT: {json.dumps(alert_log, indent=2)}")
    
    def log_heartbeat(self):
        """Log system heartbeat"""
        self.last_heartbeat = datetime.now()
        heartbeat_log = {
            'timestamp': self.last_heartbeat.isoformat(),
            'status': 'ALIVE',
            'uptime_seconds': (datetime.now() - self.last_heartbeat).total_seconds(),
            'bot_health': self.get_bot_health()
        }
        
        self.trading_logger.info(f"Heartbeat: {json.dumps(heartbeat_log)}")
    
    def get_market_snapshot(self) -> Dict[str, Any]:
        """Get current market conditions snapshot"""
        # This would be populated with real market data
        return {
            'volatility': 'medium',
            'volume': 'normal',
            'trend': 'neutral',
            'timestamp': datetime.now().isoformat()
        }
    
    def get_bot_health(self) -> Dict[str, Any]:
        """Get bot health status"""
        return {
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'error_count': self.error_count,
            'uptime_hours': (datetime.now() - self.last_heartbeat).total_seconds() / 3600,
            'memory_usage': 'normal',  # Would be populated with actual memory usage
            'cpu_usage': 'normal'      # Would be populated with actual CPU usage
        }
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict]:
        """Get recent error logs"""
        # This would read from error log file
        return [
            {
                'timestamp': datetime.now().isoformat(),
                'error_type': 'EXAMPLE',
                'message': 'Example error for demonstration'
            }
        ]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            'total_trades': len(self.trade_history),
            'error_count': self.error_count,
            'last_update': datetime.now().isoformat(),
            'key_metrics': self.performance_metrics
        }
    
    def export_logs(self, start_date: datetime = None, end_date: datetime = None) -> str:
        """Export logs for analysis"""
        if not start_date:
            start_date = datetime.now() - timedelta(days=1)
        if not end_date:
            end_date = datetime.now()
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'date_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'trade_history': self.trade_history,
            'performance_metrics': self.performance_metrics,
            'bot_health': self.get_bot_health()
        }
        
        export_file = self.log_dir / f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(export_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return str(export_file)

# Global logger instance
trading_logger = TradingLogger()
