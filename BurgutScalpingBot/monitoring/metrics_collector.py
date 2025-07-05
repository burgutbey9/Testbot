import asyncio
import time
import json
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import statistics
import numpy as np
from dataclasses import dataclass, asdict
import sqlite3
import os


@dataclass
class MetricSnapshot:
    """Metric snapshot data structure"""
    timestamp: datetime
    metric_name: str
    value: float
    metadata: Dict[str, Any]


class MetricsCollector:
    """
    Performance metrics yig'uvchi va tahlil qiluvchi
    Real-time monitoring va historical analysis
    """
    
    def __init__(self, db_path: str = "data/metrics.db", collection_interval: int = 30):
        self.db_path = db_path
        self.collection_interval = collection_interval
        
        # In-memory storage for recent metrics
        self.recent_metrics = defaultdict(lambda: deque(maxlen=1000))
        self.system_metrics = deque(maxlen=1000)
        self.trading_metrics = deque(maxlen=1000)
        self.ai_metrics = deque(maxlen=1000)
        
        # Performance counters
        self.performance_counters = {
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'api_calls': 0,
            'api_errors': 0,
            'uptime_start': datetime.now()
        }
        
        # Real-time alerts
        self.alert_thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 80.0,
            'error_rate': 0.05,
            'latency': 1000.0,  # milliseconds
            'drawdown': 0.05,   # 5%
            'win_rate': 0.3     # Below 30%
        }
        
        # Thread safety
        self.lock = threading.Lock()
        self.collection_active = False
        
        # Initialize database
        self._init_database()
        
        # Start background collection
        self._start_background_collection()
    
    def _init_database(self):
        """Initialize SQLite database for metrics storage"""
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else '.', exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    category TEXT,
                    metric_name TEXT,
                    value REAL,
                    metadata TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    alert_type TEXT,
                    severity TEXT,
                    message TEXT,
                    resolved BOOLEAN DEFAULT FALSE
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp 
                ON metrics(timestamp)
            ''')
    
    def _start_background_collection(self):
        """Start background metrics collection"""
        self.collection_active = True
        self.collection_thread = threading.Thread(target=self._collect_metrics_loop, daemon=True)
        self.collection_thread.start()
    
    def _collect_metrics_loop(self):
        """Background metrics collection loop"""
        while self.collection_active:
            try:
                self._collect_system_metrics()
                self._calculate_derived_metrics()
                self._check_alerts()
                time.sleep(self.collection_interval)
            except Exception as e:
                print(f"Metrics collection error: {e}")
                time.sleep(5)  # Wait before retrying
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network (if available)
            try:
                network = psutil.net_io_counters()
                network_metrics = {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                }
            except:
                network_metrics = {}
            
            # Process-specific metrics
            process = psutil.Process()
            process_metrics = {
                'cpu_percent': process.cpu_percent(),
                'memory_mb': process.memory_info().rss / 1024 / 1024,
                'open_files': len(process.open_files()),
                'threads': process.num_threads()
            }
            
            system_snapshot = {
                'timestamp': datetime.now(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_mb': memory.available / 1024 / 1024,
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / 1024 / 1024 / 1024,
                'process': process_metrics,
                'network': network_metrics
            }
            
            with self.lock:
                self.system_metrics.append(system_snapshot)
                
            # Store in database
            self._store_metrics('system', system_snapshot)
            
        except Exception as e:
            print(f"System metrics collection error: {e}")
    
    def record_trade_metric(self, trade_data: Dict[str, Any]):
        """Record trade-related metrics"""
        with self.lock:
            self.performance_counters['total_trades'] += 1
            
            if trade_data.get('status') == 'success':
                self.performance_counters['successful_trades'] += 1
                profit = trade_data.get('profit', 0)
                if profit > 0:
                    self.performance_counters['total_profit'] += profit
                else:
                    self.performance_counters['total_loss'] += abs(profit)
            else:
                self.performance_counters['failed_trades'] += 1
            
            # Store trade metrics
            trade_metrics = {
                'timestamp': datetime.now(),
                'action': trade_data.get('action'),
                'pair': trade_data.get('pair'),
                'price': trade_data.get('price'),
                'amount': trade_data.get('amount'),
                'profit': trade_data.get('profit', 0),
                'execution_time': trade_data.get('execution_time', 0),
                'slippage': trade_data.get('slippage', 0),
                'status': trade_data.get('status')
            }
            
            self.trading_metrics.append(trade_metrics)
            
        # Store in database
        self._store_metrics('trading', trade_metrics)
    
    def record_api_metric(self, api_name: str, response_time: float, success: bool):
        """Record API call metrics"""
        with self.lock:
            self.performance_counters['api_calls'] += 1
            if not success:
                self.performance_counters['api_errors'] += 1
            
            api_metrics = {
                'timestamp': datetime.now(),
                'api_name': api_name,
                'response_time': response_time,
                'success': success
            }
            
            self.recent_metrics[f'api_{api_name}'].append(api_metrics)
        
        # Store in database
        self._store_metrics('api', api_metrics)
    
    def record_ai_metric(self, model_name: str, prediction_confidence: float, 
                        prediction_time: float, accuracy: Optional[float] = None):
        """Record AI model metrics"""
        ai_metrics = {
            'timestamp': datetime.now(),
            'model_name': model_name,
            'prediction_confidence': prediction_confidence,
            'prediction_time': prediction_time,
            'accuracy': accuracy
        }
        
        with self.lock:
            self.ai_metrics.append(ai_metrics)
        
        # Store in database
        self._store_metrics('ai', ai_metrics)
    
    def _calculate_derived_metrics(self):
        """Calculate derived metrics from raw data"""
        try:
            with self.lock:
                # Trading performance metrics
                if self.performance_counters['total_trades'] > 0:
                    win_rate = self.performance_counters['successful_trades'] / self.performance_counters['total_trades']
                    
                    # Calculate drawdown
                    if len(self.trading_metrics) > 1:
                        drawdown = self._calculate_drawdown()
                    else:
                        drawdown = 0.0
                    
                    # Calculate Sharpe ratio (simplified)
                    sharpe_ratio = self._calculate_sharpe_ratio()
                    
                    derived_metrics = {
                        'timestamp': datetime.now(),
                        'win_rate': win_rate,
                        'drawdown': drawdown,
                        'sharpe_ratio': sharpe_ratio,
                        'total_pnl': self.performance_counters['total_profit'] - self.performance_counters['total_loss'],
                        'avg_profit_per_trade': (self.performance_counters['total_profit'] - self.performance_counters['total_loss']) / self.performance_counters['total_trades']
                    }
                    
                    self.recent_metrics['derived'].append(derived_metrics)
                    self._store_metrics('derived', derived_metrics)
                
                # API performance metrics
                if self.performance_counters['api_calls'] > 0:
                    error_rate = self.performance_counters['api_errors'] / self.performance_counters['api_calls']
                    
                    api_performance = {
                        'timestamp': datetime.now(),
                        'error_rate': error_rate,
                        'total_calls': self.performance_counters['api_calls'],
                        'total_errors': self.performance_counters['api_errors']
                    }
                    
                    self.recent_metrics['api_performance'].append(api_performance)
                    self._store_metrics('api_performance', api_performance)
                
        except Exception as e:
            print(f"Derived metrics calculation error: {e}")
    
    def _calculate_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if len(self.trading_metrics) < 2:
            return 0.0
        
        # Calculate cumulative returns
        cumulative_pnl = 0
        peak = 0
        max_drawdown = 0
        
        for trade in self.trading_metrics:
            cumulative_pnl += trade.get('profit', 0)
            if cumulative_pnl > peak:
                peak = cumulative_pnl
            
            drawdown = (peak - cumulative_pnl) / max(peak, 1)
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        if len(self.trading_metrics) < 10:
            return 0.0
        
        returns = [trade.get('profit', 0) for trade in self.trading_metrics]
        
        if len(returns) < 2:
            return 0.0
        
        avg_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)
        
        if std_return == 0:
            return 0.0
        
        # Assume risk-free rate of 0 for simplicity
        return avg_return / std_return
    
    def _check_alerts(self):
        """Check for alert conditions"""
        try:
            current_time = datetime.now()
            
            # System alerts
            if self.system_metrics:
                latest_system = self.system_metrics[-1]
                
                if latest_system['cpu_percent'] > self.alert_thresholds['cpu_usage']:
                    self._trigger_alert('system', 'warning', 
                                      f"High CPU usage: {latest_system['cpu_percent']:.1f}%")
                
                if latest_system['memory_percent'] > self.alert_thresholds['memory_usage']:
                    self._trigger_alert('system', 'warning',
                                      f"High memory usage: {latest_system['memory_percent']:.1f}%")
            
            # Trading alerts
            with self.lock:
                if self.performance_counters['total_trades'] > 10:
                    win_rate = self.performance_counters['successful_trades'] / self.performance_counters['total_trades']
                    
                    if win_rate < self.alert_thresholds['win_rate']:
                        self._trigger_alert('trading', 'critical',
                                          f"Low win rate: {win_rate:.2%}")
                
                # Drawdown alert
                if len(self.trading_metrics) > 5:
                    drawdown = self._calculate_drawdown()
                    if drawdown > self.alert_thresholds['drawdown']:
                        self._trigger_alert('trading', 'critical',
                                          f"High drawdown: {drawdown:.2%}")
                
                # API error rate alert
                if self.performance_counters['api_calls'] > 20:
                    error_rate = self.performance_counters['api_errors'] / self.performance_counters['api_calls']
                    if error_rate > self.alert_thresholds['error_rate']:
                        self._trigger_alert('api', 'warning',
                                          f"High API error rate: {error_rate:.2%}")
            
        except Exception as e:
            print(f"Alert checking error: {e}")
    
    def _trigger_alert(self, alert_type: str, severity: str, message: str):
        """Trigger an alert"""
        alert_data = {
            'timestamp': datetime.now(),
            'alert_type': alert_type,
            'severity': severity,
            'message': message,
            'resolved': False
        }
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO alerts (timestamp, alert_type, severity, message, resolved)
                VALUES (?, ?, ?, ?, ?)
            ''', (alert_data['timestamp'], alert_type, severity, message, False))
        
        print(f"ALERT [{severity.upper()}] {alert_type}: {message}")
    
    def _store_metrics(self, category: str, metrics_data: Dict[str, Any]):
        """Store metrics in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for key, value in metrics_data.items():
                    if key == 'timestamp':
                        continue
                    
                    if isinstance(value, dict):
                        # Store nested dict as JSON
                        conn.execute('''
                            INSERT INTO metrics (timestamp, category, metric_name, value, metadata)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (metrics_data['timestamp'], category, key, 0, json.dumps(value)))
                    elif isinstance(value, (int, float)):
                        conn.execute('''
                            INSERT INTO metrics (timestamp, category, metric_name, value, metadata)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (metrics_data['timestamp'], category, key, value, '{}'))
        except Exception as e:
            print(f"Metrics storage error: {e}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health summary"""
        health_data = {
            'timestamp': datetime.now(),
            'uptime': str(datetime.now() - self.performance_counters['uptime_start']),
            'status': 'healthy'
        }
        
        try:
            if self.system_metrics:
                latest_system = self.system_metrics[-1]
                health_data.update({
                    'cpu_usage': latest_system['cpu_percent'],
                    'memory_usage': latest_system['memory_percent'],
                    'disk_usage': latest_system['disk_percent']
                })
                
                # Determine overall health
                if (latest_system['cpu_percent'] > 90 or 
                    latest_system['memory_percent'] > 90):
                    health_data['status'] = 'critical'
                elif (latest_system['cpu_percent'] > 70 or 
                      latest_system['memory_percent'] > 70):
                    health_data['status'] = 'warning'
            
            with self.lock:
                # Trading health
                if self.performance_counters['total_trades'] > 0:
                    win_rate = self.performance_counters['successful_trades'] / self.performance_counters['total_trades']
                    health_data.update({
                        'total_trades': self.performance_counters['total_trades'],
                        'win_rate': win_rate,
                        'total_pnl': self.performance_counters['total_profit'] - self.performance_counters['total_loss']
                    })
                
                # API health
                if self.performance_counters['api_calls'] > 0:
                    error_rate = self.performance_counters['api_errors'] / self.performance_counters['api_calls']
                    health_data.update({
                        'api_calls': self.performance_counters['api_calls'],
                        'api_error_rate': error_rate
                    })
                    
                    if error_rate > 0.1:  # 10% error rate
                        health_data['status'] = 'warning'
            
        except Exception as e:
            health_data['status'] = 'error'
            health_data['error'] = str(e)
        
        return health_data
    
    def get_trading_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get trading summary for specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        summary = {
            'period_hours': hours,
            'total_trades': 0,
            'successful_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'win_rate': 0.0,
            'avg_profit_per_trade': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0,
            'trading_pairs': set()
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT value, metadata FROM metrics 
                    WHERE category = 'trading' AND timestamp >= ?
                    ORDER BY timestamp
                ''', (cutoff_time,))
                
                trades = []
                for row in cursor:
                    try:
                        metadata = json.loads(row[1]) if row[1] else {}
                        if 'profit' in metadata:
                            trades.append(metadata)
                    except:
                        continue
                
                if trades:
                    summary['total_trades'] = len(trades)
                    summary['successful_trades'] = sum(1 for t in trades if t.get('profit', 0) > 0)
                    
                    profits = [t.get('profit', 0) for t in trades]
                    positive_profits = [p for p in profits if p > 0]
                    negative_profits = [p for p in profits if p < 0]
                    
                    summary['total_profit'] = sum(positive_profits)
                    summary['total_loss'] = abs(sum(negative_profits))
                    summary['win_rate'] = len(positive_profits) / len(trades) if trades else 0
                    summary['avg_profit_per_trade'] = sum(profits) / len(trades) if trades else 0
                    summary['best_trade'] = max(profits) if profits else 0
                    summary['worst_trade'] = min(profits) if profits else 0
                    
                    # Collect trading pairs
                    summary['trading_pairs'] = list(set(t.get('pair', '') for t in trades if t.get('pair')))
                
        except Exception as e:
            summary['error'] = str(e)
        
        return summary
    
    def get_performance_chart_data(self, hours: int = 24) -> Dict[str, List]:
        """Get performance chart data for visualization"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        chart_data = {
            'timestamps': [],
            'cumulative_pnl': [],
            'cpu_usage': [],
            'memory_usage': [],
            'trade_count': []
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get trading data
                cursor = conn.execute('''
                    SELECT timestamp, value, metadata FROM metrics 
                    WHERE category = 'trading' AND timestamp >= ?
                    ORDER BY timestamp
                ''', (cutoff_time,))
                
                cumulative_pnl = 0
                for row in cursor:
                    try:
                        timestamp = datetime.fromisoformat(row[0])
                        metadata = json.loads(row[2]) if row[2] else {}
                        profit = metadata.get('profit', 0)
                        
                        cumulative_pnl += profit
                        chart_data['timestamps'].append(timestamp)
                        chart_data['cumulative_pnl'].append(cumulative_pnl)
                    except:
                        continue
                
                # Get system metrics
                cursor = conn.execute('''
                    SELECT timestamp, metric_name, value FROM metrics 
                    WHERE category = 'system' AND metric_name IN ('cpu_percent', 'memory_percent')
                    AND timestamp >= ?
                    ORDER BY timestamp
                ''', (cutoff_time,))
                
                for row in cursor:
                    try:
                        timestamp = datetime.fromisoformat(row[0])
                        metric_name = row[1]
                        value = row[2]
                        
                        if metric_name == 'cpu_percent':
                            chart_data['cpu_usage'].append({'timestamp': timestamp, 'value': value})
                        elif metric_name == 'memory_percent':
                            chart_data['memory_usage'].append({'timestamp': timestamp, 'value': value})
                    except:
                        continue
                
        except Exception as e:
            chart_data['error'] = str(e)
        
        return chart_data
    
    def cleanup_old_data(self, days: int = 30):
        """Clean up old metrics data"""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('DELETE FROM metrics WHERE timestamp < ?', (cutoff_time,))
                conn.execute('DELETE FROM alerts WHERE timestamp < ? AND resolved = TRUE', (cutoff_time,))
                conn.execute('VACUUM')  # Optimize database
        except Exception as e:
            print(f"Cleanup error: {e}")
    
    def stop_collection(self):
        """Stop metrics collection"""
        self.collection_active = False
        if hasattr(self, 'collection_thread'):
            self.collection_thread.join(timeout=5)
    
    def export_metrics(self, filename: str, days: int = 7):
        """Export metrics to JSON file"""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT * FROM metrics WHERE timestamp >= ?
                    ORDER BY timestamp
                ''', (cutoff_time,))
                
                metrics = []
                for row in cursor:
                    metrics.append({
                        'id': row[0],
                        'timestamp': row[1],
                        'category': row[2],
                        'metric_name': row[3],
                        'value': row[4],
                        'metadata': json.loads(row[5]) if row[5] else {}
                    })
                
                with open(filename, 'w') as f:
                    json.dump(metrics, f, indent=2, default=str)
                    
                return len(metrics)
                
        except Exception as e:
            print(f"Export error: {e}")
            return 0


# Global metrics collector instance
metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance"""
    return metrics_collector


if __name__ == "__main__":
    # Test the metrics collector
    collector = MetricsCollector()
    
    # Simulate some metrics
    collector.record_trade_metric({
        'action': 'BUY',
        'pair': 'BTC/USDT',
        'price': 47200,
        'amount': 0.01,
        'profit': 15.5,
        'status': 'success'
    })
    
    collector.record_api_metric('binance', 150.5, True)
    collector.record_ai_metric('scalping_model', 0.85, 50.2, 0.78)
    
    # Get health summary
    health = collector.get_system_health()
    print("System Health:", json.dumps(health, indent=2, default=str))
    
    # Get trading summary
    summary = collector.get_trading_summary(24)
    print("Trading Summary:", json.dumps(summary, indent=2, default=str))
    
    time.sleep(2)
    collector.stop_collection()
