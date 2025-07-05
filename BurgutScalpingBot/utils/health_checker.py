import psutil
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import os

@dataclass
class HealthMetrics:
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_connections: int
    uptime_seconds: float
    error_count: int
    last_trade_time: Optional[datetime] = None
    api_response_times: Dict[str, float] = None
    
    def __post_init__(self):
        if self.api_response_times is None:
            self.api_response_times = {}

class HealthChecker:
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.start_time = datetime.now()
        self.last_check = datetime.now()
        self.health_history = []
        self.alerts_sent = {}
        self.logger = logging.getLogger('health_checker')
        
        # Health thresholds
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_usage_percent': 90.0,
            'max_error_count': 10,
            'max_api_response_time': 5.0,
            'heartbeat_timeout': 300  # 5 minutes
        }
        
        self.is_running = False
        self.health_task = None
        
    async def start_monitoring(self):
        """Start continuous health monitoring"""
        self.is_running = True
        self.health_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Health monitoring started")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self.is_running = False
        if self.health_task:
            self.health_task.cancel()
            try:
                await self.health_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                await self.perform_health_check()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Health check failed: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def perform_health_check(self) -> HealthMetrics:
        """Perform comprehensive health check"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            net_connections = len(psutil.net_connections())
            
            # Bot metrics
            uptime = (datetime.now() - self.start_time).total_seconds()
            error_count = self.get_error_count()
            
            # API response times
            api_times = await self.check_api_response_times()
            
            # Create health metrics
            metrics = HealthMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_usage_percent=disk.percent,
                network_connections=net_connections,
                uptime_seconds=uptime,
                error_count=error_count,
                api_response_times=api_times
            )
            
            # Store in history
            self.health_history.append(metrics)
            
            # Keep only last 1000 checks
            if len(self.health_history) > 1000:
                self.health_history = self.health_history[-1000:]
            
            # Check for alerts
            await self.check_alerts(metrics)
            
            # Update last check time
            self.last_check = datetime.now()
            
            self.logger.info(f"Health check completed: CPU={cpu_percent:.1f}%, Memory={memory.percent:.1f}%, Disk={disk.percent:.1f}%")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            raise
    
    def get_error_count(self) -> int:
        """Get current error count from logs"""
        # This would integrate with your logging system
        # For now, return a placeholder
        return 0
    
    async def check_api_response_times(self) -> Dict[str, float]:
        """Check response times for critical APIs"""
        api_times = {}
        
        # Test APIs with dummy calls
        test_apis = [
            ('binance', self.test_binance_api),
            ('telegram', self.test_telegram_api),
            ('dex', self.test_dex_api)
        ]
        
        for api_name, test_func in test_apis:
            try:
                start_time = time.time()
                await test_func()
                response_time = time.time() - start_time
                api_times[api_name] = response_time
            except Exception as e:
                self.logger.warning(f"API {api_name} health check failed: {e}")
                api_times[api_name] = -1  # Error indicator
        
        return api_times
    
    async def test_binance_api(self):
        """Test Binance API connectivity"""
        # Placeholder - implement actual API test
        await asyncio.sleep(0.1)
        return True
    
    async def test_telegram_api(self):
        """Test Telegram API connectivity"""
        # Placeholder - implement actual API test
        await asyncio.sleep(0.1)
        return True
    
    async def test_dex_api(self):
        """Test DEX API connectivity"""
        # Placeholder - implement actual API test
        await asyncio.sleep(0.1)
        return True
    
    async def check_alerts(self, metrics: HealthMetrics):
        """Check metrics against thresholds and send alerts"""
        alerts = []
        
        # CPU check
        if metrics.cpu_percent > self.thresholds['cpu_percent']:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        # Memory check
        if metrics.memory_percent > self.thresholds['memory_percent']:
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        # Disk check
        if metrics.disk_usage_percent > self.thresholds['disk_usage_percent']:
            alerts.append(f"High disk usage: {metrics.disk_usage_percent:.1f}%")
        
        # Error count check
        if metrics.error_count > self.thresholds['max_error_count']:
            alerts.append(f"High error count: {metrics.error_count}")
        
        # API response time check
        for api_name, response_time in metrics.api_response_times.items():
            if response_time > self.thresholds['max_api_response_time']:
                alerts.append(f"Slow API response {api_name}: {response_time:.2f}s")
            elif response_time == -1:
                alerts.append(f"API {api_name} is unreachable")
        
        # Send alerts if any
        for alert in alerts:
            await self.send_alert(alert, metrics)
    
    async def send_alert(self, alert_message: str, metrics: HealthMetrics):
        """Send alert with cooldown to prevent spam"""
        current_time = datetime.now()
        
        # Check cooldown (10 minutes per alert type)
        alert_key = alert_message.split(':')[0]  # Get alert type
        if alert_key in self.alerts_sent:
            if current_time - self.alerts_sent[alert_key] < timedelta(minutes=10):
                return  # Skip if in cooldown
        
        self.alerts_sent[alert_key] = current_time
        
        # Log alert
        self.logger.critical(f"HEALTH ALERT: {alert_message}")
        
        # Here you would integrate with your telegram notification system
        # await telegram_reporter.send_critical_alert(alert_message)
    
    def get_current_health_status(self) -> Dict[str, any]:
        """Get current health status"""
        if not self.health_history:
            return {"status": "No health data available"}
        
        latest_metrics = self.health_history[-1]
        
        # Determine overall health
        health_score = self.calculate_health_score(latest_metrics)
        
        status = {
            "overall_health": self.get_health_level(health_score),
            "health_score": health_score,
            "timestamp": latest_metrics.timestamp.isoformat(),
            "uptime_hours": latest_metrics.uptime_seconds / 3600,
            "system_metrics": {
                "cpu_percent": latest_metrics.cpu_percent,
                "memory_percent": latest_metrics.memory_percent,
                "disk_usage_percent": latest_metrics.disk_usage_percent,
                "network_connections": latest_metrics.network_connections
            },
            "bot_metrics": {
                "error_count": latest_metrics.error_count,
                "last_trade_time": latest_metrics.last_trade_time.isoformat() if latest_metrics.last_trade_time else None,
                "api_response_times": latest_metrics.api_response_times
            },
            "alerts_active": len(self.alerts_sent),
            "last_check": self.last_check.isoformat()
        }
        
        return status
    
    def calculate_health_score(self, metrics: HealthMetrics) -> float:
        """Calculate overall health score (0-100)"""
        score = 100.0
        
        # CPU penalty
        if metrics.cpu_percent > 50:
            score -= min(50, (metrics.cpu_percent - 50) * 2)
        
        # Memory penalty
        if metrics.memory_percent > 70:
            score -= min(30, (metrics.memory_percent - 70) * 2)
        
        # Disk penalty
        if metrics.disk_usage_percent > 80:
            score -= min(20, (metrics.disk_usage_percent - 80) * 2)
        
        # Error penalty
        if metrics.error_count > 0:
            score -= min(30, metrics.error_count * 3)
        
        # API response time penalty
        for api_name, response_time in metrics.api_response_times.items():
            if response_time > 1.0:
                score -= min(10, (response_time - 1.0) * 5)
            elif response_time == -1:
                score -= 20  # API unreachable
        
        return max(0, score)
    
    def get_health_level(self, score: float) -> str:
        """Convert health score to level"""
        if score >= 90:
            return "EXCELLENT"
        elif score >= 75:
            return "GOOD"
        elif score >= 60:
            return "FAIR"
        elif score >= 40:
            return "POOR"
        else:
            return "CRITICAL"
    
    def get_health_history(self, hours: int = 24) -> List[Dict]:
        """Get health history for specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        history = []
        for metrics in self.health_history:
            if metrics.timestamp >= cutoff_time:
                history.append({
                    "timestamp": metrics.timestamp.isoformat(),
                    "cpu_percent": metrics.cpu_percent,
                    "memory_percent": metrics.memory_percent,
                    "disk_usage_percent": metrics.disk_usage_percent,
                    "error_count": metrics.error_count,
                    "health_score": self.calculate_health_score(metrics)
                })
        
        return history
    
    def get_performance_trends(self) -> Dict[str, any]:
        """Analyze performance trends"""
        if len(self.health_history) < 10:
            return {"error": "Insufficient data for trend analysis"}
        
        recent_metrics = self.health_history[-10:]
        
        # Calculate averages
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        avg_errors = sum(m.error_count for m in recent_metrics) / len(recent_metrics)
        
        # Calculate trends (simple linear trend)
        cpu_trend = self.calculate_trend([m.cpu_percent for m in recent_metrics])
        memory_trend = self.calculate_trend([m.memory_percent for m in recent_metrics])
        
        return {
            "averages": {
                "cpu_percent": avg_cpu,
                "memory_percent": avg_memory,
                "error_count": avg_errors
            },
            "trends": {
                "cpu_trend": cpu_trend,
                "memory_trend": memory_trend
            },
            "analysis_period": "last_10_checks"
        }
    
    def calculate_trend(self, values: List[float]) -> str:
        """Calculate simple trend direction"""
        if len(values) < 2:
            return "STABLE"
        
        first_half = sum(values[:len(values)//2]) / (len(values)//2)
        second_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
        
        change_percent = ((second_half - first_half) / first_half) * 100
        
        if change_percent > 10:
            return "INCREASING"
        elif change_percent < -10:
            return "DECREASING"
        else:
            return "STABLE"
    
    def export_health_report(self) -> str:
        """Export comprehensive health report"""
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "bot_uptime_hours": (datetime.now() - self.start_time).total_seconds() / 3600,
            "current_status": self.get_current_health_status(),
            "performance_trends": self.get_performance_trends(),
            "health_history_24h": self.get_health_history(24),
            "thresholds": self.thresholds,
            "alerts_sent_today": len(self.alerts_sent)
        }
        
        # Save to file
        report_file = f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report_file
    
    def emergency_shutdown_check(self) -> Tuple[bool, str]:
        """Check if emergency shutdown is needed"""
        if not self.health_history:
            return False, "No health data"
        
        latest_metrics = self.health_history[-1]
        
        # Critical conditions that require shutdown
        critical_conditions = []
        
        if latest_metrics.cpu_percent > 95:
            critical_conditions.append("CPU usage > 95%")
        
        if latest_metrics.memory_percent > 95:
            critical_conditions.append("Memory usage > 95%")
        
        if latest_metrics.disk_usage_percent > 98:
            critical_conditions.append("Disk usage > 98%")
        
        if latest_metrics.error_count > 50:
            critical_conditions.append("Error count > 50")
        
        # Check if all APIs are down
        api_down_count = sum(1 for rt in latest_metrics.api_response_times.values() if rt == -1)
        if api_down_count == len(latest_metrics.api_response_times):
            critical_conditions.append("All APIs unreachable")
        
        if critical_conditions:
            return True, f"Emergency shutdown required: {'; '.join(critical_conditions)}"
        
        return False, "System stable"
    
    async def handle_emergency_shutdown(self):
        """Handle emergency shutdown procedure"""
        self.logger.critical("EMERGENCY SHUTDOWN INITIATED")
        
        # Stop all trading activities
        # Close all positions
        # Send emergency alerts
        # Save current state
        
        # Here you would integrate with your trading system
        # await trading_system.emergency_stop()
        # await telegram_reporter.send_emergency_alert("Bot emergency shutdown")
        
        # Stop health monitoring
        await self.stop_monitoring()

# Global health checker instance
health_checker = HealthChecker()

# Async context manager for health monitoring
class HealthMonitoringContext:
    def __init__(self, health_checker: HealthChecker):
        self.health_checker = health_checker
    
    async def __aenter__(self):
        await self.health_checker.start_monitoring()
        return self.health_checker
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.health_checker.stop_monitoring()

# Usage example
async def main():
    async with HealthMonitoringContext(health_checker) as hc:
        # Your bot logic here
        while True:
            # Check if emergency shutdown is needed
            should_shutdown, reason = hc.emergency_shutdown_check()
            if should_shutdown:
                print(f"Emergency shutdown: {reason}")
                await hc.handle_emergency_shutdown()
                break
            
            # Your trading logic here
            await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
