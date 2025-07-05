import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import aiohttp
import logging
from dataclasses import dataclass
from enum import Enum

class AlertType(Enum):
    INFO = "ℹ️"
    WARNING = "⚠️"
    ERROR = "❌"
    CRITICAL = "🚨"
    SUCCESS = "✅"
    TRADE = "📊"
    BALANCE = "💰"
    HEARTBEAT = "💓"

@dataclass
class TelegramMessage:
    message: str
    alert_type: AlertType
    timestamp: datetime
    priority: int = 1  # 1=low, 2=medium, 3=high, 4=critical

class TelegramReporter:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}"
        
        # Message management
        self.message_queue: List[TelegramMessage] = []
        self.last_heartbeat = time.time()
        self.alert_cooldown: Dict[str, float] = {}
        self.failed_messages: List[TelegramMessage] = []
        
        # Rate limiting
        self.rate_limit_window = 60  # seconds
        self.max_messages_per_minute = 20
        self.sent_messages_timestamps: List[float] = []
        
        # Configuration
        self.batch_size = 5
        self.batch_timeout = 30  # seconds
        self.cooldown_periods = {
            AlertType.INFO: 60,      # 1 minute
            AlertType.WARNING: 300,  # 5 minutes  
            AlertType.ERROR: 600,    # 10 minutes
            AlertType.CRITICAL: 0,   # No cooldown
            AlertType.TRADE: 30,     # 30 seconds
            AlertType.BALANCE: 300,  # 5 minutes
        }
        
        # Statistics
        self.stats = {
            'messages_sent': 0,
            'messages_failed': 0,
            'last_success': None,
            'last_failure': None,
            'uptime_start': datetime.now()
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Start background tasks
        self.running = True
        asyncio.create_task(self._message_processor())
        asyncio.create_task(self._heartbeat_monitor())
        
    async def send_message(self, message: str, alert_type: AlertType = AlertType.INFO, 
                          priority: int = 1, bypass_cooldown: bool = False) -> bool:
        """Send message to Telegram with advanced features"""
        
        # Check cooldown
        if not bypass_cooldown and self._is_cooldown_active(alert_type):
            self.logger.debug(f"Message blocked by cooldown: {alert_type.name}")
            return False
            
        # Create message object
        telegram_msg = TelegramMessage(
            message=message,
            alert_type=alert_type,
            timestamp=datetime.now(),
            priority=priority
        )
        
        # Add to queue based on priority
        if priority >= 4:  # Critical - send immediately
            return await self._send_immediately(telegram_msg)
        else:
            self.message_queue.append(telegram_msg)
            self._sort_queue_by_priority()
            return True
    
    async def send_trade_alert(self, trade_data: Dict) -> bool:
        """Send formatted trade alert"""
        emoji = "🟢" if trade_data.get('side') == 'buy' else "🔴"
        
        message = f"""{emoji} **TRADE EXECUTED**
        
🏷️ **Pair:** {trade_data.get('symbol', 'N/A')}
📊 **Side:** {trade_data.get('side', 'N/A').upper()}
💰 **Size:** ${trade_data.get('size', 0):,.2f}
💵 **Price:** ${trade_data.get('price', 0):,.4f}
📈 **PnL:** ${trade_data.get('pnl', 0):,.2f}
🤖 **AI Confidence:** {trade_data.get('ai_confidence', 0):.1%}
⏰ **Time:** {datetime.now().strftime('%H:%M:%S')}"""
        
        return await self.send_message(message, AlertType.TRADE, priority=2)
    
    async def send_balance_update(self, balance_data: Dict) -> bool:
        """Send formatted balance update"""
        total_balance = balance_data.get('total_balance', 0)
        available_balance = balance_data.get('available_balance', 0)
        pnl_24h = balance_data.get('pnl_24h', 0)
        
        pnl_emoji = "📈" if pnl_24h >= 0 else "📉"
        
        message = f"""💰 **BALANCE UPDATE**
        
💵 **Total Balance:** ${total_balance:,.2f}
💳 **Available:** ${available_balance:,.2f}
{pnl_emoji} **24h PnL:** ${pnl_24h:,.2f} ({pnl_24h/total_balance*100:.1f}%)
📊 **Open Positions:** {balance_data.get('open_positions', 0)}
⏰ **Updated:** {datetime.now().strftime('%H:%M:%S')}"""
        
        return await self.send_message(message, AlertType.BALANCE, priority=2)
    
    async def send_error_alert(self, error_msg: str, error_type: str = "SYSTEM_ERROR") -> bool:
        """Send formatted error alert"""
        message = f"""❌ **{error_type}**
        
🚨 **Error:** {error_msg}
⏰ **Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🔧 **Action:** Check logs for details"""
        
        return await self.send_message(message, AlertType.ERROR, priority=3)
    
    async def send_system_status(self) -> bool:
        """Send system status report"""
        uptime = datetime.now() - self.stats['uptime_start']
        
        message = f"""🤖 **SYSTEM STATUS**
        
✅ **Status:** Running
⏱️ **Uptime:** {str(uptime).split('.')[0]}
📨 **Messages Sent:** {self.stats['messages_sent']}
❌ **Failed Messages:** {self.stats['messages_failed']}
📊 **Success Rate:** {self._calculate_success_rate():.1%}
💓 **Last Heartbeat:** {datetime.fromtimestamp(self.last_heartbeat).strftime('%H:%M:%S')}"""
        
        return await self.send_message(message, AlertType.HEARTBEAT, priority=1)
    
    async def send_critical_alert(self, message: str) -> bool:
        """Send critical alert immediately"""
        critical_msg = f"""🚨 **CRITICAL ALERT** 🚨
        
{message}
        
⏰ **Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🔥 **Immediate action required!**"""
        
        return await self.send_message(critical_msg, AlertType.CRITICAL, priority=4, bypass_cooldown=True)
    
    async def _send_immediately(self, telegram_msg: TelegramMessage) -> bool:
        """Send message immediately without queuing"""
        if not self._check_rate_limit():
            self.logger.warning("Rate limit exceeded, queuing critical message")
            self.message_queue.insert(0, telegram_msg)
            return False
            
        return await self._send_to_telegram(telegram_msg)
    
    async def _send_to_telegram(self, telegram_msg: TelegramMessage) -> bool:
        """Actually send message to Telegram API"""
        try:
            formatted_message = f"{telegram_msg.alert_type.value} {telegram_msg.message}"
            
            payload = {
                'chat_id': self.chat_id,
                'text': formatted_message,
                'parse_mode': 'Markdown',
                'disable_web_page_preview': True
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/sendMessage",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    
                    if response.status == 200:
                        self.stats['messages_sent'] += 1
                        self.stats['last_success'] = datetime.now()
                        self._set_cooldown(telegram_msg.alert_type)
                        self._update_rate_limit()
                        return True
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Telegram API error: {response.status} - {error_text}")
                        self.stats['messages_failed'] += 1
                        self.stats['last_failure'] = datetime.now()
                        return False
                        
        except Exception as e:
            self.logger.error(f"Failed to send Telegram message: {str(e)}")
            self.stats['messages_failed'] += 1
            self.stats['last_failure'] = datetime.now()
            return False
    
    async def _message_processor(self):
        """Background task to process message queue"""
        while self.running:
            try:
                if self.message_queue:
                    # Process batch of messages
                    batch = self.message_queue[:self.batch_size]
                    self.message_queue = self.message_queue[self.batch_size:]
                    
                    for msg in batch:
                        if self._check_rate_limit():
                            await self._send_to_telegram(msg)
                            await asyncio.sleep(0.5)  # Small delay between messages
                        else:
                            # Re-queue if rate limited
                            self.message_queue.insert(0, msg)
                            break
                
                await asyncio.sleep(1)  # Check queue every second
                
            except Exception as e:
                self.logger.error(f"Error in message processor: {str(e)}")
                await asyncio.sleep(5)
    
    async def _heartbeat_monitor(self):
        """Background task to monitor system health"""
        while self.running:
            try:
                current_time = time.time()
                
                # Update heartbeat
                self.last_heartbeat = current_time
                
                # Send periodic status updates (every 6 hours)
                if current_time % 21600 < 60:  # 6 hours = 21600 seconds
                    await self.send_system_status()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat monitor: {str(e)}")
                await asyncio.sleep(60)
    
    def _is_cooldown_active(self, alert_type: AlertType) -> bool:
        """Check if alert type is in cooldown period"""
        if alert_type not in self.cooldown_periods:
            return False
            
        cooldown_period = self.cooldown_periods[alert_type]
        if cooldown_period == 0:  # No cooldown
            return False
            
        last_sent = self.alert_cooldown.get(alert_type.name, 0)
        return (time.time() - last_sent) < cooldown_period
    
    def _set_cooldown(self, alert_type: AlertType):
        """Set cooldown for alert type"""
        self.alert_cooldown[alert_type.name] = time.time()
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        current_time = time.time()
        
        # Remove old timestamps
        self.sent_messages_timestamps = [
            ts for ts in self.sent_messages_timestamps 
            if current_time - ts < self.rate_limit_window
        ]
        
        return len(self.sent_messages_timestamps) < self.max_messages_per_minute
    
    def _update_rate_limit(self):
        """Update rate limit tracking"""
        self.sent_messages_timestamps.append(time.time())
    
    def _sort_queue_by_priority(self):
        """Sort message queue by priority (highest first)"""
        self.message_queue.sort(key=lambda x: x.priority, reverse=True)
    
    def _calculate_success_rate(self) -> float:
        """Calculate message success rate"""
        total_messages = self.stats['messages_sent'] + self.stats['messages_failed']
        if total_messages == 0:
            return 100.0
        return (self.stats['messages_sent'] / total_messages) * 100
    
    def get_stats(self) -> Dict:
        """Get reporter statistics"""
        return {
            **self.stats,
            'queue_size': len(self.message_queue),
            'failed_queue_size': len(self.failed_messages),
            'cooldown_status': {
                alert_type.name: self._is_cooldown_active(alert_type)
                for alert_type in AlertType
            },
            'rate_limit_usage': f"{len(self.sent_messages_timestamps)}/{self.max_messages_per_minute}"
        }
    
    async def shutdown(self):
        """Gracefully shutdown the reporter"""
        self.running = False
        
        # Send remaining messages
        if self.message_queue:
            self.logger.info(f"Sending {len(self.message_queue)} remaining messages...")
            for msg in self.message_queue[:10]:  # Send max 10 on shutdown
                await self._send_to_telegram(msg)
                await asyncio.sleep(0.5)
        
        self.logger.info("Telegram reporter shutdown complete")

# Usage example
if __name__ == "__main__":
    async def main():
        reporter = TelegramReporter("YOUR_BOT_TOKEN", "YOUR_CHAT_ID")
        
        # Test different message types
        await reporter.send_message("Bot started successfully!", AlertType.SUCCESS)
        
        # Test trade alert
        trade_data = {
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'size': 1000,
            'price': 50000,
            'pnl': 25.50,
            'ai_confidence': 0.85
        }
        await reporter.send_trade_alert(trade_data)
        
        # Test balance update
        balance_data = {
            'total_balance': 10000,
            'available_balance': 9500,
            'pnl_24h': 150,
            'open_positions': 3
        }
        await reporter.send_balance_update(balance_data)
        
        # Test error alert
        await reporter.send_error_alert("Database connection failed", "DATABASE_ERROR")
        
        # Test critical alert
        await reporter.send_critical_alert("Bot stopped due to critical error!")
        
        # Wait a bit then shutdown
        await asyncio.sleep(10)
        await reporter.shutdown()
    
    asyncio.run(main())
