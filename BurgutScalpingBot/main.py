import asyncio
import logging
import signal
import sys
from typing import Dict, Any
from datetime import datetime
import traceback

from modules.orderflow import OrderFlowManager
from modules.ai_sentiment import SentimentAnalyzer
from modules.api_manager import APIManager
from modules.utils import TelegramNotifier
from modules.backtest import BacktestEngine
from modules.strategy_manager import StrategyManager
from config import Config

# Logging sozlamasi
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BurgutScalpingBot:
    def __init__(self):
        self.config = Config()
        self.running = False
        self.components = {}
        self.last_health_check = datetime.now()
        
    async def initialize(self):
        """Komponentlarni boshlash"""
        try:
            logger.info("🚀 BurgutScalpingBot boshlanyapti...")
            
            # API Manager
            self.components['api_manager'] = APIManager(self.config)
            await self.components['api_manager'].initialize()
            
            # Telegram Notifier
            self.components['telegram'] = TelegramNotifier(self.config)
            await self.components['telegram'].initialize()
            
            # Order Flow Manager
            self.components['orderflow'] = OrderFlowManager(self.config)
            await self.components['orderflow'].initialize()
            
            # AI Sentiment Analyzer
            self.components['sentiment'] = SentimentAnalyzer(self.config)
            await self.components['sentiment'].initialize()
            
            # Strategy Manager
            self.components['strategy'] = StrategyManager(self.config)
            await self.components['strategy'].initialize()
            
            # Backtest Engine
            self.components['backtest'] = BacktestEngine(self.config)
            
            logger.info("✅ Barcha komponentlar muvaffaqiyatli boshlandi")
            await self.components['telegram'].send_message("🟢 Bot muvaffaqiyatli boshlandi")
            
        except Exception as e:
            logger.error(f"❌ Boshlashda xato: {str(e)}")
            await self.emergency_shutdown()
            sys.exit(1)
    
    async def run_cycle(self):
        """Asosiy ish tsikli"""
        try:
            # Health check
            await self.health_check()
            
            # Order Flow tahlili
            orderflow_data = await self.components['orderflow'].analyze()
            
            # Sentiment tahlili
            sentiment_data = await self.components['sentiment'].analyze()
            
            # Strategiya yangilash
            strategy_signal = await self.components['strategy'].generate_signal(
                orderflow_data, sentiment_data
            )
            
            # Trading signal
            if strategy_signal['action'] != 'HOLD':
                await self.execute_trade(strategy_signal)
            
            # Status yuborish
            await self.send_status_update(orderflow_data, sentiment_data, strategy_signal)
            
        except Exception as e:
            logger.error(f"❌ Tsikl xatosi: {str(e)}")
            logger.error(traceback.format_exc())
            await self.components['telegram'].send_message(f"⚠️ Tsikl xatosi: {str(e)}")
    
    async def health_check(self):
        """Komponentlar sog'ligini tekshirish"""
        now = datetime.now()
        if (now - self.last_health_check).seconds < 300:  # 5 daqiqa
            return
            
        for name, component in self.components.items():
            try:
                if hasattr(component, 'health_check'):
                    await component.health_check()
                logger.debug(f"✅ {name} component sog'lom")
            except Exception as e:
                logger.error(f"❌ {name} component xatosi: {str(e)}")
                await self.components['telegram'].send_message(f"⚠️ {name} xatosi: {str(e)}")
        
        self.last_health_check = now
    
    async def execute_trade(self, signal: Dict[str, Any]):
        """Savdo buyrug'ini bajarish"""
        try:
            logger.info(f"📊 Trade signal: {signal}")
            # TODO: Real trading logic
            await self.components['telegram'].send_message(f"📊 Signal: {signal['action']}")
        except Exception as e:
            logger.error(f"❌ Trade xatosi: {str(e)}")
    
    async def send_status_update(self, orderflow, sentiment, strategy):
        """Status yangilanishi yuborish"""
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'orderflow': orderflow,
                'sentiment': sentiment,
                'strategy': strategy,
                'health': 'OK'
            }
            await self.components['telegram'].send_status(status)
        except Exception as e:
            logger.error(f"❌ Status yuborishda xato: {str(e)}")
    
    async def emergency_shutdown(self):
        """Favqulodda to'xtatish"""
        logger.warning("🚨 Favqulodda to'xtatish...")
        if 'telegram' in self.components:
            await self.components['telegram'].send_message("🔴 Bot favqulodda to'xtatildi")
        self.running = False
    
    async def graceful_shutdown(self):
        """Nazokat bilan to'xtatish"""
        logger.info("🛑 Nazokat bilan to'xtatish...")
        self.running = False
        
        for name, component in self.components.items():
            try:
                if hasattr(component, 'shutdown'):
                    await component.shutdown()
                logger.info(f"✅ {name} to'xtatildi")
            except Exception as e:
                logger.error(f"❌ {name} to'xtatishda xato: {str(e)}")
        
        if 'telegram' in self.components:
            await self.components['telegram'].send_message("🟡 Bot nazokat bilan to'xtatildi")
    
    async def run(self):
        """Asosiy ishga tushirish"""
        self.running = True
        
        # Signal handlers
        def signal_handler(sig, frame):
            logger.info(f"Signal {sig} qabul qilindi")
            asyncio.create_task(self.graceful_shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            while self.running:
                await self.run_cycle()
                await asyncio.sleep(self.config.CYCLE_INTERVAL)
        except KeyboardInterrupt:
            await self.graceful_shutdown()
        except Exception as e:
            logger.error(f"❌ Kutilmagan xato: {str(e)}")
            await self.emergency_shutdown()

async def main():
    bot = BurgutScalpingBot()
    await bot.initialize()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())
