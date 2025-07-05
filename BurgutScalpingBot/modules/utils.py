import logging
import requests
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
import asyncio
import aiohttp
from pathlib import Path

class BurgutLogger:
    """Burgut Scalping Bot uchun maxsus logger sinfi"""
    
    def __init__(self, log_level: str = "INFO"):
        self.log_level = getattr(logging, log_level.upper())
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Logger sozlash"""
        # Logs papkasini yaratish
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Logger konfiguratsiyasi
        logger = logging.getLogger("BurgutScalpingBot")
        logger.setLevel(self.log_level)
        
        # Agar handler allaqachon mavjud bo'lsa, qaytarish
        if logger.handlers:
            return logger
        
        # Fayl handler
        file_handler = logging.FileHandler(
            log_dir / "bot.log", 
            encoding='utf-8'
        )
        file_handler.setLevel(self.log_level)
        
        # Konsol handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Handler'larni qo'shish
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def info(self, message: str, **kwargs):
        """Info log"""
        self.logger.info(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """Error log"""
        self.logger.error(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Warning log"""
        self.logger.warning(message, extra=kwargs)
    
    def debug(self, message: str, **kwargs):
        """Debug log"""
        self.logger.debug(message, extra=kwargs)

class TelegramNotifier:
    """Telegram xabar yuborish sinfi"""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.logger = BurgutLogger().logger
    
    async def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """Asinxron Telegram xabar yuborish"""
        url = f"{self.base_url}/sendMessage"
        payload = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': parse_mode
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=payload) as response:
                    if response.status == 200:
                        self.logger.info(f"Telegram xabar yuborildi: {message[:50]}...")
                        return True
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Telegram xatolik: {response.status} - {error_text}")
                        return False
        except Exception as e:
            self.logger.error(f"Telegram xabar yuborishda xatolik: {e}")
            return False
    
    def send_message_sync(self, message: str, parse_mode: str = "HTML") -> bool:
        """Sinxron Telegram xabar yuborish"""
        url = f"{self.base_url}/sendMessage"
        payload = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': parse_mode
        }
        
        try:
            response = requests.post(url, data=payload, timeout=10)
            response.raise_for_status()
            self.logger.info(f"Telegram xabar yuborildi: {message[:50]}...")
            return True
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Telegram xatolik: {e}")
            return False
    
    async def send_status_update(self, status: str, balance: float, profit_loss: float):
        """Bot holati va balans haqida xabar"""
        message = f"""
🤖 <b>Burgut Scalping Bot Status</b>

📊 <b>Holat:</b> {status}
💰 <b>Balans:</b> ${balance:.2f}
📈 <b>P&L:</b> ${profit_loss:.2f}
🕒 <b>Vaqt:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        await self.send_message(message)

class DataManager:
    """Ma'lumotlar bilan ishlash sinfi"""
    
    def __init__(self):
        self.logger = BurgutLogger().logger
        self._create_directories()
    
    def _create_directories(self):
        """Kerakli papkalarni yaratish"""
        directories = [
            "data/strategies",
            "data/backtest_results", 
            "data/ai_models",
            "data/historical_data",
            "logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def save_json(self, data: Dict[str, Any], filepath: str) -> bool:
        """JSON fayl saqlash"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"JSON saqlandi: {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"JSON saqlashda xatolik: {e}")
            return False
    
    def load_json(self, filepath: str) -> Optional[Dict[str, Any]]:
        """JSON fayl yuklash"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.info(f"JSON yuklandi: {filepath}")
            return data
        except FileNotFoundError:
            self.logger.warning(f"JSON fayl topilmadi: {filepath}")
            return None
        except Exception as e:
            self.logger.error(f"JSON yuklashda xatolik: {e}")
            return None
    
    def save_backtest_results(self, results: Dict[str, Any], strategy_name: str) -> str:
        """Backtest natijalarini saqlash"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_{strategy_name}_{timestamp}.json"
        filepath = f"data/backtest_results/{filename}"
        
        if self.save_json(results, filepath):
            return filepath
        return ""

class PriceValidator:
    """Narx validatsiyasi va tekshirish"""
    
    @staticmethod
    def validate_price(price: float, min_price: float = 0.000001) -> bool:
        """Narx validatsiyasi"""
        return isinstance(price, (int, float)) and price > min_price
    
    @staticmethod
    def calculate_slippage(expected_price: float, actual_price: float) -> float:
        """Slippage hisoblash"""
        if expected_price == 0:
            return 0.0
        return abs(actual_price - expected_price) / expected_price * 100
    
    @staticmethod
    def format_price(price: float, decimals: int = 6) -> str:
        """Narxni formatlash"""
        return f"{price:.{decimals}f}"

class PerformanceMonitor:
    """Ishlash ko'rsatkichlarini kuzatish"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.trades_count = 0
        self.successful_trades = 0
        self.total_profit = 0.0
        self.logger = BurgutLogger().logger
    
    def record_trade(self, success: bool, profit: float = 0.0):
        """Savdoni qayd etish"""
        self.trades_count += 1
        if success:
            self.successful_trades += 1
            self.total_profit += profit
            self.logger.info(f"Muvaffaqiyatli savdo: Profit ${profit:.2f}")
        else:
            self.logger.warning("Muvaffaqiyatsiz savdo")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Statistika olish"""
        runtime = datetime.now() - self.start_time
        success_rate = (self.successful_trades / self.trades_count * 100) if self.trades_count > 0 else 0
        
        return {
            "runtime_minutes": runtime.total_seconds() / 60,
            "total_trades": self.trades_count,
            "successful_trades": self.successful_trades,
            "success_rate": success_rate,
            "total_profit": self.total_profit,
            "avg_profit_per_trade": self.total_profit / self.successful_trades if self.successful_trades > 0 else 0
        }

class ConfigValidator:
    """Konfiguratsiya validatsiyasi"""
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> List[str]:
        """Konfiguratsiyani tekshirish"""
        errors = []
        
        # API kalitlari tekshiruvi
        required_keys = [
            "ONEINCH_API_KEY",
            "ALCHEMY_API_KEY", 
            "NEWS_API_KEY",
            "TELEGRAM_BOT_TOKEN",
            "TELEGRAM_CHAT_ID"
        ]
        
        for key in required_keys:
            if not config.get(key):
                errors.append(f"Majburiy kalit yo'q: {key}")
        
        # Raqamli qiymatlar tekshiruvi
        if config.get("MIN_TRADE_AMOUNT", 0) <= 0:
            errors.append("MIN_TRADE_AMOUNT musbat bo'lishi kerak")
            
        if config.get("MAX_SLIPPAGE", 0) <= 0:
            errors.append("MAX_SLIPPAGE musbat bo'lishi kerak")
        
        return errors

# Utility funksiyalar
def setup_logger(log_level: str = "INFO") -> logging.Logger:
    """Eski interfeys uchun wrapper"""
    return BurgutLogger(log_level).logger

def send_telegram_message(bot_token: str, chat_id: str, message: str) -> bool:
    """Eski interfeys uchun wrapper"""
    notifier = TelegramNotifier(bot_token, chat_id)
    return notifier.send_message_sync(message)

async def send_telegram_message_async(bot_token: str, chat_id: str, message: str) -> bool:
    """Asinxron telegram xabar yuborish"""
    notifier = TelegramNotifier(bot_token, chat_id)
    return await notifier.send_message(message)

def format_currency(amount: float, currency: str = "USD") -> str:
    """Valyutani formatlash"""
    return f"${amount:.2f}" if currency == "USD" else f"{amount:.6f} {currency}"

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Foiz o'zgarishini hisoblash"""
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100
