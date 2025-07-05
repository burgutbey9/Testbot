import os
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    """API konfiguratsiyalari"""
    inch_api_key: str
    hf_api_key: str
    gemini_api_key: str
    newsapi_key: str
    
    # Alchemy endpoints
    alchemy_eth: str
    alchemy_bnb: str
    alchemy_arb: str
    alchemy_polygon: str
    
    # API limits
    max_requests_per_minute: int = 60
    timeout_seconds: int = 30
    retry_attempts: int = 3
    
    def validate(self) -> bool:
        """API kalitlarini tekshirish"""
        required_keys = [
            self.inch_api_key, self.hf_api_key, 
            self.gemini_api_key, self.newsapi_key
        ]
        return all(key and len(key) > 10 for key in required_keys)

@dataclass
class TelegramConfig:
    """Telegram konfiguratsiyalari"""
    bot_token: str
    chat_id: str
    
    # Notification settings
    send_status_interval: int = 300  # 5 daqiqa
    send_trade_signals: bool = True
    send_errors: bool = True
    
    def validate(self) -> bool:
        return bool(self.bot_token and self.chat_id)

@dataclass
class TradingConfig:
    """Trading konfiguratsiyalari"""
    # Scalping parametrlari
    min_profit_threshold: float = 0.002  # 0.2%
    max_loss_threshold: float = 0.01    # 1%
    
    # Position management
    max_position_size: float = 0.1      # Portfolio ning 10%
    max_concurrent_trades: int = 3
    
    # DEX settings
    supported_dexes: List[str] = None
    supported_tokens: List[str] = None
    slippage_tolerance: float = 0.005   # 0.5%
    
    def __post_init__(self):
        if self.supported_dexes is None:
            self.supported_dexes = ['uniswap', 'sushiswap', 'pancakeswap']
        if self.supported_tokens is None:
            self.supported_tokens = ['WETH', 'USDC', 'USDT', 'DAI']

@dataclass
class AIConfig:
    """AI konfiguratsiyalari"""
    # Sentiment analysis
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    sentiment_threshold: float = 0.7
    
    # Strategy update
    strategy_update_interval: int = 604800  # 1 hafta
    backtest_days: int = 30
    
    # Learning parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    
    # Data sources
    news_sources: List[str] = None
    social_sources: List[str] = None
    
    def __post_init__(self):
        if self.news_sources is None:
            self.news_sources = ['cryptonews', 'cointelegraph', 'coindesk']
        if self.social_sources is None:
            self.social_sources = ['twitter', 'reddit']

class Config:
    """Asosiy konfiguratsiya klassi"""
    
    def __init__(self):
        self.load_environment()
        self.setup_paths()
        self.setup_logging()
        self.validate_config()
    
    def load_environment(self):
        """Muhit o'zgaruvchilarini yuklash"""
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            logger.warning("python-dotenv o'rnatilmagan")
        
        # API Configuration
        self.api = APIConfig(
            inch_api_key=self._get_env("INCH_API_KEY"),
            hf_api_key=self._get_env("HF_API_KEY"),
            gemini_api_key=self._get_env("GEMINI_API_KEY"),
            newsapi_key=self._get_env("NEWSAPI_KEY"),
            alchemy_eth=self._get_env("ALCHEMY_ETH"),
            alchemy_bnb=self._get_env("ALCHEMY_BNB"),
            alchemy_arb=self._get_env("ALCHEMY_ARB"),
            alchemy_polygon=self._get_env("ALCHEMY_POLYGON")
        )
        
        # Telegram Configuration
        self.telegram = TelegramConfig(
            bot_token=self._get_env("TELEGRAM_BOT_TOKEN"),
            chat_id=self._get_env("TELEGRAM_CHAT_ID")
        )
        
        # Trading Configuration
        self.trading = TradingConfig()
        
        # AI Configuration
        self.ai = AIConfig()
        
        # General settings
        self.DEBUG = self._get_env("DEBUG", "false").lower() == "true"
        self.CYCLE_INTERVAL = int(self._get_env("CYCLE_INTERVAL", "60"))
        self.LOG_LEVEL = self._get_env("LOG_LEVEL", "INFO")
    
    def _get_env(self, key: str, default: str = None) -> str:
        """Muhit o'zgaruvchisini olish"""
        value = os.getenv(key, default)
        if value is None:
            raise ValueError(f"Muhit o'zgaruvchisi topilmadi: {key}")
        return value
    
    def setup_paths(self):
        """Papka yo'llarini sozlash"""
        self.BASE_DIR = Path(__file__).parent
        self.LOGS_DIR = self.BASE_DIR / "logs"
        self.DATA_DIR = self.BASE_DIR / "data"
        self.MODELS_DIR = self.DATA_DIR / "ai_models"
        self.STRATEGIES_DIR = self.DATA_DIR / "strategies"
        self.BACKTEST_DIR = self.DATA_DIR / "backtest_results"
        
        # Papkalarni yaratish
        for path in [self.LOGS_DIR, self.DATA_DIR, self.MODELS_DIR, 
                    self.STRATEGIES_DIR, self.BACKTEST_DIR]:
            path.mkdir(exist_ok=True)
    
    def setup_logging(self):
        """Logging sozlash"""
        log_file = self.LOGS_DIR / "bot.log"
        logging.basicConfig(
            level=getattr(logging, self.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def validate_config(self):
        """Konfiguratsiyani tekshirish"""
        errors = []
        
        if not self.api.validate():
            errors.append("API kalitlari noto'g'ri yoki yo'q")
        
        if not self.telegram.validate():
            errors.append("Telegram konfiguratsiyasi noto'g'ri")
        
        if errors:
            raise ValueError(f"Konfiguratsiya xatolari: {', '.join(errors)}")
        
        logger.info("✅ Konfiguratsiya muvaffaqiyatli yuklandi")
    
    def get_network_config(self, network: str) -> Dict:
        """Tarmoq konfiguratsiyasini olish"""
        networks = {
            'ethereum': {'rpc': self.api.alchemy_eth, 'chain_id': 1},
            'bsc': {'rpc': self.api.alchemy_bnb, 'chain_id': 56},
            'arbitrum': {'rpc': self.api.alchemy_arb, 'chain_id': 42161},
            'polygon': {'rpc': self.api.alchemy_polygon, 'chain_id': 137}
        }
        return networks.get(network, {})
