"""
Configuration validation module for BurgutScalpingBot
Startup paytida barcha konfiguratsiyalarni tekshiradi
"""

import os
import re
import requests
from typing import Dict, List, Tuple, Optional
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)

class ConfigValidator:
    """Konfiguratsiya validatori"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.required_env_vars = [
            'TELEGRAM_BOT_TOKEN',
            'TELEGRAM_CHAT_ID',
            'DEX_API_KEY',
            'DEX_SECRET_KEY',
            'TRADING_PAIR',
            'BASE_CURRENCY',
            'QUOTE_CURRENCY',
            'MAX_POSITION_SIZE',
            'RISK_PERCENTAGE',
            'AI_MODEL_PATH'
        ]
        
    def validate_all(self) -> Dict[str, any]:
        """Barcha konfiguratsiyalarni tekshiradi"""
        logger.info("Starting configuration validation...")
        
        # Environment variables validation
        self.validate_env_variables()
        
        # API connections validation
        self.validate_api_connections()
        
        # Trading parameters validation
        self.validate_trading_parameters()
        
        # File paths validation
        self.validate_file_paths()
        
        # AI model validation
        self.validate_ai_models()
        
        # Telegram validation
        self.validate_telegram_config()
        
        result = {
            'success': len(self.errors) == 0,
            'errors': self.errors,
            'warnings': self.warnings,
            'total_checks': len(self.required_env_vars) + 6
        }
        
        if result['success']:
            logger.info("✅ All configuration validations passed")
        else:
            logger.error(f"❌ Configuration validation failed: {len(self.errors)} errors")
            
        return result
    
    def validate_env_variables(self) -> None:
        """Environment variables tekshirish"""
        logger.info("Validating environment variables...")
        
        missing_vars = []
        for var in self.required_env_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            self.errors.append(f"Missing environment variables: {', '.join(missing_vars)}")
        
        # Specific format validations
        self._validate_telegram_token()
        self._validate_trading_pair()
        self._validate_numeric_configs()
    
    def _validate_telegram_token(self) -> None:
        """Telegram bot token format tekshirish"""
        token = os.getenv('TELEGRAM_BOT_TOKEN')
        if token:
            # Telegram bot token format: 123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11
            pattern = r'^\d{8,10}:[a-zA-Z0-9_-]{35}$'
            if not re.match(pattern, token):
                self.errors.append("Invalid TELEGRAM_BOT_TOKEN format")
    
    def _validate_trading_pair(self) -> None:
        """Trading pair format tekshirish"""
        pair = os.getenv('TRADING_PAIR')
        if pair:
            # Format: BTC/USDT, ETH/USDT, etc.
            pattern = r'^[A-Z]{3,5}/[A-Z]{3,5}$'
            if not re.match(pattern, pair):
                self.errors.append("Invalid TRADING_PAIR format (should be like BTC/USDT)")
    
    def _validate_numeric_configs(self) -> None:
        """Numeric konfiguratsiyalarni tekshirish"""
        numeric_configs = {
            'MAX_POSITION_SIZE': (0.01, 10000.0),
            'RISK_PERCENTAGE': (0.1, 10.0),
            'STOP_LOSS_PERCENTAGE': (0.1, 20.0),
            'TAKE_PROFIT_PERCENTAGE': (0.1, 50.0)
        }
        
        for config_name, (min_val, max_val) in numeric_configs.items():
            value_str = os.getenv(config_name)
            if value_str:
                try:
                    value = float(value_str)
                    if not (min_val <= value <= max_val):
                        self.errors.append(f"{config_name} must be between {min_val} and {max_val}")
                except ValueError:
                    self.errors.append(f"{config_name} must be a valid number")
    
    def validate_api_connections(self) -> None:
        """API ulanishlarini tekshirish"""
        logger.info("Validating API connections...")
        
        # DEX API test
        self._test_dex_api()
        
        # Telegram API test
        self._test_telegram_api()
        
        # AI service test (if using external AI)
        self._test_ai_service()
    
    def _test_dex_api(self) -> None:
        """DEX API ulanishini tekshirish"""
        api_key = os.getenv('DEX_API_KEY')
        secret_key = os.getenv('DEX_SECRET_KEY')
        base_url = os.getenv('DEX_BASE_URL', 'https://api.dex.com')
        
        if not all([api_key, secret_key]):
            self.errors.append("DEX API credentials missing")
            return
        
        try:
            # Test endpoint call
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            response = requests.get(
                f"{base_url}/v1/account/balance",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 401:
                self.errors.append("DEX API authentication failed")
            elif response.status_code != 200:
                self.warnings.append(f"DEX API returned status {response.status_code}")
            else:
                logger.info("✅ DEX API connection successful")
                
        except requests.exceptions.RequestException as e:
            self.errors.append(f"DEX API connection failed: {str(e)}")
    
    def _test_telegram_api(self) -> None:
        """Telegram API ulanishini tekshirish"""
        token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not all([token, chat_id]):
            self.errors.append("Telegram credentials missing")
            return
        
        try:
            # Test bot info
            response = requests.get(
                f"https://api.telegram.org/bot{token}/getMe",
                timeout=10
            )
            
            if response.status_code != 200:
                self.errors.append("Telegram bot token invalid")
                return
            
            # Test chat access
            test_response = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={
                    'chat_id': chat_id,
                    'text': '🤖 BurgutScalpingBot configuration test - SUCCESS',
                    'parse_mode': 'HTML'
                },
                timeout=10
            )
            
            if test_response.status_code != 200:
                self.errors.append("Telegram chat access failed")
            else:
                logger.info("✅ Telegram API connection successful")
                
        except requests.exceptions.RequestException as e:
            self.errors.append(f"Telegram API connection failed: {str(e)}")
    
    def _test_ai_service(self) -> None:
        """AI xizmatini tekshirish"""
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        
        if not ai_endpoint:
            self.warnings.append("AI service endpoint not configured")
            return
        
        try:
            response = requests.get(
                f"{ai_endpoint}/health",
                timeout=5
            )
            
            if response.status_code != 200:
                self.warnings.append("AI service health check failed")
            else:
                logger.info("✅ AI service connection successful")
                
        except requests.exceptions.RequestException as e:
            self.warnings.append(f"AI service connection failed: {str(e)}")
    
    def validate_trading_parameters(self) -> None:
        """Trading parametrlarini tekshirish"""
        logger.info("Validating trading parameters...")
        
        # Risk management validations
        risk_percentage = float(os.getenv('RISK_PERCENTAGE', '0'))
        max_position_size = float(os.getenv('MAX_POSITION_SIZE', '0'))
        
        if risk_percentage > 5.0:
            self.warnings.append("Risk percentage is high (>5%)")
        
        if max_position_size > 1000.0:
            self.warnings.append("Max position size is very high")
        
        # Stop loss and take profit validation
        stop_loss = float(os.getenv('STOP_LOSS_PERCENTAGE', '0'))
        take_profit = float(os.getenv('TAKE_PROFIT_PERCENTAGE', '0'))
        
        if stop_loss > take_profit:
            self.errors.append("Stop loss percentage cannot be greater than take profit")
        
        # Minimum spread validation
        min_spread = float(os.getenv('MIN_SPREAD_PERCENTAGE', '0.01'))
        if min_spread < 0.001:
            self.warnings.append("Minimum spread is very low - may cause frequent trades")
    
    def validate_file_paths(self) -> None:
        """Fayl yo'llarini tekshirish"""
        logger.info("Validating file paths...")
        
        important_files = [
            'AI_MODEL_PATH',
            'BACKTEST_DATA_PATH',
            'LOG_FILE_PATH'
        ]
        
        for file_env in important_files:
            file_path = os.getenv(file_env)
            if file_path:
                if not os.path.exists(file_path):
                    self.errors.append(f"File not found: {file_path} ({file_env})")
                elif not os.access(file_path, os.R_OK):
                    self.errors.append(f"File not readable: {file_path} ({file_env})")
    
    def validate_ai_models(self) -> None:
        """AI modellarini tekshirish"""
        logger.info("Validating AI models...")
        
        model_path = os.getenv('AI_MODEL_PATH')
        if model_path and os.path.exists(model_path):
            try:
                # Model fayl hajmini tekshirish
                file_size = os.path.getsize(model_path)
                if file_size < 1024:  # 1KB dan kichik
                    self.errors.append("AI model file is too small - may be corrupted")
                elif file_size > 500 * 1024 * 1024:  # 500MB dan katta
                    self.warnings.append("AI model file is very large - may cause memory issues")
                
                # Model format tekshirish
                if not model_path.endswith(('.pkl', '.joblib', '.h5', '.pb')):
                    self.warnings.append("AI model format may not be supported")
                    
            except Exception as e:
                self.errors.append(f"AI model validation failed: {str(e)}")
    
    def validate_telegram_config(self) -> None:
        """Telegram konfiguratsiyasini tekshirish"""
        logger.info("Validating Telegram configuration...")
        
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        if chat_id:
            try:
                # Chat ID format tekshirish
                if not (chat_id.startswith('-') or chat_id.isdigit()):
                    self.errors.append("Invalid Telegram chat ID format")
                
                # Chat ID uzunligini tekshirish
                if len(chat_id) < 5:
                    self.errors.append("Telegram chat ID is too short")
                    
            except Exception as e:
                self.errors.append(f"Telegram config validation failed: {str(e)}")
        
        # Telegram reporting intervals
        report_interval = int(os.getenv('TELEGRAM_REPORT_INTERVAL', '300'))
        if report_interval < 60:
            self.warnings.append("Telegram report interval is very frequent (<1 min)")
        elif report_interval > 3600:
            self.warnings.append("Telegram report interval is very long (>1 hour)")

def validate_startup_config() -> bool:
    """Startup configuration validator"""
    validator = ConfigValidator()
    result = validator.validate_all()
    
    if not result['success']:
        logger.error("Configuration validation failed!")
        for error in result['errors']:
            logger.error(f"ERROR: {error}")
        
        for warning in result['warnings']:
            logger.warning(f"WARNING: {warning}")
        
        return False
    
    # Warninglar bo'lsa ham davom etish
    if result['warnings']:
        logger.warning("Configuration warnings found:")
        for warning in result['warnings']:
            logger.warning(f"WARNING: {warning}")
    
    return True

if __name__ == "__main__":
    # Test run
    logging.basicConfig(level=logging.INFO)
    success = validate_startup_config()
    print(f"Validation result: {'PASSED' if success else 'FAILED'}")
