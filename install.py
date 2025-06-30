import os

# === 1. Papkalarni yaratish ===
os.makedirs('logs', exist_ok=True)

# === 2. main.py ===
main_py = """
import os
import time
import logging
from indicators import calculate_rsi, calculate_macd, calculate_bollinger_bands, calculate_atr
from risk_manager import RiskManager
from ai_sentiment import TextCleaner, SentimentAnalyzer
from api_manager import APIManager
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
API_KEYS = os.getenv("AI_API_KEYS").split(",")
api_manager = APIManager(API_KEYS)
risk_manager = RiskManager(
    initial_capital=float(os.getenv("INITIAL_CAPITAL", 10.0)),
    max_risk_percent=float(os.getenv("MAX_RISK_PERCENT", 2.0))
)
cleaner = TextCleaner()
analyzer = SentimentAnalyzer()

logger = logging.getLogger("bot_logger")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("logs/bot.log")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        logger.error(f"Telegram error: {e}")

def fetch_latest_ohlcv():
    data = {
        'open': np.random.uniform(20000, 21000, 100),
        'high': np.random.uniform(21000, 22000, 100),
        'low': np.random.uniform(19000, 20000, 100),
        'close': np.random.uniform(20000, 21000, 100),
        'volume': np.random.uniform(10, 100, 100)
    }
    df = pd.DataFrame(data)
    return df

def main_loop():
    while True:
        df = fetch_latest_ohlcv()
        df = calculate_rsi(df)
        df = calculate_macd(df)
        df = calculate_bollinger_bands(df)
        df = calculate_atr(df)

        last = df.iloc[-1]
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]

        signal = "HOLD"
        if (
            last['rsi'] < 30 and
            last['macd'] > last['macdsignal'] and
            last['close'] < last['bb_lower'] and
            last['volume'] > avg_volume
        ):
            news = "Bitcoin news!"
            cleaned = cleaner.clean_single_text(news)
            sentiment = analyzer.analyze_sentiment(cleaned)

            if sentiment['sentiment'] == 'positive':
                signal = "BUY"
            elif sentiment['sentiment'] == 'negative':
                signal = "HOLD"

        if signal == "BUY":
            entry = last['close']
            atr = last['atr']
            sl, tp = risk_manager.set_dynamic_sl_tp("BTCUSDT", entry, atr)
            qty = risk_manager.calculate_position_size(entry, sl)
            msg = f"✅ BUY SIGNAL! Qty: {qty:.4f} | SL: {sl:.2f} | TP: {tp:.2f}"
            send_telegram_message(msg)
            logger.info(msg)
        else:
            logger.info("HOLD...")

        time.sleep(60)

if __name__ == "__main__":
    main_loop()
"""

with open("main.py", "w") as f:
    f.write(main_py.strip())

# === 3. indicators.py ===
indicators_py = """
import pandas as pd
import numpy as np

def calculate_rsi(data, period=9):
    delta = data['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period-1, adjust=False).mean()
    avg_loss = loss.ewm(com=period-1, adjust=False).mean()
    rs = avg_gain / avg_loss
    data['rsi'] = 100 - (100 / (1 + rs))
    return data

def calculate_macd(data, fast=6, slow=13, signal=5):
    ema_fast = data['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['close'].ewm(span=slow, adjust=False).mean()
    data['macd'] = ema_fast - ema_slow
    data['macdsignal'] = data['macd'].ewm(span=signal, adjust=False).mean()
    data['macdhist'] = data['macd'] - data['macdsignal']
    return data

def calculate_bollinger_bands(data, period=20, std_dev=2):
    ma = data['close'].rolling(window=period).mean()
    std = data['close'].rolling(window=period).std()
    data['bb_middle'] = ma
    data['bb_upper'] = ma + std * std_dev
    data['bb_lower'] = ma - std * std_dev
    return data

def calculate_atr(data, period=7):
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    data['atr'] = true_range.ewm(com=period-1, adjust=False).mean()
    return data
"""
with open("indicators.py", "w") as f:
    f.write(indicators_py.strip())

# === 4. Qolgan fayllar ===
risk_manager_py = """import logging
logger = logging.getLogger("bot_logger")
class RiskManager:
    def __init__(self, initial_capital=10.0, max_risk_percent=2.0, emergency_stop_loss=20.0):
        self.current_capital = initial_capital
        self.max_risk_percent = max_risk_percent
    def calculate_position_size(self, entry_price, stop_loss_price):
        risk_amount = self.current_capital * (self.max_risk_percent / 100)
        risk_per_unit = abs(entry_price - stop_loss_price)
        if risk_per_unit <= 0:
            logger.error("Stop-loss noto'g'ri!")
            return 0
        qty = risk_amount / risk_per_unit
        return qty
    def set_dynamic_sl_tp(self, symbol, entry_price, atr_value, side='buy', risk_reward_ratio=2):
        sl_distance = atr_value * 1.5
        if side == 'buy':
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + sl_distance * risk_reward_ratio
        else:
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - sl_distance * risk_reward_ratio
        return stop_loss, take_profit
"""
with open("risk_manager.py", "w") as f:
    f.write(risk_manager_py.strip())

ai_sentiment_py = """import re
from textblob import TextBlob
class TextCleaner:
    def clean_single_text(self, text):
        text = re.sub(r"http\\\\S+", "", text)
        text = re.sub(r"[^\\\\w\\\\s]", "", text)
        return text.strip()
class SentimentAnalyzer:
    def analyze_sentiment(self, text):
        if not text:
            return {'sentiment': 'neutral'}
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            return {'sentiment': 'positive'}
        elif polarity < -0.1:
            return {'sentiment': 'negative'}
        return {'sentiment': 'neutral'}
"""
with open("ai_sentiment.py", "w") as f:
    f.write(ai_sentiment_py.strip())

api_manager_py = """import logging
logger = logging.getLogger("bot_logger")
class APIManager:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        self.current_key_index = 0
    def get_key(self):
        return self.api_keys[self.current_key_index]
    def rotate_key(self):
        self.current_key_index += 1
        if self.current_key_index >= len(self.api_keys):
            logger.error("Barcha API kalitlar tugadi.")
            return None
        return self.get_key()
"""
with open("api_manager.py", "w") as f:
    f.write(api_manager_py.strip())

with open(".env.example", "w") as f:
    f.write("TELEGRAM_TOKEN=YOUR_TELEGRAM_BOT_TOKEN\nTELEGRAM_CHAT_ID=YOUR_CHAT_ID\nAI_API_KEYS=key1,key2,key3\nINITIAL_CAPITAL=10.0\nMAX_RISK_PERCENT=2.0\n")

with open("requirements.txt", "w") as f:
    f.write("pandas\nnumpy\nrequests\npython-dotenv\ntextblob\n")

with open("Procfile", "w") as f:
    f.write("python main.py\n")

with open("README.md", "w") as f:
    f.write("# Scalping Bot\\n\\nAvval install.py ni ishga tushiring — u barcha fayllarni yaratadi.\\nKeyin GitHub push, Railway deploy. Log fayl `logs/bot.log`da.\\n")

print("✅ Barcha fayllar yaratildi!")
