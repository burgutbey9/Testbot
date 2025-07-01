"""
MAIN.PY
--------
Real-time signal ishlab chiqaruvchi asosiy fayl
"""

from indicators import calculate_rsi, calculate_macd, calculate_bollinger_bands, calculate_atr
from api_manager import APIManager
from ai_sentiment import clean_text, get_sentiment
from risk_manager import RiskManager
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Main")

# Namuna ma'lumot (real vaqtda o'qiladigan qism)
df = pd.read_csv('data/sample_data.csv')
df = calculate_rsi(df)
df = calculate_macd(df)
df = calculate_bollinger_bands(df)
df = calculate_atr(df)

# Hozirgi sham
row = df.iloc[-1]

# AI sentiment tekshirish
cleaned_text = clean_text("Bitcoin bugun kuchli o'smoqda!")
sentiment = get_sentiment(cleaned_text)

# Risk boshqarish
risk = RiskManager(10.0, 2.0)
entry_price = row['close']
atr = row['atr']

stop_loss = entry_price - atr * 1.5
tp = entry_price + atr * 3.0
position_size = risk.calculate_position_size(entry_price, stop_loss)

logger.info(f"AI Sentiment: {sentiment}")
logger.info(f"Signal: Entry={entry_price}, SL={stop_loss}, TP={tp}, Size={position_size}")

# Agar favqulodda stop bo'lsa
if risk.check_emergency_stop(8.0, False):
    risk.pause_trading()
