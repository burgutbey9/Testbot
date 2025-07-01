"""
BACKTEST.PY
------------
Tarixiy ma'lumotlar asosida strategiyani sinovdan o'tkazadi
"""

import pandas as pd
from indicators import calculate_rsi, calculate_macd, calculate_bollinger_bands, calculate_atr
from trainer import Trainer
import logging

# Logger sozlash
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Backtest")

# Tarixiy ma'lumotni o'qish
df = pd.read_csv('data/sample_data.csv')
df = calculate_rsi(df)
df = calculate_macd(df)
df = calculate_bollinger_bands(df)
df = calculate_atr(df)

# Signal logikasini aniqlash
def signal_logic(row):
    signal = {}
    if (
        row['rsi'] < 30 and
        row['macd'] > row['macdsignal'] and
        row['close'] < row['bb_lower']
    ):
        signal['entry'] = row['close']
        signal['tp'] = row['close'] * 1.01  # 1% profit
        signal['sl'] = row['close'] * 0.99  # 1% stop loss
        return signal
    return None

# Backtest ishga tushirish
candles = df.to_dict('records')
trainer = Trainer()
result = trainer.backtest_signal(candles, signal_logic)

logger.info(f"Backtest natija: {result}")
