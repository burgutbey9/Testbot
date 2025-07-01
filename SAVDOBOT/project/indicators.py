"""
INDICATORS.PY
--------------
Texnik indikatorlar:
- RSI
- MACD
- Bollinger Bands
- ATR
"""

import pandas as pd
import numpy as np

def calculate_rsi(data: pd.DataFrame, period: int = 9) -> pd.Series:
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data: pd.DataFrame, fast: int = 6, slow: int = 13, signal: int = 5) -> pd.DataFrame:
    ema_fast = data['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['close'].ewm(span=slow, adjust=False).mean()
    data['macd'] = ema_fast - ema_slow
    data['macdsignal'] = data['macd'].ewm(span=signal, adjust=False).mean()
    return data

def calculate_bollinger_bands(data: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    ma = data['close'].rolling(window=period).mean()
    std = data['close'].rolling(window=period).std()
    data['bb_middle'] = ma
    data['bb_upper'] = ma + std * std_dev
    data['bb_lower'] = ma - std * std_dev
    return data

def calculate_atr(data: pd.DataFrame, period: int = 7) -> pd.Series:
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df['rsi'] = calculate_rsi(df)
    df = calculate_macd(df)
    df = calculate_bollinger_bands(df)
    df['atr'] = calculate_atr(df)
    df.dropna(inplace=True)
    return df
