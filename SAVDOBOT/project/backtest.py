# backtest.py - Backtest logika

import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime

# Log konfiguratsiyasi
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOGS_DIR, exist_ok=True) # logs papkasini yaratish

# Asosiy backtest loglari (trades.log ga yoziladi)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(os.path.join(LOGS_DIR, "trades.log"), encoding='utf-8'),
                        logging.StreamHandler()
                    ])

# Xatolar uchun alohida logger
error_logger = logging.getLogger('errors')
error_handler = logging.FileHandler(os.path.join(LOGS_DIR, "errors.log"), encoding='utf-8')
error_handler.setLevel(logging.ERROR)
error_logger.addHandler(error_handler)
error_logger.propagate = False # Asosiy loggerga ikki marta yozmaslik uchun

# Natijalar CSV fayli
RESULTS_CSV_PATH = os.path.join(LOGS_DIR, "results.csv")

def run_backtest_csv(historical_csv_path):
    """
    Tarixiy CSV ma'lumotlari bilan backtest o'tkazadi.
    Args:
        historical_csv_path (str): Tarixiy OHLCV ma'lumotlari joylashgan CSV fayl yo'li.
    """
    logging.info(f"Tarixiy CSV ({historical_csv_path}) bilan backtest boshlanmoqda...")
    try:
        df = pd.read_csv(historical_csv_path)
        # Ma'lumotlarni tozalash va tayyorlash
        df.columns = [col.lower() for col in df.columns] # Ustun nomlarini kichik harfga o'tkazish
        if 'timestamp' not in df.columns:
            error_logger.error("CSV faylida 'timestamp' ustuni topilmadi. Backtest to'xtatildi.")
            return
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms') # Agar timestamp millisekundlarda bo'lsa
        df = df.set_index('timestamp').sort_index()

        if 'close' not in df.columns:
            error_logger.error("CSV faylida 'close' ustuni topilmadi. Backtest to'xtatildi.")
            return

        # Misol uchun: Oddiy SMA kesishmasi strategiyasi
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()

        df['signal'] = 0 # 0: ushlab turish, 1: sotib olish, -1: sotish
        # Buy signal: 20-SMA 50-SMAni kesib o'tganda va oldingi shamda kesishmagan bo'lsa
        df.loc[(df['sma_20'] > df['sma_50']) & (df['sma_20'].shift(1) <= df['sma_50'].shift(1)), 'signal'] = 1
        # Sell signal: 20-SMA 50-SMAni pastga kesib o'tganda va oldingi shamda kesishmagan bo'lsa
        df.loc[(df['sma_20'] < df['sma_50']) & (df['sma_20'].shift(1) >= df['sma_50'].shift(1)), 'signal'] = -1

        # Savdo simulyatsiyasi va PnL hisoblash
        initial_capital = 10000.0
        capital = initial_capital
        position = 0 # 0: pozitsiya yo'q, 1: long, -1: short
        entry_price = 0.0
        trades_list = []

        for i, row in df.iterrows():
            if row['signal'] == 1 and position == 0: # Sotib olish signali va pozitsiya yo'q
                entry_price = row['close']
                position = 1
                logging.info(f"BUY: {i} narx: {entry_price:.2f}")
                trades_list.append({'time': i, 'type': 'BUY', 'price': entry_price, 'pnl_percent': 0, 'capital': capital})
            elif row['signal'] == -1 and position == 1: # Sotish signali va long pozitsiya bor
                exit_price = row['close']
                pnl = (exit_price - entry_price) / entry_price * 100 # Foizdagi PnL
                capital *= (1 + pnl / 100)
                logging.info(f"SELL: {i} narx: {exit_price:.2f}, PnL: {pnl:.2f}%, Kapital: {capital:.2f}")
                trades_list.append({'time': i, 'type': 'SELL', 'price': exit_price, 'pnl_percent': pnl, 'capital': capital})
                position = 0
            elif row['signal'] == -1 and position == -1: # Short pozitsiya yopish (agar short strategiya bo'lsa)
                # Bu yerda short pozitsiyani yopish logikasi bo'ladi
                pass
            elif row['signal'] == 1 and position == -1: # Short ochish (agar short strategiya bo'lsa)
                # Bu yerda short pozitsiyani ochish logikasi bo'ladi
                pass

        final_capital = capital
        total_pnl_percent = (final_capital - initial_capital) / initial_capital * 100

        logging.info(f"CSV Backtest yakunlandi.")
        logging.info(f"Boshlang'ich kapital: {initial_capital:.2f}")
        logging.info(f"Yakuniy kapital: {final_capital:.2f}")
        logging.info(f"Jami foyda/zarar (foizda): {total_pnl_percent:.2f}%")

        # Natijalarni 'results.csv' ga yozish
        trades_df = pd.DataFrame(trades_list)
        if not trades_df.empty:
            # Agar fayl mavjud bo'lmasa, sarlavhani yozamiz
            header = not os.path.exists(RESULTS_CSV_PATH)
            trades_df.to_csv(RESULTS_CSV_PATH, mode='a', header=header, index=False, encoding='utf-8')
            logging.info(f"Backtest savdo natijalari '{RESULTS_CSV_PATH}' fayliga yozildi.")
        else:
            logging.warning("Backtest natijasida savdolar amalga oshirilmadi. 'results.csv' yangilanmadi.")

    except FileNotFoundError:
        error_logger.error(f"Xato: {historical_csv_path} fayli topilmadi. Iltimos, to'g'ri yo'lni kiriting.")
    except Exception as e:
        error_logger.error(f"CSV backtestda kutilmagan xato yuz berdi: {e}", exc_info=True)

def run_backtest_tick_level(tick_data_csv_path):
    """
    Tick-level ma'lumotlari bilan backtest o'tkazadi.
    Args:
        tick_data_csv_path (str): Tick-level ma'lumotlari joylashgan CSV fayl yo'li.
    """
    logging.info(f"Tick-level CSV ({tick_data_csv_path}) bilan backtest boshlanmoqda...")
    try:
        df = pd.read_csv(tick_data_csv_path)
        # Ma'lumotlarni tozalash va tayyorlash
        df.columns = [col.lower() for col in df.columns]
        if 'timestamp' not in df.columns or 'price' not in df.columns or 'quantity' not in df.columns:
            error_logger.error("Tick-level CSV faylida 'timestamp', 'price' yoki 'quantity' ustunlari topilmadi. Backtest to'xtatildi.")
            return
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms') # Agar timestamp millisekundlarda bo'lsa
        df = df.set_index('timestamp').sort_index()

        # Bu yerda tick-level backtest logikasi bo'ladi:
        # - Har bir tickni qayta ishlash
        # - Order Flow indikatorlarini qo'llash (masalan, Volume Weighted Average Price - VWAP, Cumulative Delta)
        # - Yuqori chastotali savdolarni simulyatsiya qilish
        # - PnL va slippage hisoblash

        logging.info("Tick-level backtest logikasi shu yerda amalga oshiriladi. Bu juda murakkab bo'lishi mumkin.")
        logging.info("Misol uchun, har bir tickda savdo qarorlari qabul qilinadi.")

        initial_capital = 10000.0
        capital = initial_capital
        position = 0
        entry_price = 0.0
        tick_trades_list = []

        # Oddiy misol: Har 100-tickda savdo qilish
        for i, row in df.iterrows():
            # Bu yerda murakkab Order Flow tahlili bo'ladi
            # Misol uchun, har 100 tickda bir savdo qilish
            if df.index.get_loc(i) % 100 == 0:
                if position == 0: # Sotib olish
                    entry_price = row['price']
                    position = 1
                    logging.info(f"TICK BUY: {i} narx: {entry_price:.4f}")
                    tick_trades_list.append({'time': i, 'type': 'BUY', 'price': entry_price, 'pnl_percent': 0, 'capital': capital})
                elif position == 1: # Sotish
                    exit_price = row['price']
                    pnl = (exit_price - entry_price) / entry_price * 100
                    capital *= (1 + pnl / 100)
                    logging.info(f"TICK SELL: {i} narx: {exit_price:.4f}, PnL: {pnl:.2f}%, Kapital: {capital:.2f}")
                    tick_trades_list.append({'time': i, 'type': 'SELL', 'price': exit_price, 'pnl_percent': pnl, 'capital': capital})
                    position = 0

        final_capital = capital
        total_pnl_percent = (final_capital - initial_capital) / initial_capital * 100

        logging.info(f"Tick-level Backtest yakunlandi.")
        logging.info(f"Boshlang'ich kapital: {initial_capital:.2f}")
        logging.info(f"Yakuniy kapital: {final_capital:.2f}")
        logging.info(f"Jami foyda/zarar (foizda): {total_pnl_percent:.2f}%")

        # Natijalarni 'results.csv' ga yozish
        tick_trades_df = pd.DataFrame(tick_trades_list)
        if not tick_trades_df.empty:
            header = not os.path.exists(RESULTS_CSV_PATH)
            tick_trades_df.to_csv(RESULTS_CSV_PATH, mode='a', header=header, index=False, encoding='utf-8')
            logging.info(f"Tick-level backtest savdo natijalari '{RESULTS_CSV_PATH}' fayliga yozildi.")
        else:
            logging.warning("Tick-level backtest natijasida savdolar amalga oshirilmadi. 'results.csv' yangilanmadi.")

    except FileNotFoundError:
        error_logger.error(f"Xato: {tick_data_csv_path} fayli topilmadi. Iltimos, to'g'ri yo'lni kiriting.")
    except Exception as e:
        error_logger.error(f"Tick-level backtestda kutilmagan xato yuz berdi: {e}", exc_info=True)

if __name__ == "__main__":
    # Misol uchun foydalanish
    # Bu yerda siz o'zingizning test CSV fayllaringizni yaratishingiz kerak.
    # Misol: sample_historical_data.csv (timestamp, open, high, low, close, volume)
    # Misol: sample_tick_data.csv (timestamp, price, quantity, is_buy_maker)
    #
    # # Test uchun namunaviy CSV fayllarni yaratish (faqat test uchun)
    # sample_ohlcv_data = {
    #     'timestamp': pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 00:01:00', '2023-01-01 00:02:00', '2023-01-01 00:03:00', '2023-01-01 00:04:00', '2023-01-01 00:05:00', '2023-01-01 00:06:00', '2023-01-01 00:07:00', '2023-01-01 00:08:00', '2023-01-01 00:09:00', '2023-01-01 00:10:00', '2023-01-01 00:11:00', '2023-01-01 00:12:00', '2023-01-01 00:13:00', '2023-01-01 00:14:00', '2023-01-01 00:15:00', '2023-01-01 00:16:00', '2023-01-01 00:17:00', '2023-01-01 00:18:00', '2023-01-01 00:19:00', '2023-01-01 00:20:00', '2023-01-01 00:21:00', '2023-01-01 00:22:00', '2023-01-01 00:23:00', '2023-01-01 00:24:00', '2023-01-01 00:25:00', '2023-01-01 00:26:00', '2023-01-01 00:27:00', '2023-01-01 00:28:00', '2023-01-01 00:29:00']),
    #     'open': np.linspace(100, 110, 30),
    #     'high': np.linspace(101, 111, 30),
    #     'low': np.linspace(99, 109, 30),
    #     'close': np.linspace(100.5, 109.5, 30),
    #     'volume': np.random.randint(10, 100, 30)
    # }
    # sample_ohlcv_df = pd.DataFrame(sample_ohlcv_data)
    # sample_ohlcv_df.to_csv("sample_historical_data.csv", index=False)
    #
    # sample_tick_data = {
    #     'timestamp': pd.to_datetime(np.arange(1000), unit='ms', origin='2023-01-01'),
    #     'price': np.random.normal(loc=105, scale=0.5, size=1000),
    #     'quantity': np.random.randint(1, 10, size=1000),
    #     'is_buy_maker': np.random.choice([True, False], size=1000)
    # }
    # sample_tick_df = pd.DataFrame(sample_tick_data)
    # sample_tick_df.to_csv("sample_tick_data.csv", index=False)
    #
    # run_backtest_csv("sample_historical_data.csv")
    # run_backtest_tick_level("sample_tick_data.csv")
    pass
