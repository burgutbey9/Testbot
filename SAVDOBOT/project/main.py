# main.py - Asosiy bot kodi

import asyncio
import websockets
import json
import logging
import os
import time
from datetime import datetime

# Loyiha modullarini import qilish
# .env faylidan API kalitlarini olish uchun api_manager modulidan foydalanamiz
from project.api_manager import (
    get_binance_api_key, get_huggingface_api_key,
    get_gemini_api_key, get_openai_api_key,
    get_twitter_api_key, get_reddit_api_key, get_bybit_api_key
)
from project.orderflow import get_binance_depth_snapshot, detect_large_orders, calculate_tick_speed, apply_imbalance_filter
from project.ai_sentiment import analyze_sentiment_hf, analyze_sentiment_gemini, analyze_sentiment_local
from project.trainer import optimize_strategy
from project.backtest import run_backtest_csv, run_backtest_tick_level

# Log konfiguratsiyasi
# Barcha loglar project/logs papkasiga yoziladi
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOGS_DIR, exist_ok=True) # logs papkasini yaratish

# Asosiy bot loglari
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

# API rotatsiyasi loglari uchun alohida logger
api_rotation_logger = logging.getLogger('api_rotation')
api_rotation_handler = logging.FileHandler(os.path.join(LOGS_DIR, "api_rotation.log"), encoding='utf-8')
api_rotation_handler.setLevel(logging.INFO)
api_rotation_logger.addHandler(api_rotation_handler)
api_rotation_logger.propagate = False

# Global o'zgaruvchilar
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
TRADE_SIZE = 0.001 # Misol uchun savdo hajmi
BINANCE_WEBSOCKET_URI = f"wss://stream.binance.com:9443/ws/{SYMBOL.lower()}@depth"

# Mock data for technical indicators (in a real scenario, this would come from OHLCV data)
# This is a simplified example; actual indicator calculation requires a series of data.
current_price = 30000.0
rsi_value = 50.0
macd_value = 0.0
bb_upper = 30500.0
bb_lower = 29500.0
atr_value = 500.0

async def fetch_ohlcv_data():
    """
    Binance API orqali OHLCV ma'lumotlarini olish logikasi.
    Bu funksiya real vaqtda OHLCV ma'lumotlarini olish uchun API chaqiruvlarini amalga oshiradi.
    Hozircha faqat mock ma'lumot qaytaradi.
    """
    logging.info(f"{SYMBOL} uchun OHLCV ma'lumotlari yuklanmoqda...")
    # Real implementatsiya uchun 'python-binance' yoki 'ccxt' kabi kutubxonalar kerak bo'ladi.
    # api_key, secret_key = get_binance_api_key()
    # if api_key and secret_key:
    #     client = BinanceClient(api_key=api_key, api_secret=secret_key)
    #     klines = client.get_klines(symbol=SYMBOL, interval=INTERVAL, limit=100)
    #     return klines
    # else:
    #     error_logger.error("Binance API kalitlari topilmadi. OHLCV ma'lumotlarini olish imkonsiz.")
    #     return []
    await asyncio.sleep(1) # Simulate network delay
    return [{"timestamp": datetime.now().timestamp() * 1000, "open": 30000, "high": 30100, "low": 29900, "close": 30050, "volume": 100}]

async def calculate_technical_indicators(ohlcv_data):
    """
    OHLCV ma'lumotlari asosida texnik indikatorlarni hisoblaydi.
    Bu yerda 'ta' (Technical Analysis) kutubxonasidan foydalanish mumkin.
    """
    if not ohlcv_data:
        return {"rsi": None, "macd": None, "bb": None, "atr": None}

    # Bu yerda real hisoblashlar bo'ladi. Hozircha mock qiymatlar.
    # Misol uchun:
    # df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    # df['close'] = pd.to_numeric(df['close'])
    # rsi = ta.momentum.RSIIndicator(df['close']).rsi()[-1]
    # macd = ta.trend.MACD(df['close']).macd()[-1]
    # bb_upper = ta.volatility.BollingerBands(df['close']).bollinger_hband()[-1]
    # bb_lower = ta.volatility.BollingerBands(df['close']).bollinger_lband()[-1]
    # atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()[-1]

    return {
        "rsi": rsi_value,
        "macd": macd_value,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "atr": atr_value
    }

async def execute_trade(trade_type, size, price=None, stop_loss=None, take_profit=None):
    """
    Savdo operatsiyasini amalga oshiradi va riskni boshqaradi.
    Args:
        trade_type (str): "BUY" yoki "SELL".
        size (float): Savdo hajmi.
        price (float, optional): Buyurtma narxi (limit order uchun). None bo'lsa, market order.
        stop_loss (float, optional): Stop-Loss narxi.
        take_profit (float, optional): Take-Profit narxi.
    """
    logging.info(f"Savdo amalga oshirilmoqda: {trade_type} {size} {SYMBOL} @ {price if price else 'MARKET'}")
    # Bu yerda real savdo API chaqiruvlari bo'ladi.
    # Misol uchun:
    # api_key, secret_key = get_binance_api_key()
    # if api_key and secret_key:
    #     client = BinanceClient(api_key=api_key, api_secret=secret_key)
    #     if trade_type == "BUY":
    #         order = client.order_market_buy(symbol=SYMBOL, quantity=size)
    #     elif trade_type == "SELL":
    #         order = client.order_market_sell(symbol=SYMBOL, quantity=size)
    #     logging.info(f"Savdo muvaffaqiyatli: {order}")
    #     # trades.log ga yozish
    #     with open(os.path.join(LOGS_DIR, "trades.log"), "a", encoding='utf-8') as f:
    #         f.write(f"{datetime.now().isoformat()}, {trade_type}, {size}, {order['fills'][0]['price'] if order.get('fills') else 'N/A'}, {order['status']}\n")
    # else:
    #     error_logger.error("Binance API kalitlari topilmadi. Savdo amalga oshirilmadi.")
    #
    # Risk boshqaruvi (SL/TP) logikasi shu yerda bo'ladi.
    #
    await asyncio.sleep(0.5) # Simulate trade execution delay
    logging.info(f"Savdo {trade_type} {size} {SYMBOL} simulyatsiya qilindi.")
    with open(os.path.join(LOGS_DIR, "trades.log"), "a", encoding='utf-8') as f:
        f.write(f"{datetime.now().isoformat()}, {trade_type}, {size}, {price if price else 'MARKET'}, SIMULATED\n")


async def connect_binance_websocket():
    """
    Binance WebSocket orqali real vaqt ma'lumotlarini oladi.
    """
    while True:
        try:
            async with websockets.connect(BINANCE_WEBSOCKET_URI) as websocket:
                logging.info(f"Binance WebSocketga ulanildi: {BINANCE_WEBSOCKET_URI}")
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)

                    # Order Book ma'lumotlarini qayta ishlash
                    # Order Flow, Imbalance, Tick Speed hisoblash
                    depth_snapshot = get_binance_depth_snapshot(SYMBOL) # Bu yerda real vaqtda kelgan ma'lumotlardan foydalanish afzal
                    if depth_snapshot:
                        large_orders = detect_large_orders(depth_snapshot)
                        imbalance = apply_imbalance_filter(depth_snapshot)
                        logging.debug(f"Order Book yangilanishi: {data}")
                        logging.info(f"Katta buyurtmalar: {large_orders}")
                        logging.info(f"Imbalance: {imbalance}")
                    else:
                        logging.warning("Buyurtma kitobi snapshot olib bo'lmadi.")

                    tick_speed = calculate_tick_speed(data) # Har bir yangi xabar tick hisoblanadi
                    logging.info(f"Tick Tezligi: {tick_speed}")

                    # AI Sentiment tahlili
                    # Real yangilik manbalaridan ma'lumot olish kerak
                    news_text = "Bitcoin narxi biroz ko'tarildi, bozor ishonchli ko'rinadi." # Misol uchun yangilik
                    sentiment_hf = analyze_sentiment_hf(news_text)
                    sentiment_gemini = analyze_sentiment_gemini(news_text)
                    sentiment_local = analyze_sentiment_local(news_text)
                    logging.info(f"AI Sentiment (HF): {sentiment_hf}, (Gemini): {sentiment_gemini}, (Local): {sentiment_local}")

                    # Texnik indikatorlar
                    ohlcv_data = await fetch_ohlcv_data() # Bu yerda real vaqtda ohlcv ma'lumotlari bo'lishi kerak
                    indicators = await calculate_technical_indicators(ohlcv_data)
                    logging.info(f"Indikatorlar: {indicators}")

                    # Signal generator: indikator + sentiment + order flow asosida savdo signali yaratish
                    # Bu yerda savdo logikasi joylashadi
                    if (sentiment_hf == "positive" or sentiment_gemini == "positive") and \
                       indicators["rsi"] and indicators["rsi"] < 30 and \
                       large_orders and imbalance > 0.1: # Misol shartlar
                        logging.info("LONG savdo signali! Savdo amalga oshirilmoqda...")
                        await execute_trade("BUY", TRADE_SIZE, price=current_price * 1.001, stop_loss=current_price * 0.995)
                    elif (sentiment_hf == "negative" or sentiment_gemini == "negative") and \
                         indicators["rsi"] and indicators["rsi"] > 70 and \
                         large_orders and imbalance < -0.1: # Misol shartlar
                        logging.info("SHORT savdo signali! Savdo amalga oshirilmoqda...")
                        await execute_trade("SELL", TRADE_SIZE, price=current_price * 0.999, take_profit=current_price * 0.995)
                    else:
                        logging.info("Savdo signali mavjud emas. Kutish...")

            # Ulanish uzilgan bo'lsa, qayta ulanishga urinish
        except websockets.exceptions.ConnectionClosedOK:
            logging.warning("WebSocket ulanishi yopildi. 5 soniyadan keyin qayta ulanishga urinish...")
            await asyncio.sleep(5)
        except websockets.exceptions.ConnectionClosedError as e:
            error_logger.error(f"WebSocket ulanish xatosi: {e}. 10 soniyadan keyin qayta ulanishga urinish.", exc_info=True)
            await asyncio.sleep(10)
        except Exception as e:
            error_logger.error(f"WebSocketda kutilmagan xato: {e}. 15 soniyadan keyin qayta ulanishga urinish.", exc_info=True)
            await asyncio.sleep(15)

async def main():
    """
    Asosiy ishga tushirish funksiyasi.
    Botni ishga tushiradi va backtest/optimallashtirish funksiyalarini boshqaradi.
    """
    logging.info("Bot ishga tushirildi! WebSocket ulanishini kutmoqda...")

    # Backtestni ishga tushirish misoli (ishga tushirishdan oldin # belgisini olib tashlang)
    # logging.info("Backtest ishga tushirilmoqda...")
    # run_backtest_csv(os.path.join(os.path.dirname(__file__), "sample_historical_data.csv")) # Mavjud CSV fayl yo'lini kiriting
    # run_backtest_tick_level(os.path.join(os.path.dirname(__file__), "sample_tick_data.csv")) # Mavjud CSV fayl yo'lini kiriting

    # Strategiyani optimallashtirish misoli (ishga tushirishdan oldin # belgisini olib tashlang)
    # logging.info("Strategiya optimallashtirilmoqda...")
    # optimize_strategy(os.path.join(LOGS_DIR, "results.csv"))

    # Asosiy botni ishga tushirish
    await connect_binance_websocket()

if __name__ == "__main__":
    # Loyihani ishga tushirish uchun asosiy nuqta
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Bot foydalanuvchi tomonidan to'xtatildi.")
    except Exception as e:
        error_logger.critical(f"Botning ishga tushirilishida halokatli xato: {e}", exc_info=True)

