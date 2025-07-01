# orderflow.py - Order Flow strategiya kodlari

import requests
import json
import time
import logging
import os

# Log konfiguratsiyasi
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOGS_DIR, exist_ok=True) # logs papkasini yaratish

# Asosiy loglar (api_rotation.log ga yoziladi)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(os.path.join(LOGS_DIR, "api_rotation.log"), encoding='utf-8'),
                        logging.StreamHandler()
                    ])

# Xatolar uchun alohida logger
error_logger = logging.getLogger('errors')
error_handler = logging.FileHandler(os.path.join(LOGS_DIR, "errors.log"), encoding='utf-8')
error_handler.setLevel(logging.ERROR)
error_logger.addHandler(error_handler)
error_logger.propagate = False # Asosiy loggerga ikki marta yozmaslik uchun

BINANCE_API_BASE_URL = "https://api.binance.com/api/v3"

def get_binance_depth_snapshot(symbol, limit=100):
    """
    Binance spot birjasidan buyurtma kitobi (depth) snapshotini oladi.
    Args:
        symbol (str): Savdo juftligi (masalan, "BTCUSDT").
        limit (int): Qaytariladigan buyurtmalar soni (5, 10, 20, 50, 100, 500, 1000, 5000).
    Returns:
        dict: Buyurtma kitobi ma'lumotlari (bids, asks) yoki None agar xato yuz bersa.
    """
    endpoint = f"{BINANCE_API_BASE_URL}/depth"
    params = {"symbol": symbol, "limit": limit}
    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status() # HTTP xatolarini tekshirish
        data = response.json()
        logging.info(f"{symbol} uchun buyurtma kitobi snapshot olindi.")
        return data
    except requests.exceptions.RequestException as e:
        error_logger.error(f"Binance depth snapshot olishda xato: {e}", exc_info=True)
        return None

def detect_large_orders(depth_data, threshold_percentage=0.005): # Chegarani 0.5% ga o'zgartirdim
    """
    Buyurtma kitobidagi katta buyurtmalarni aniqlaydi.
    Katta buyurtma deb, umumiy hajmdan belgilangan foizdan yuqori bo'lgan buyurtmalar hisoblanadi.
    Args:
        depth_data (dict): Buyurtma kitobi ma'lumotlari (bids, asks).
        threshold_percentage (float): Katta buyurtma deb hisoblash uchun umumiy hajmdan foiz (masalan, 0.005 = 0.5%).
    Returns:
        dict: Katta sotib olish (large_bids) va sotish (large_asks) buyurtmalari.
    """
    if not depth_data or "bids" not in depth_data or "asks" not in depth_data:
        logging.warning("detect_large_orders: Buyurtma kitobi ma'lumotlari mavjud emas.")
        return {"large_bids": [], "large_asks": []}

    all_bids_volume = sum([float(b[1]) for b in depth_data["bids"]])
    all_asks_volume = sum([float(a[1]) for a in depth_data["asks"]])

    large_bids = []
    for bid in depth_data["bids"]:
        price, volume = float(bid[0]), float(bid[1])
        if all_bids_volume > 0 and (volume / all_bids_volume >= threshold_percentage):
            large_bids.append({"price": price, "volume": volume})

    large_asks = []
    for ask in depth_data["asks"]:
        price, volume = float(ask[0]), float(ask[1])
        if all_asks_volume > 0 and (volume / all_asks_volume >= threshold_percentage):
            large_asks.append({"price": price, "volume": volume})

    logging.info(f"Katta buyurtmalar aniqlandi. Bids: {len(large_bids)}, Asks: {len(large_asks)}")
    return {"large_bids": large_bids, "large_asks": large_asks}

_last_tick_time = time.time()
_tick_count = 0
_tick_interval_seconds = 1 # Tick tezligini hisoblash intervali (sekundda)

def calculate_tick_speed(new_tick_data):
    """
    Tick tezligini hisoblaydi (ma'lum vaqt oralig'idagi ticklar soni).
    Args:
        new_tick_data (dict): Yangi kelgan tick ma'lumotlari (misol uchun WebSocket xabari).
    Returns:
        float: Tick tezligi (sekundiga ticklar).
    """
    global _last_tick_time, _tick_count
    current_time = time.time()
    _tick_count += 1

    time_diff = current_time - _last_tick_time
    if time_diff >= _tick_interval_seconds: # Belgilangan interval o'tganda hisoblash
        tick_speed = _tick_count / time_diff
        logging.debug(f"Tick tezligi hisoblandi: {tick_speed:.2f} ticks/sec")
        _last_tick_time = current_time # Hisoblagichni reset qilish
        _tick_count = 0
        return tick_speed
    return 0.0 # Agar interval tugamagan bo'lsa, 0 qaytarish

def apply_imbalance_filter(depth_data, imbalance_threshold=0.6):
    """
    Buyurtma kitobidagi imbalance (nomutanosiblik)ni hisoblaydi va filtrlaydi.
    Imbalance = (Bids Volume - Asks Volume) / (Bids Volume + Asks Volume)
    Args:
        depth_data (dict): Buyurtma kitobi ma'lumotlari (bids, asks).
        imbalance_threshold (float): Imbalance filtri chegarasi (0 dan 1 gacha, absolyut qiymat).
    Returns:
        float: Hisoblangan imbalance qiymati.
    """
    if not depth_data or "bids" not in depth_data or "asks" not in depth_data:
        logging.warning("apply_imbalance_filter: Buyurtma kitobi ma'lumotlari mavjud emas.")
        return 0.0

    bids_volume = sum([float(b[1]) for b in depth_data["bids"]])
    asks_volume = sum([float(a[1]) for a in depth_data["asks"]])

    total_volume = bids_volume + asks_volume
    if total_volume == 0:
        return 0.0

    imbalance = (bids_volume - asks_volume) / total_volume
    logging.debug(f"Imbalance hisoblandi: {imbalance:.2f}")

    # Filtrni qo'llash (agar kerak bo'lsa, bu yerda savdo signali berilishi mumkin)
    if abs(imbalance) >= imbalance_threshold:
        logging.info(f"Yuqori imbalance aniqlandi: {imbalance:.2f}")
    return imbalance

if __name__ == "__main__":
    # Misol uchun foydalanish
    symbol = "BTCUSDT"
    print(f"{symbol} uchun buyurtma kitobi snapshot olinmoqda...")
    depth = get_binance_depth_snapshot(symbol)
    if depth:
        print("Snapshot muvaffaqiyatli olindi.")
        large_orders = detect_large_orders(depth)
        print(f"Katta sotib olish buyurtmalari: {large_orders['large_bids']}")
        print(f"Katta sotish buyurtmalari: {large_orders['large_asks']}")
        imbalance = apply_imbalance_filter(depth)
        print(f"Imbalance: {imbalance:.2f}")

    print("\nTick speedni sinash...")
    # Tick speedni sinash uchun
    for i in range(20):
        time.sleep(0.05) # Har 50ms da bir tick
        speed = calculate_tick_speed({"event_type": "trade", "data": i}) # Har qanday yangi voqea tick hisoblanadi
        if speed > 0: # Faqat hisoblangan tezlikni ko'rsatish
            print(f"Tick Speed: {speed:.2f} ticks/sec")
