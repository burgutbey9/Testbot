# project/orderflow.py - Order Flow tahlili (DEX ma'lumotlariga moslashtirilgan)

import logging
import os
import time

# Log konfiguratsiyasi
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOGS_DIR, exist_ok=True) # logs papkasini yaratish

# Asosiy loglar (api_rotation.log ga yoziladi)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(os.path.join(LOGS_DIR, "api_rotation.log"), encoding='utf-8', mode='a'), # Order Flow loglari uchun
                        logging.StreamHandler()
                    ])

# Xatolar uchun alohida logger
error_logger = logging.getLogger('errors')
error_handler = logging.FileHandler(os.path.join(LOGS_DIR, "errors.log"), encoding='utf-8')
error_handler.setLevel(logging.ERROR)
error_logger.addHandler(error_handler)
error_logger.propagate = False # Asosiy loggerga ikki marta yozmaslik uchun

# get_binance_depth_snapshot funksiyasi olib tashlandi, chunki Binance API ishlatilmaydi.
# DEX uchun Order Flow tahlili on-chain ma'lumotlarga asoslanadi.

def detect_large_orders(depth_data: dict, threshold_percentage: float = 0.005) -> dict:
    """
    Buyurtma kitobidagi katta buyurtmalarni aniqlaydi.
    DEX kontekstida bu 'depth_data' mock yoki The Graph kabi manbalardan olingan
    likvidlik ma'lumotlari bo'lishi mumkin.
    Args:
        depth_data (dict): Buyurtma kitobi ma'lumotlari (bids, asks).
        threshold_percentage (float): Katta buyurtma deb hisoblash uchun umumiy hajmdan foiz (masalan, 0.005 = 0.5%).
    Returns:
        dict: Katta sotib olish (large_bids) va sotish (large_asks) buyurtmalari.
    """
    if not depth_data or "bids" not in depth_data or "asks" not in depth_data:
        logging.warning("detect_large_orders: Buyurtma kitobi ma'lumotlari mavjud emas yoki noto'g'ri formatda.")
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

def calculate_tick_speed(new_tick_data: dict) -> float:
    """
    Tick tezligini hisoblaydi (ma'lum vaqt oralig'idagi ticklar soni).
    DEX kontekstida bu tranzaksiya tezligi yoki blok tezligi bo'lishi mumkin.
    Args:
        new_tick_data (dict): Yangi kelgan tick ma'lumotlari (misol uchun WebSocket xabari yoki on-chain tranzaksiya).
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

def apply_imbalance_filter(depth_data: dict, imbalance_threshold: float = 0.05) -> float: # Chegarani 0.05 ga o'zgartirdim
    """
    Buyurtma kitobidagi imbalance (nomutanosiblik)ni hisoblaydi va filtrlaydi.
    Imbalance = (Bids Volume - Asks Volume) / (Bids Volume + Asks Volume)
    DEX kontekstida bu on-chain likvidlik yoki savdo hajmi nomutanosibligi bo'lishi mumkin.
    Args:
        depth_data (dict): Buyurtma kitobi ma'lumotlari (bids, asks).
        imbalance_threshold (float): Imbalance filtri chegarasi (0 dan 1 gacha, absolyut qiymat).
    Returns:
        float: Hisoblangan imbalance qiymati.
    """
    if not depth_data or "bids" not in depth_data or "asks" not in depth_data:
        logging.warning("apply_imbalance_filter: Buyurtma kitobi ma'lumotlari mavjud emas yoki noto'g'ri formatda.")
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
    # Misol uchun foydalanish (DEX ga moslashtirilgan mock data)
    print("Order Flow tahlili misoli (DEX ma'lumotlari bilan):")
    mock_depth = {
        "bids": [["2990.0", "100"], ["2980.0", "50"]],
        "asks": [["3010.0", "120"], ["3020.0", "60"]]
    }
    
    large_orders = detect_large_orders(mock_depth)
    print(f"Katta sotib olish buyurtmalari: {large_orders['large_bids']}")
    print(f"Katta sotish buyurtmalari: {large_orders['large_asks']}")
    
    imbalance = apply_imbalance_filter(mock_depth)
    print(f"Imbalance: {imbalance:.2f}")

    print("\nTick speedni sinash...")
    # Tick speedni sinash uchun
    for i in range(20):
        time.sleep(0.05) # Har 50ms da bir tick (simulyatsiya)
        speed = calculate_tick_speed({"event_type": "transaction", "data": f"tx_{i}"}) # Har qanday yangi voqea tick hisoblanadi
        if speed > 0: # Faqat hisoblangan tezlikni ko'rsatish
            print(f"Tick Speed: {speed:.2f} ticks/sec")
