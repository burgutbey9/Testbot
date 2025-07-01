# api_manager.py - API rotation manager

import os
from dotenv import load_dotenv
import logging
import time

# .env faylini yuklash
load_dotenv()

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

# API kalitlari ro'yxatlari
# .env faylidan kalitlarni o'qish
# Har bir API uchun kamida bitta kalit bo'lishi kerak.
# Agar .env faylida mos kalit topilmasa, ro'yxat bo'sh bo'ladi.
BINANCE_API_KEYS = [
    (os.getenv(f"BINANCE_API_KEY_{i}"), os.getenv(f"BINANCE_SECRET_KEY_{i}"))
    for i in range(1, 6) if os.getenv(f"BINANCE_API_KEY_{i}") and os.getenv(f"BINANCE_SECRET_KEY_{i}")
]
HUGGING_FACE_API_KEYS = [os.getenv(f"HUGGING_FACE_API_KEY_{i}") for i in range(1, 5) if os.getenv(f"HUGGING_FACE_API_KEY_{i}")]
GEMINI_API_KEYS = [os.getenv(f"GEMINI_API_KEY_{i}") for i in range(1, 6) if os.getenv(f"GEMINI_API_KEY_{i}")]
OPENAI_API_KEYS = [os.getenv(f"OPENAI_API_KEY_{i}") for i in range(1, 4) if os.getenv(f"OPENAI_API_KEY_{i}")]
TWITTER_API_KEYS = [
    (os.getenv(f"TWITTER_API_KEY_{i}"), os.getenv(f"TWITTER_SECRET_KEY_{i}"))
    for i in range(1, 3) if os.getenv(f"TWITTER_API_KEY_{i}") and os.getenv(f"TWITTER_SECRET_KEY_{i}")
]
REDDIT_API_KEYS = [
    {
        "client_id": os.getenv(f"REDDIT_CLIENT_ID_{i}"),
        "client_secret": os.getenv(f"REDDIT_CLIENT_SECRET_{i}"),
        "username": os.getenv(f"REDDIT_USERNAME_{i}"),
        "password": os.getenv(f"REDDIT_PASSWORD_{i}")
    }
    for i in range(1, 2) if os.getenv(f"REDDIT_CLIENT_ID_{i}")
]
BYBIT_API_KEYS = [
    (os.getenv(f"BYBIT_API_KEY_{i}"), os.getenv(f"BYBIT_SECRET_KEY_{i}"))
    for i in range(1, 2) if os.getenv(f"BYBIT_API_KEY_{i}") and os.getenv(f"BYBIT_SECRET_KEY_{i}")
]


# API kalitlari uchun indekslar
_binance_key_index = 0
_huggingface_key_index = 0
_gemini_key_index = 0
_openai_key_index = 0
_twitter_key_index = 0
_reddit_key_index = 0
_bybit_key_index = 0


def _get_next_key(keys_list, current_index, api_name):
    """Umumiy funksiya keylarni aylantirish uchun."""
    if not keys_list:
        logging.warning(f"{api_name} uchun API kalitlari topilmadi. None qaytarilmoqda.")
        return None, current_index

    key = keys_list[current_index]
    next_index = (current_index + 1) % len(keys_list)
    
    # Kalitni qisman ko'rsatish (xavfsizlik uchun)
    if isinstance(key, tuple): # Binance, Twitter, Bybit kabi juftliklar uchun
        display_key = f"{key[0][:5]}...{key[0][-5:]}" if key[0] else "N/A"
    elif isinstance(key, dict): # Reddit kabi lug'at uchun
        display_key = f"{key.get('client_id', '')[:5]}..." if key.get('client_id') else "N/A"
    else: # Oddiy string kalitlar uchun
        display_key = f"{key[:5]}...{key[-5:]}" if key else "N/A"

    logging.info(f"{api_name} uchun API kaliti ishlatildi: {display_key}")
    return key, next_index

def get_binance_api_key():
    global _binance_key_index
    key, _binance_key_index = _get_next_key(BINANCE_API_KEYS, _binance_key_index, "Binance")
    return key

def get_huggingface_api_key():
    global _huggingface_key_index
    key, _huggingface_key_index = _get_next_key(HUGGING_FACE_API_KEYS, _huggingface_key_index, "Hugging Face")
    return key

def get_gemini_api_key():
    global _gemini_key_index
    key, _gemini_key_index = _get_next_key(GEMINI_API_KEYS, _gemini_key_index, "Gemini")
    return key

def get_openai_api_key():
    global _openai_key_index
    key, _openai_key_index = _get_next_key(OPENAI_API_KEYS, _openai_key_index, "OpenAI")
    return key

def get_twitter_api_key():
    global _twitter_key_index
    key, _twitter_key_index = _get_next_key(TWITTER_API_KEYS, _twitter_key_index, "Twitter")
    return key

def get_reddit_api_key():
    global _reddit_key_index
    key, _reddit_key_index = _get_next_key(REDDIT_API_KEYS, _reddit_key_index, "Reddit")
    return key

def get_bybit_api_key():
    global _bybit_key_index
    key, _bybit_key_index = _get_next_key(BYBIT_API_KEYS, _bybit_key_index, "Bybit")
    return key

if __name__ == "__main__":
    # .env.example faylini yaratish (faqat test uchun)
    env_example_path = os.path.join(os.path.dirname(__file__), ".env.example")
    if not os.path.exists(env_example_path):
        with open(env_example_path, "w", encoding="utf-8") as f:
            f.write("""
# Barcha API key namunalari shu yerga yoziladi.

# Binance API kalitlari (kamida 5 ta)
BINANCE_API_KEY_1="test_binance_api_key_1"
BINANCE_SECRET_KEY_1="test_binance_secret_key_1"
BINANCE_API_KEY_2="test_binance_api_key_2"
BINANCE_SECRET_KEY_2="test_binance_secret_key_2"

# Hugging Face API kalitlari (kamida 4 ta)
HUGGING_FACE_API_KEY_1="test_huggingface_api_key_1"
HUGGING_FACE_API_KEY_2="test_huggingface_api_key_2"

# Gemini API kalitlari (kamida 3-5 ta)
GEMINI_API_KEY_1="test_gemini_api_key_1"
GEMINI_API_KEY_2="test_gemini_api_key_2"

# OpenAI API kalitlari (kamida 2-3 ta)
OPENAI_API_KEY_1="test_openai_api_key_1"

# Twitter API kalitlari (kamida 2 ta)
TWITTER_API_KEY_1="test_twitter_api_key_1"
TWITTER_SECRET_KEY_1="test_twitter_secret_key_1"

# Reddit API kaliti (kamida 1 ta)
REDDIT_CLIENT_ID_1="test_reddit_client_id_1"
REDDIT_CLIENT_SECRET_1="test_reddit_client_secret_1"
REDDIT_USERNAME_1="test_reddit_username_1"
REDDIT_PASSWORD_1="test_reddit_password_1"

# Bybit API kaliti (zaxira)
BYBIT_API_KEY_1="test_bybit_api_key_1"
BYBIT_SECRET_KEY_1="test_bybit_secret_key_1"
""")
        print(f"'.env.example' fayli yaratildi. Iltimos, uni '.env' nomiga o'zgartirib, API kalitlarini to'ldiring.")

    # Test uchun .env faylini yuklash
    # load_dotenv(dotenv_path=env_example_path) # Test uchun .env.example dan o'qish

    print("API kalitlarini aylantirish misoli:")
    # Binance kalitlarini sinash
    for _ in range(len(BINANCE_API_KEYS) + 2): # Ro'yxatdan bir necha marta ko'proq chaqirish
        key = get_binance_api_key()
        if key:
            print(f"Binance API kaliti: {key[0][:5]}...")
        else:
            print("Binance API kaliti topilmadi.")

    # Hugging Face kalitlarini sinash
    for _ in range(len(HUGGING_FACE_API_KEYS) + 2):
        key = get_huggingface_api_key()
        if key:
            print(f"Hugging Face API kaliti: {key[:5]}...")
        else:
            print("Hugging Face API kaliti topilmadi.")

    # Gemini kalitlarini sinash
    for _ in range(len(GEMINI_API_KEYS) + 2):
        key = get_gemini_api_key()
        if key:
            print(f"Gemini API kaliti: {key[:5]}...")
        else:
            print("Gemini API kaliti topilmadi.")

    # OpenAI kalitlarini sinash
    for _ in range(len(OPENAI_API_KEYS) + 2):
        key = get_openai_api_key()
        if key:
            print(f"OpenAI API kaliti: {key[:5]}...")
        else:
            print("OpenAI API kaliti topilmadi.")

    # Twitter kalitlarini sinash
    for _ in range(len(TWITTER_API_KEYS) + 2):
        key = get_twitter_api_key()
        if key:
            print(f"Twitter API kaliti: {key[0][:5]}...")
        else:
            print("Twitter API kaliti topilmadi.")

    # Reddit kalitlarini sinash
    for _ in range(len(REDDIT_API_KEYS) + 2):
        key = get_reddit_api_key()
        if key:
            print(f"Reddit API kaliti: {key['client_id'][:5]}...")
        else:
            print("Reddit API kaliti topilmadi.")

    # Bybit kalitlarini sinash
    for _ in range(len(BYBIT_API_KEYS) + 2):
        key = get_bybit_api_key()
        if key:
            print(f"Bybit API kaliti: {key[0][:5]}...")
        else:
            print("Bybit API kaliti topilmadi.")
