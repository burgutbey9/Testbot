# project/api_manager.py - API rotation manager

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
                        logging.FileHandler(os.path.join(LOGS_DIR, "api_rotation.log"), encoding='utf-8', mode='a'),
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

# Yangi API kalitlari
ONE_INCH_API_KEYS = [os.getenv(f"ONE_INCH_API_KEY_{i}") for i in range(1, 6) if os.getenv(f"ONE_INCH_API_KEY_{i}")]
NEWS_API_KEYS = [os.getenv(f"NEWS_API_KEY_{i}") for i in range(1, 2) if os.getenv(f"NEWS_API_KEY_{i}")] # Sizda bitta bor
ALCHEMY_API_KEYS = [os.getenv(f"ALCHEMY_API_KEY_{i}") for i in range(1, 2) if os.getenv(f"ALCHEMY_API_KEY_{i}")] # Sizda bitta bor
HUGGING_FACE_API_KEYS = [os.getenv(f"HUGGING_FACE_API_KEY_{i}") for i in range(1, 5) if os.getenv(f"HUGGING_FACE_API_KEY_{i}")] # Sizda 2 ta bor
GEMINI_API_KEYS = [os.getenv(f"GEMINI_API_KEY_{i}") for i in range(1, 6) if os.getenv(f"GEMINI_API_KEY_{i}")] # Sizda 5 ta bor
REDDIT_API_KEYS = [
    {
        "client_id": os.getenv(f"REDDIT_CLIENT_ID_{i}"),
        "client_secret": os.getenv(f"REDDIT_CLIENT_SECRET_{i}"),
        "username": os.getenv(f"REDDIT_USERNAME_{i}"),
        "password": os.getenv(f"REDDIT_PASSWORD_{i}")
    }
    for i in range(1, 2) if os.getenv(f"REDDIT_CLIENT_ID_{i}")
]

# API kalitlari uchun indekslar
_one_inch_key_index = 0
_news_api_key_index = 0
_alchemy_api_key_index = 0
_huggingface_key_index = 0
_gemini_key_index = 0
_reddit_key_index = 0


def _get_next_key(keys_list: list, current_index: int, api_name: str) -> tuple[str | dict | None, int]:
    """Umumiy funksiya keylarni aylantirish uchun."""
    if not keys_list:
        logging.warning(f"{api_name} uchun API kalitlari topilmadi. None qaytarilmoqda.")
        return None, current_index

    key = keys_list[current_index]
    next_index = (current_index + 1) % len(keys_list)
    
    # Kalitni qisman ko'rsatish (xavfsizlik uchun)
    if isinstance(key, tuple): # Kelajakda tuple formatidagi kalitlar uchun
        display_key = f"{key[0][:5]}...{key[0][-5:]}" if key[0] else "N/A"
    elif isinstance(key, dict): # Reddit kabi lug'at uchun
        display_key = f"{key.get('client_id', '')[:5]}..." if key.get('client_id') else "N/A"
    else: # Oddiy string kalitlar uchun (1inch, NewsAPI, Alchemy, Hugging Face, Gemini)
        display_key = f"{key[:5]}...{key[-5:]}" if key else "N/A"

    logging.info(f"{api_name} uchun API kaliti ishlatildi: {display_key}")
    return key, next_index

def get_one_inch_api_key() -> str | None:
    global _one_inch_key_index
    key, _one_inch_key_index = _get_next_key(ONE_INCH_API_KEYS, _one_inch_key_index, "1inch")
    return key

def get_news_api_key() -> str | None:
    global _news_api_key_index
    key, _news_api_key_index = _get_next_key(NEWS_API_KEYS, _news_api_key_index, "NewsAPI")
    return key

def get_alchemy_api_key() -> str | None:
    global _alchemy_api_key_index
    key, _alchemy_api_key_index = _get_next_key(ALCHEMY_API_KEYS, _alchemy_api_key_index, "Alchemy")
    return key

def get_huggingface_api_key() -> str | None:
    global _huggingface_key_index
    key, _huggingface_key_index = _get_next_key(HUGGING_FACE_API_KEYS, _huggingface_key_index, "Hugging Face")
    return key

def get_gemini_api_key() -> str | None:
    global _gemini_key_index
    key, _gemini_key_index = _get_next_key(GEMINI_API_KEYS, _gemini_key_index, "Gemini")
    return key

def get_reddit_api_key() -> dict | None:
    global _reddit_key_index
    key, _reddit_key_index = _get_next_key(REDDIT_API_KEYS, _reddit_key_index, "Reddit")
    return key

# Olib tashlangan API'lar uchun funksiyalar (agar ularga boshqa joyda murojaat qilingan bo'lsa, xato bermasligi uchun)
def get_binance_api_key() -> tuple[None, None]:
    logging.warning("Binance API kalitlari loyihadan olib tashlandi.")
    return None, None

def get_openai_api_key() -> None:
    logging.warning("OpenAI API kalitlari loyihadan olib tashlandi.")
    return None

def get_twitter_api_key() -> tuple[None, None]:
    logging.warning("Twitter API kalitlari loyihadan olib tashlandi.")
    return None, None

def get_bybit_api_key() -> tuple[None, None]:
    logging.warning("Bybit API kalitlari loyihadan olib tashlandi.")
    return None, None


if __name__ == "__main__":
    # .env.example faylini yaratish (faqat test uchun)
    env_example_path = os.path.join(os.path.dirname(__file__), ".env.example")
    if not os.path.exists(env_example_path):
        with open(env_example_path, "w", encoding="utf-8") as f:
            f.write("""
# Barcha API key namunalari shu yerga yoziladi.

# 1inch Developer Portal API kalitlari (kamida 5 ta tavsiya etiladi)
ONE_INCH_API_KEY_1="sizning_1inch_api_key_1"
ONE_INCH_API_KEY_2="sizning_1inch_api_key_2"
ONE_INCH_API_KEY_3="sizning_1inch_api_key_3"
ONE_INCH_API_KEY_4="sizning_1inch_api_key_4"
ONE_INCH_API_KEY_5="sizning_1inch_api_key_5"

# NewsAPI.org kalitlari (kamida 1 ta)
NEWS_API_KEY_1="sizning_newsapi_key_1"

# Alchemy API kalitlari (kamida 1 ta)
ALCHEMY_API_KEY_1="sizning_alchemy_api_key_1"

# Hugging Face API kalitlari (kamida 4 ta tavsiya etiladi)
HUGGING_FACE_API_KEY_1="sizning_huggingface_api_key_1"
HUGGING_FACE_API_KEY_2="sizning_huggingface_api_key_2"
HUGGING_FACE_API_KEY_3="sizning_huggingface_api_key_3"
HUGGING_FACE_API_KEY_4="sizning_huggingface_api_key_4"

# Gemini API kalitlari (kamida 3-5 ta)
GEMINI_API_KEY_1="sizning_gemini_api_key_1"
GEMINI_API_KEY_2="sizning_gemini_api_key_2"
GEMINI_API_KEY_3="sizning_gemini_api_key_3"
GEMINI_API_KEY_4="sizning_gemini_api_key_4"
GEMINI_API_KEY_5="sizning_gemini_api_key_5"

# Reddit API kaliti (kamida 1 ta)
REDDIT_CLIENT_ID_1="sizning_reddit_client_id_1"
REDDIT_CLIENT_SECRET_1="sizning_reddit_client_secret_1"
REDDIT_USERNAME_1="sizning_reddit_username_1"
REDDIT_PASSWORD_1="sizning_reddit_password_1"
""")
        print(f"'.env.example' fayli yaratildi. Iltimos, uni '.env' nomiga o'zgartirib, API kalitlarini to'ldiring.")

    # Test uchun .env faylini yuklash
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

    print("API kalitlarini aylantirish misoli:")
    print("1inch API kaliti:", get_one_inch_api_key())
    print("NewsAPI API kaliti:", get_news_api_key())
    print("Alchemy API kaliti:", get_alchemy_api_key())
    print("Hugging Face API kaliti:", get_huggingface_api_key())
    print("Gemini API kaliti:", get_gemini_api_key())
    print("Reddit API kaliti:", get_reddit_api_key())

    # Olib tashlangan API'lar
    print("Binance API kaliti:", get_binance_api_key())
    print("OpenAI API kaliti:", get_openai_api_key())
    print("Twitter API kaliti:", get_twitter_api_key())
    print("Bybit API kaliti:", get_bybit_api_key())

    # Keylarni aylantirishni ko'rsatish
    print("\nKeylarni aylantirish misoli (Gemini):")
    for _ in range(7): # 7 marta chaqirish
        print("Gemini key:", get_gemini_api_key())
