# project/ai_sentiment.py - Sentiment analysis qismi

import requests
import json
import logging
import joblib # Modelni yuklash uchun
from textblob import TextBlob # Offline sentiment uchun
import os

# Log konfiguratsiyasi
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOGS_DIR, exist_ok=True) # logs papkasini yaratish

# Asosiy loglar (api_rotation.log ga yoziladi)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(os.path.join(LOGS_DIR, "api_rotation.log"), encoding='utf-8', mode='a'), # API chaqiruvlari logi uchun
                        logging.StreamHandler()
                    ])

# Xatolar uchun alohida logger
error_logger = logging.getLogger('errors')
error_handler = logging.FileHandler(os.path.join(LOGS_DIR, "errors.log"), encoding='utf-8')
error_handler.setLevel(logging.ERROR)
error_logger.addHandler(error_handler)
error_logger.propagate = False # Asosiy loggerga ikki marta yozmaslik uchun

# API kalitlarini olish (api_manager dan)
from project.api_manager import get_huggingface_api_key, get_gemini_api_key

# Hugging Face API konfiguratsiyasi
HF_API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment" # Misol model

# Gemini API konfiguratsiyasi
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Local fallback model
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "local_sentiment_model.pkl")
_local_model = None

def _load_local_model():
    """Lokal sentiment modelini yuklaydi, agar u mavjud bo'lsa."""
    global _local_model
    if _local_model is None:
        if os.path.exists(LOCAL_MODEL_PATH):
            try:
                _local_model = joblib.load(LOCAL_MODEL_PATH)
                logging.info("Lokal sentiment modeli yuklandi.")
            except Exception as e:
                error_logger.error(f"Lokal modelni yuklashda xato: {e}", exc_info=True)
                _local_model = None # Xato bo'lsa modelni None qilamiz
        else:
            logging.warning(f"Lokal model topilmadi: {LOCAL_MODEL_PATH}. Offline sentiment TextBlob orqali amalga oshiriladi.")
    return _local_model

def analyze_sentiment_hf(text: str) -> str:
    """
    Hugging Face API orqali matnning sentimentini tahlil qiladi.
    Args:
        text (str): Tahlil qilinadigan matn.
    Returns:
        str: "positive", "negative" yoki "neutral".
    """
    api_key = get_huggingface_api_key()
    if not api_key:
        logging.warning("Hugging Face API kaliti topilmadi yoki tugadi. Gemini fallbackga o'tish.")
        return analyze_sentiment_gemini(text)

    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": text}
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status() # HTTP xatolarini tekshirish
        result = response.json()
        # Natijani qayta ishlash
        # Misol: [{"label": "POSITIVE", "score": 0.99}, ...]
        if result and isinstance(result, list) and result[0] and isinstance(result[0], list) and result[0][0]:
            sentiment = result[0][0]['label'].lower()
            logging.info(f"Hugging Face sentiment: {sentiment}")
            return sentiment
        return "neutral"
    except requests.exceptions.RequestException as e:
        error_logger.error(f"Hugging Face API xatosi: {e}. Gemini fallbackga o'tish.", exc_info=True)
        return analyze_sentiment_gemini(text)
    except Exception as e:
        error_logger.error(f"Hugging Face sentiment tahlilida kutilmagan xato: {e}. Gemini fallbackga o'tish.", exc_info=True)
        return analyze_sentiment_gemini(text)

def analyze_sentiment_gemini(text: str) -> str:
    """
    Gemini API orqali matnning sentimentini tahlil qiladi.
    Args:
        text (str): Tahlil qilinadigan matn.
    Returns:
        str: "positive", "negative" yoki "neutral".
    """
    api_key = get_gemini_api_key()
    if not api_key:
        logging.warning("Gemini API kaliti topilmadi yoki tugadi. Lokal fallbackga o'tish.")
        return analyze_sentiment_local(text)

    # Gemini API uchun payload
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    { "text": f"Analyze the sentiment of the following text: '{text}'. Respond with a single word: 'positive', 'negative', or 'neutral'." }
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "sentiment": { "type": "STRING", "enum": ["positive", "negative", "neutral"] }
                }
            }
        }
    }
    apiUrl = f"{GEMINI_API_URL}?key={api_key}"

    try:
        response = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload), timeout=10)
        response.raise_for_status()
        result = response.json()

        if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
            # JSON formatida kelgan javobni parse qilish
            json_response_text = result["candidates"][0]["content"]["parts"][0]["text"]
            parsed_json = json.loads(json_response_text)
            sentiment = parsed_json.get("sentiment", "neutral").lower()
            logging.info(f"Gemini sentiment: {sentiment}")
            return sentiment
        return "neutral"
    except requests.exceptions.RequestException as e:
        error_logger.error(f"Gemini API xatosi: {e}. Lokal fallbackga o'tish.", exc_info=True)
        return analyze_sentiment_local(text)
    except json.JSONDecodeError as e:
        error_logger.error(f"Gemini javobini parse qilishda xato: {e}. Lokal fallbackga o'tish.", exc_info=True)
        return analyze_sentiment_local(text)
    except Exception as e:
        error_logger.error(f"Gemini sentiment tahlilida kutilmagan xato: {e}. Lokal fallbackga o'tish.", exc_info=True)
        return analyze_sentiment_local(text)


def analyze_sentiment_local(text: str) -> str:
    """
    Lokal model yoki TextBlob orqali matnning sentimentini tahlil qiladi.
    Args:
        text (str): Tahlil qilinadigan matn.
    Returns:
        str: "positive", "negative" yoki "neutral".
    """
    logging.info("Lokal sentiment tahlili ishga tushirildi.")
    model = _load_local_model()
    if model:
        # Agar model mavjud bo'lsa, u orqali bashorat qilish
        # Bu yerda model.predict() ga mos keladigan xususiyatlarni tayyorlash kerak.
        # Hozircha oddiy TextBlob misolini qaytaramiz, chunki modelni o'qitish
        # murakkabroq va bu skriptning doirasidan tashqarida.
        logging.info("Lokal model orqali sentiment tahlili (hozircha TextBlob orqali simulyatsiya qilinmoqda).")
        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0.1: # 0.1 dan yuqori bo'lsa positive
            return "positive"
        elif analysis.sentiment.polarity < -0.1: # -0.1 dan past bo'lsa negative
            return "negative"
        else: # -0.1 va 0.1 oralig'ida bo'lsa neutral
            return "neutral"
    else:
        # Agar model topilmasa, TextBlob dan foydalanish
        logging.info("TextBlob orqali sentiment tahlili.")
        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0.1:
            return "positive"
        elif analysis.sentiment.polarity < -0.1:
            return "negative"
        else:
            return "neutral"

if __name__ == "__main__":
    # Test uchun .env faylidan API kalitlarini yuklash
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

    print("Hugging Face sentiment (positive):", analyze_sentiment_hf("Bugun ob-havo juda yaxshi!"))
    print("Hugging Face sentiment (negative):", analyze_sentiment_hf("Qo'rqinchli xato yuz berdi."))
    print("Hugging Face sentiment (neutral):", analyze_sentiment_hf("Narx o'zgarmadi."))

    print("\nGemini sentiment (positive):", analyze_sentiment_gemini("Bitcoin narxi keskin ko'tarildi."))
    print("Gemini sentiment (negative):", analyze_sentiment_gemini("Bozor quladi, bu juda yomon."))
    print("Gemini sentiment (neutral):", analyze_sentiment_gemini("Bugun hech qanday muhim yangilik yo'q."))

    print("\nLokal sentiment (positive):", analyze_sentiment_local("Bu loyiha juda qiziqarli va foydali."))
    print("Lokal sentiment (negative):", analyze_sentiment_local("Men bu vaziyatdan noroziman."))
    print("Lokal sentiment (neutral):", analyze_sentiment_local("Stol ustida kitob bor."))
