Scalping AI Bot
Loyiha g'oyasi:
Ushbu loyiha AI, Order Flow, Sentiment tahlili va Backtest mexanizmlarini birlashtirgan avtomatlashtirilgan scalping botini yaratishga qaratilgan. Bot API rotation tizimi orqali API limitlariga tushib qolmasdan ishlashni ta'minlaydi.

Xususiyatlar:
Real vaqt API integratsiyasi: Binance WebSocket, OHLCV va Order Book ma'lumotlari.

Texnik indikatorlar: RSI, MACD, Bollinger Bands (BB), Average True Range (ATR) hisob-kitoblari.

Order Flow tahlili: Imbalance, tick tezligi va katta buyurtmalarni aniqlash.

AI Sentiment tahlili: Hugging Face, Gemini va lokal fallback modellari orqali yangiliklar va ijtimoiy tarmoqlardagi sentimentni aniqlash.

Signal generator: Indikatorlar, sentiment va Order Flow ma'lumotlari asosida savdo signallarini yaratish.

Savdo manager: Riskni boshqarish, Stop-Loss (SL) / Take-Profit (TP) va favqulodda to'xtatish funksiyalari.

Backtesting: Tarixiy CSV va tick-level ma'lumotlar bilan strategiyani sinash.

Strategiya optimallashtirish: results.csv asosida strategiya parametrlarini avtomatik optimallashtirish va optimized_config.json ga saqlash.

API Rotation: API kalitlarini avtomatik almashtirish tizimi.

Loglash: Barcha savdolar, xatolar, natijalar va API rotatsiyasi loglari.

Minimal API rotation talablari:
Botning uzluksiz ishlashi uchun quyidagi API kalitlari soni tavsiya etiladi:

Binance: kamida 5 ta API key

Hugging Face: kamida 4 ta API key

Gemini: kamida 3-5 ta API key

OpenAI: kamida 2-3 ta API key

Twitter: kamida 2 ta API key

Reddit: kamida 1 ta API key

Zaxira: Bybit API key va lokal fallback modeli (sentiment uchun)

Papka tuzilishi:
project/ — Asosiy loyiha papkasi.

main.py — Asosiy bot logikasi.

backtest.py — Backtest funksiyalari.

orderflow.py — Order Flow tahlili.

trainer.py — Strategiya optimallashtirish va AI o'qitish.

ai_sentiment.py — AI sentiment tahlili.

api_manager.py — API kalitlarini boshqarish va rotatsiya.

README.md — Ushbu fayl.

PROJECT_IDEA.txt — Loyiha g'oyasi va batafsil tushuntirish.

requirements.txt — Loyiha uchun zarur Python kutubxonalari.

.env.example — API kalitlari uchun namuna fayl.

optimized_config.json — Optimizatsiya qilingan strategiya sozlamalari.

local_sentiment_model.pkl — Lokal sentiment modeli (agar mavjud bo'lsa).

logs/ — Barcha natija, error va rotation loglar shu yerga tushadi.

trades.log — Har bir savdo operatsiyasi logi.

errors.log — Xatoliklar logi.

results.csv — Backtest va PnL natijalari.

api_rotation.log — API rotatsiyasi tarixi.

Qadam-baqadam o'rnatish:
Python o'rnatish: Kompyuteringizda Python 3.9+ o'rnatilganligiga ishonch hosil qiling.
(Agar o'rnatilmagan bo'lsa, rasmiy saytdan yuklab oling yoki terminal orqali o'rnating.)

Loyiha fayllarini yaratish:

Ushbu README.md fayli joylashgan loyiha papkasini yarating (masalan, SAVDOBOT).

generate_project.py skriptini loyiha ildiz papkasiga joylashtiring.

Terminalni oching, loyiha papkasiga o'ting (cd C:\Users\YourUser\Desktop\SAVDOBOT).

Quyidagi buyruqni ishga tushiring:

python generate_project.py

Bu buyruq project/ papkasini va uning ichidagi barcha fayllarni avtomatik yaratadi.

Virtual muhit yaratish (tavsiya etiladi):

cd project
python -m venv venv

Virtual muhitni faollashtirish:

Windows:

.\venv\Scripts\activate

macOS/Linux:

source venv/bin/activate

Kutubxonalarni o'rnatish:

pip install -r requirements.txt

API kalitlarini sozlash:

project/.env.example faylini project/.env nomiga o'zgartiring.

project/.env faylini oching va o'z API kalitlaringizni kiriting. Har bir API uchun minimal talab qilingan kalitlarni kiritishingiz kerak.

.env namunasi:
# Barcha API key namunalari shu yerga yoziladi.

# Binance API kalitlari (kamida 5 ta)
BINANCE_API_KEY_1="sizning_binance_api_key_1"
BINANCE_SECRET_KEY_1="sizning_binance_secret_key_1"
BINANCE_API_KEY_2="sizning_binance_api_key_2"
BINANCE_SECRET_KEY_2="sizning_binance_secret_key_2"
BINANCE_API_KEY_3="sizning_binance_api_key_3"
BINANCE_SECRET_KEY_3="sizning_binance_secret_key_3"
BINANCE_API_KEY_4="sizning_binance_api_key_4"
BINANCE_SECRET_KEY_4="sizning_binance_secret_key_4"
BINANCE_API_KEY_5="sizning_binance_api_key_5"
BINANCE_SECRET_KEY_5="sizning_binance_secret_key_5"

# Hugging Face API kalitlari (kamida 4 ta)
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

# OpenAI API kalitlari (kamida 2-3 ta)
OPENAI_API_KEY_1="sizning_openai_api_key_1"
OPENAI_API_KEY_2="sizning_openai_api_key_2"
OPENAI_API_KEY_3="sizning_openai_api_key_3"

# Twitter API kalitlari (kamida 2 ta)
TWITTER_API_KEY_1="sizning_twitter_api_key_1"
TWITTER_SECRET_KEY_1="sizning_twitter_secret_key_1"
TWITTER_API_KEY_2="sizning_twitter_api_key_2"
TWITTER_SECRET_KEY_2="sizning_twitter_secret_key_2"

# Reddit API kaliti (kamida 1 ta)
REDDIT_CLIENT_ID_1="sizning_reddit_client_id_1"
REDDIT_CLIENT_SECRET_1="sizning_reddit_client_secret_1"
REDDIT_USERNAME_1="sizning_reddit_username_1"
REDDIT_PASSWORD_1="sizning_reddit_password_1"

# Bybit API kaliti (zaxira)
BYBIT_API_KEY_1="sizning_bybit_api_key_1"
BYBIT_SECRET_KEY_1="sizning_bybit_secret_key_1"

Loyihani ishga tushirish:
Virtual muhit faol bo'lgan holda, main.py faylini ishga tushiring:

python main.py

GitHub bilan sinxronizatsiya va yangilash:
Loyihani ZIP qilish: project/ papkasini ZIP arxiviga aylantiring.

GitHub'ga yuklash: ZIP arxivini GitHub repozitoriyangizga yuklang.

Hostda yangilash: Agar loyiha serverda joylashgan bo'lsa, git pull buyrug'i bilan eng so'nggi o'zgarishlarni tortib oling.

Loglarni kuzatish: Barcha loglar project/logs/ papkasiga tushadi. Menga GitHub linkini bersangiz, men loglardan natija va xatoni ko'rib, takomillashtirish bo'yicha yordam beraman.
