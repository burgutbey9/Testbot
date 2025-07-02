Scalping AI Bot
Loyiha g'oyasi:
Ushbu loyiha AI, Order Flow, Sentiment tahlili va Backtest mexanizmlarini birlashtirgan avtomatlashtirilgan scalping botini yaratishga qaratilgan. Bot API rotation tizimi orqali API limitlariga tushib qolmasdan ishlashni ta'minlaydi.

Xususiyatlar:
Real vaqt API integratsiyasi: 1inch API va The Graph orqali DEX narxlari, Alchemy orqali blokcheyn ma'lumotlari.

Texnik indikatorlar: RSI, MACD, Bollinger Bands (BB), Average True Range (ATR) hisob-kitoblari (DEX ma'lumotlariga moslashtirilgan).

Order Flow tahlili: DEX kontekstida likvidlik, katta tranzaksiyalar va on-chain imbalance'ni aniqlash.

AI Sentiment tahlili: Hugging Face, Gemini va lokal fallback modellari orqali yangiliklar (NewsAPI.org) va ijtimoiy tarmoqlardagi (Reddit) sentimentni aniqlash.

Signal generator: Indikatorlar, sentiment va Order Flow ma'lumotlari asosida savdo signallarini yaratish.

Savdo manager: Riskni boshqarish, Stop-Loss (SL) / Take-Profit (TP) va favqulodda to'xtatish funksiyalari (DEX savdolariga moslashtirilgan).

Backtesting: Tarixiy CSV va tick-level ma'lumotlar bilan strategiyani sinash.

Strategiya optimallashtirish: results.csv asosida strategiya parametrlarini avtomatik optimallashtirish va optimized_config.json ga saqlash.

API Rotation: API kalitlarini avtomatik almashtirish tizimi.

Loglash: Barcha savdolar, xatolar, natijalar va API rotatsiyasi loglari.

Minimal API rotation talablari:
Botning uzluksiz ishlashi uchun quyidagi API kalitlari soni tavsiya etiladi:

1inch Developer Portal API: kamida 1 ta (ideal 5 ta, qolganlari The Graph bilan qoplanadi)

NewsAPI.org: kamida 1 ta API key

Alchemy API: kamida 1 ta API key

Hugging Face: kamida 2 ta API key (ideal 4 ta)

Gemini: kamida 3-5 ta API key

Reddit: kamida 1 ta API key

Papka tuzilishi:
project/ — Asosiy loyiha papkasi.

main.py — Asosiy bot logikasi.

data_fetcher.py — Turli API'lardan ma'lumotlarni olish funksiyalari.

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

Loyiha fayllarini yaratish/yangilash:

GitHub repozitoriyangizni kompyuteringizga klonlang.

Ushbu javobda berilgan har bir fayl kontentini o'zining tegishli GitHub fayliga to'liq almashtiring va o'zgarishlarni commit qilib, push qiling.

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

# 1inch Developer Portal API kalitlari (kamida 1 ta, ideal 5 ta)
ONE_INCH_API_KEY_1="sizning_1inch_api_key_1"
ONE_INCH_API_KEY_2="sizning_1inch_api_key_2"
ONE_INCH_API_KEY_3="sizning_1inch_api_key_3"
ONE_INCH_API_KEY_4="sizning_1inch_api_key_4"
ONE_INCH_API_KEY_5="sizning_1inch_api_key_5"

# NewsAPI.org kalitlari (kamida 1 ta)
NEWS_API_KEY_1="sizning_newsapi_key_1"

# Alchemy API kalitlari (kamida 1 ta)
ALCHEMY_API_KEY_1="sizning_alchemy_api_key_1"

# Hugging Face API kalitlari (kamida 2 ta, ideal 4 ta)
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

Loyihani ishga tushirish:
Virtual muhit faol bo'lgan holda, main.py faylini ishga tushiring:

python main.py

GitHub bilan sinxronizatsiya va yangilash:
Loyihani ZIP qilish: project/ papkasini ZIP arxiviga aylantiring.

GitHub'ga yuklash: ZIP arxivini GitHub repozitoriyangizga yuklang.

Hostda yangilash: Agar loyiha serverda joylashgan bo'lsa, git pull buyrug'i bilan eng so'nggi o'zgarishlarni tortib oling.

Loglarni kuzatish: Barcha loglar project/logs/ papkasiga tushadi. Menga GitHub linkini bersangiz, men loglardan natija va xatoni ko'rib, takomillashtirish bo'yicha yordam beraman.
