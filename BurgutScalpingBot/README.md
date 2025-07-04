# 📂 AI Scalping Bot


## 🎯 Loyiha maqsadi
AI yordamida **DEX scalping** qiladigan **avtomatik bot**. Bu bot **real vaqt Order Flow** va **AI Sentiment** bilan ishlaydi. Backtest va optimizatsiya imkoniyati mavjud. 

## ⚙️ Asosiy imkoniyatlar
- 📈 **Order Flow kuzatish** — DEX (masalan, Uniswap) da tranzaksiyalarni real vaqt tahlil qiladi.
- 🤖 **AI Sentiment** — HuggingFace va Google Gemini asosida yangiliklar va sentiment analiz qiladi.
- 🧪 **Backtest** — Tarixiy ma’lumotlarda strategiyani sinab ko‘radi.
- 🔁 **Trainer** — Strategiyani doimiy optimizatsiya qiladi va o‘zini yangilaydi.
- 📝 **Log** — Har bir jarayon `logs/bot.log` ga yoziladi.
- 🔔 **Telegram Status** — Bot ishlayotgani, holati, xatoliklar va balans bo‘yicha xabar yuboradi.

## 🔑 API’lar

- **1inch API** — Asosiy DEX ma’lumotlari.
- **Alchemy API** — Blokcheyn bilan to‘g‘ridan-to‘g‘ri ishlash.
- **NewsAPI** — Yangiliklar manbai.
- **HuggingFace + Gemini** — AI sentiment uchun.
- **Reddit API** — Ijtimoiy tahlil uchun (agar kerak bo‘lsa).

`.env.example` faylida barchasi ko‘rsatilgan.

## 🚦 Ishlash tartibi
1️⃣ `config.py` orqali barcha sozlashlar.  
2️⃣ `main.py` — Bot ishga tushiriladi.  
3️⃣ Modullar (orderflow, sentiment, backtest, trainer) parallel ishlaydi.  
4️⃣ Telegramga holat yuboriladi.  
5️⃣ Log faylga yoziladi.

## 🔑 Muhim eslatma
- GPT API ishlatilmaydi.
- Strategiyalar `data/strategies/` papkasida saqlanadi.
- Backtest natijalari `data/backtest_results/` ga yoziladi.
- Model fayllari `data/ai_models/` ga tushadi.




## Telegram
Bot faqat ish holati va balans statusini yuboradi.

## Fayllar va papkalar
- `project_info.txt` — to‘liq tushuntirish
- `requirements.txt` — kutubxonalar
- `.env.example` — sozlash
- `main.py` — ishga tushirish kodi
- `modules/` — modullar
- `strategy/` — strategiyalar
- `logs/` — loglar

## Ishga tushirish
1. `.env` ni sozlash  
2. Kutubxonalarni o‘rnatish: `pip install -r requirements.txt`  
3. `main.py` ni ishga tushirish

🚀 Omad!
