# SAVDO BOT

## 📌 Loyiha haqida
AI indikator, sentiment, risk boshqaruv, backtest — hammasi avtomatlashtirilgan kripto scalping bot.

## ⚙️ Texnik imkoniyatlar
- Binance/Bybit API (yoki o‘rnini bosuvchi bepul)
- RSI, MACD, BB, ATR indikatorlari
- AI sentiment (TextBlob, huggingface, yoki boshqa)
- Risk boshqarish, favqulodda stop
- API rotation
- Backtest
- Log fayllar: trades, errors, rotation, results

## 🚀 Ishga tushirish
```bash
# 1. Talablarni o‘rnatish
pip install -r requirements.txt

# 2. .env fayl yaratish
# .env.example ni .env ga ko‘chirib, kalitlarni to‘ldiring

# 3. Botni ishga tushirish
python main.py
