# project/main.py - Asosiy bot kodi

import asyncio
import json
import logging
import os
import time
from datetime import datetime

# Loyiha modullarini import qilish
from project.api_manager import (
    get_one_inch_api_key, get_news_api_key, get_alchemy_api_key,
    get_huggingface_api_key, get_gemini_api_key, get_reddit_api_key
)
from project.data_fetcher import (
    fetch_1inch_quote, fetch_alchemy_gas_price, fetch_news_articles,
    fetch_reddit_posts, fetch_the_graph_data, UNISWAP_V3_SUBGRAPH_URL # The Graph uchun
)
from project.orderflow import detect_large_orders, calculate_tick_speed, apply_imbalance_filter # Order Flow tahlili DEX ga moslashtiriladi
from project.ai_sentiment import analyze_sentiment_hf, analyze_sentiment_gemini, analyze_sentiment_local
from project.trainer import optimize_strategy
from project.backtest import run_backtest_csv, run_backtest_tick_level

# Log konfiguratsiyasi
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOGS_DIR, exist_ok=True) # logs papkasini yaratish

# Asosiy bot loglari
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(os.path.join(LOGS_DIR, "trades.log"), encoding='utf-8', mode='a'),
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
api_rotation_handler = logging.FileHandler(os.path.join(LOGS_DIR, "api_rotation.log"), encoding='utf-8', mode='a')
api_rotation_handler.setLevel(logging.INFO)
api_rotation_logger.addHandler(api_rotation_handler)
api_rotation_logger.propagate = False

# Global o'zgaruvchilar
# 1inch API uchun misol tokenlar va zanjir ID
FROM_TOKEN_ADDRESS = "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE" # ETH (native token)
TO_TOKEN_ADDRESS = "0xdAC17F958D2ee523a2206206994597C13D831ec7" # USDT
CHAIN_ID = 1 # Ethereum mainnet
AMOUNT = 10**18 # 1 ETH (minimal hajm)

# Mock data for technical indicators (in a real scenario, this would come from OHLCV data)
# This is a simplified example; actual indicator calculation requires a series of data.
current_price = 3000.0 # Misol narx, ETH/USDT uchun
rsi_value = 50.0
macd_value = 0.0
bb_upper = 3050.0
bb_lower = 2950.0
atr_value = 50.0

async def calculate_technical_indicators(ohlcv_data: list[dict]) -> dict:
    """
    OHLCV ma'lumotlari asosida texnik indikatorlarni hisoblaydi.
    Bu yerda 'ta' (Technical Analysis) kutubxonasidan foydalanish mumkin.
    Hozircha mock qiymatlar.
    """
    if not ohlcv_data:
        return {"rsi": None, "macd": None, "bb_upper": None, "bb_lower": None, "atr": None}

    # Bu yerda real hisoblashlar bo'ladi.
    # Misol uchun, pandas va 'ta' kutubxonasi yordamida:
    # import pandas as pd
    # import ta
    # df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    # df['close'] = pd.to_numeric(df['close'])
    # df['high'] = pd.to_numeric(df['high'])
    # df['low'] = pd.to_numeric(df['low'])
    # rsi = ta.momentum.RSIIndicator(df['close']).rsi().iloc[-1]
    # macd = ta.trend.MACD(df['close']).macd().iloc[-1]
    # bb_upper = ta.volatility.BollingerBands(df['close']).bollinger_hband().iloc[-1]
    # bb_lower = ta.volatility.BollingerBands(df['close']).bollinger_lband().iloc[-1]
    # atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range().iloc[-1]

    return {
        "rsi": rsi_value,
        "macd": macd_value,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "atr": atr_value
    }

async def execute_trade(trade_type: str, from_token: str, to_token: str, amount: int, chain_id: int, price: float | None = None, stop_loss: float | None = None, take_profit: float | None = None):
    """
    Savdo operatsiyasini 1inch API orqali amalga oshiradi va riskni boshqaradi.
    Args:
        trade_type (str): "BUY" (token sotib olish) yoki "SELL" (token sotish).
        from_token (str): Sotiladigan token manzili.
        to_token (str): Sotib olinadigan token manzili.
        amount (int): Savdo hajmi (tokenning minimal birligida, masalan, wei).
        chain_id (int): Blokcheyn zanjiri ID.
        price (float, optional): Buyurtma narxi (limit order uchun). None bo'lsa, market order.
        stop_loss (float, optional): Stop-Loss narxi.
        take_profit (float, optional): Take-Profit narxi.
    """
    logging.info(f"Savdo amalga oshirilmoqda: {trade_type} {amount} {from_token} -> {to_token} (ChainID: {chain_id})")
    
    # 1inch API orqali swap (almashtirish) operatsiyasini amalga oshirish
    # Bu yerda sizning wallet adresingiz va private key kerak bo'ladi
    # va tranzaksiyani imzolash, keyin Alchemy orqali yuborish kerak.
    # Bu qism murakkab va Web3.py kutubxonasini talab qiladi.
    # Web3.py orqali tranzaksiyani imzolash va yuborish logikasi qo'shiladi.

    # Misol uchun:
    # try:
    #     one_inch_api_key = get_one_inch_api_key()
    #     alchemy_api_key = get_alchemy_api_key()
    #     if not one_inch_api_key or not alchemy_api_key:
    #         error_logger.error("1inch yoki Alchemy API kaliti topilmadi. Savdo amalga oshirilmadi.")
    #         return

    #     # 1inch swap tranzaksiya ma'lumotlarini olish (GET /swap)
    #     # Bu yerda Web3.py va 1inch API'ning /swap endpointi ishlatiladi
    #     # swap_tx_data = await fetch_1inch_swap_transaction(one_inch_api_key, from_token, to_token, amount, chain_id, YOUR_WALLET_ADDRESS, slippage=1)
    #     # if swap_tx_data:
    #     #     # Tranzaksiyani imzolash va yuborish (Alchemy orqali)
    #     #     w3 = Web3(Web3.HTTPProvider(f"https://eth-mainnet.g.alchemy.com/v2/{alchemy_api_key}")) # Chain ID ga qarab URL o'zgaradi
    #     #     # account = w3.eth.account.from_key(YOUR_PRIVATE_KEY)
    #     #     # signed_txn = account.sign_transaction(swap_tx_data)
    #     #     # tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
    #     #     logging.info(f"Savdo muvaffaqiyatli yuborildi. Tx Hash: {tx_hash.hex()}")
    #     #     with open(os.path.join(LOGS_DIR, "trades.log"), "a", encoding='utf-8') as f:
    #     #         f.write(f"{datetime.now().isoformat()}, {trade_type}, {amount}, {from_token}, {to_token}, {chain_id}, {tx_hash.hex()}\n")
    #     # else:
    #     #     error_logger.error("1inch swap tranzaksiyasini olib bo'lmadi.")
    # except Exception as e:
    #     error_logger.error(f"Savdo amalga oshirishda xato: {e}", exc_info=True)

    await asyncio.sleep(0.5) # Savdo bajarilishini simulyatsiya qilish
    logging.info(f"Savdo {trade_type} {amount} {from_token} -> {to_token} (ChainID: {chain_id}) simulyatsiya qilindi.")
    with open(os.path.join(LOGS_DIR, "trades.log"), "a", encoding='utf-8') as f:
        f.write(f"{datetime.now().isoformat()}, {trade_type}, {amount}, {from_token}, {to_token}, SIMULATED\n")

    # Risk boshqaruvi (SL/TP) logikasi shu yerda bo'ladi.
    # DEX savdolarida SL/TP to'g'ridan-to'g'ri birja tomonidan emas,
    # balki botning o'zi tomonidan kuzatilishi kerak.
    # Bu qismda bot ochiq pozitsiyani kuzatib borishi va narx SL/TP ga yetganda
    # yangi swap tranzaksiyasini boshlashi kerak.


async def bot_main_loop():
    """
    Botning asosiy ish sikli.
    Ma'lumotlarni oladi, tahlil qiladi va savdo signallarini yaratadi.
    """
    while True:
        try:
            # 1. Narx ma'lumotlarini olish (1inch orqali asosiy, The Graph orqali zaxira)
            current_price = None
            one_inch_api_key = get_one_inch_api_key()
            if one_inch_api_key:
                quote_data = await fetch_1inch_quote(one_inch_api_key, FROM_TOKEN_ADDRESS, TO_TOKEN_ADDRESS, AMOUNT, CHAIN_ID)
                if quote_data and 'toTokenAmount' in quote_data and float(quote_data['toTokenAmount']) > 0:
                    current_price = float(quote_data['fromTokenAmount']) / float(quote_data['toTokenAmount'])
                    logging.info(f"1inch orqali {FROM_TOKEN_ADDRESS} / {TO_TOKEN_ADDRESS} narxi: {current_price:.6f}")
                else:
                    logging.warning("1inch orqali narx ma'lumotlarini olib bo'lmadi. The Graph orqali urinish.")
            else:
                logging.error("1inch API kaliti topilmadi. Narx ma'lumotlarini olish imkonsiz. The Graph orqali urinish.")
            
            # Agar 1inch orqali narx olib bo'lmasa, The Graph orqali urinish
            if current_price is None:
                logging.info("The Graph orqali narx olishga urinish...")
                # Misol GraphQL so'rovi (real Uniswap V3 subgraph uchun ETH/USD narxini olish)
                # Bu yerda sizning aniq token juftligingizga mos subgraph va so'rov bo'lishi kerak
                graphql_query = """
                {
                  bundles(first: 1, where: {id: "1"}) { # Bundle ID 1 odatda ETH/USD narxini beradi
                    ethPriceUSD
                  }
                }
                """
                graph_data = await fetch_the_graph_data(graphql_query)
                if graph_data and graph_data.get('bundles') and len(graph_data['bundles']) > 0:
                    eth_price_usd = float(graph_data['bundles'][0]['ethPriceUSD'])
                    # USDT bilan almashtirish uchun taxminiy narx
                    current_price = eth_price_usd # Agar to_token USDT bo'lsa
                    logging.info(f"The Graph orqali narx: {current_price:.6f}")
                else:
                    logging.warning("The Graph orqali ham narx olib bo'lmadi.")
            
            if current_price is None:
                error_logger.error("Narx ma'lumotlarini olib bo'lmadi. Bot ishlashni davom ettira olmaydi.")
                await asyncio.sleep(60) # Kuting va qayta urinish
                continue # Keyingi iteratsiyaga o'tish


            # 2. Gaz narxini olish (Alchemy orqali)
            alchemy_api_key = get_alchemy_api_key()
            if alchemy_api_key:
                gas_price_gwei = await fetch_alchemy_gas_price(alchemy_api_key, CHAIN_ID)
                if gas_price_gwei:
                    logging.info(f"Joriy gaz narxi ({CHAIN_ID}): {gas_price_gwei} Gwei")
                else:
                    logging.warning("Alchemy orqali gaz narxini olib bo'lmadi.")
            else:
                logging.error("Alchemy API kaliti topilmadi. Gaz narxini olish imkonsiz.")


            # 3. AI Sentiment tahlili uchun ma'lumotlarni olish
            # NewsAPI.org orqali yangiliklar
            news_api_key = get_news_api_key()
            if news_api_key:
                news_articles = await fetch_news_articles(news_api_key, "cryptocurrency")
                if news_articles:
                    for article in news_articles[:2]: # Faqat bir nechta maqolani tahlil qilish
                        logging.info(f"Yangilik sarlavhasi: {article['title']}")
                        sentiment_hf = analyze_sentiment_hf(article['title'])
                        sentiment_gemini = analyze_sentiment_gemini(article['title'])
                        logging.info(f"Sentiment (HF): {sentiment_hf}, (Gemini): {sentiment_gemini}")
                else:
                    logging.warning("NewsAPI.org orqali yangiliklarni olib bo'lmadi.")
            else:
                logging.error("NewsAPI.org API kaliti topilmadi. Yangiliklarni olish imkonsiz.")

            # Reddit API orqali postlar (sentiment uchun)
            reddit_api_config = get_reddit_api_key()
            if reddit_api_config:
                reddit_posts = await fetch_reddit_posts(reddit_api_config, "cryptocurrency")
                if reddit_posts:
                    for post in reddit_posts[:2]: # Faqat bir nechta postni tahlil qilish
                        logging.info(f"Reddit post sarlavhasi: {post['title']}")
                        sentiment_hf = analyze_sentiment_hf(post['title'])
                        sentiment_gemini = analyze_sentiment_gemini(post['title'])
                        logging.info(f"Sentiment (HF): {sentiment_hf}, (Gemini): {sentiment_gemini}")
                else:
                    logging.warning("Reddit orqali postlarni olib bo'lmadi.")
            else:
                logging.error("Reddit API kalitlari topilmadi. Reddit postlarini olish imkonsiz.")

            # 4. Order Flow tahlili (DEX ma'lumotlariga moslashtirilgan)
            # DEX da an'anaviy Order Book bo'lmagani uchun, bu yerda on-chain ma'lumotlar tahlil qilinadi.
            # Masalan, katta tranzaksiyalar hajmi, gaz narxining o'zgarishi, likvidlik holati.
            # Hozircha bu qismda simulyatsiya qilingan ma'lumotlar ishlatiladi.
            mock_depth_data = {
                "bids": [[current_price * 0.999, 100], [current_price * 0.998, 50]],
                "asks": [[current_price * 1.001, 120], [current_price * 1.002, 60]]
            }
            large_orders = detect_large_orders(mock_depth_data) # Mock depth bilan ishlash
            imbalance = apply_imbalance_filter(mock_depth_data) # Mock depth bilan ishlash
            logging.info(f"Order Flow tahlili (DEX): Katta buyurtmalar: {large_orders}, Imbalance: {imbalance}")

            # Tick tezligi (DEX da tranzaksiya tezligi yoki blok tezligi bo'lishi mumkin)
            # Bu yerda Alchemy orqali real tranzaksiya ma'lumotlari olinishi mumkin
            mock_tick_data = {"event_type": "transaction", "data": "mock_tx_hash"}
            tick_speed = calculate_tick_speed(mock_tick_data)
            logging.info(f"Tick Tezligi (DEX): {tick_speed}")


            # 5. Texnik indikatorlar (OHLCV ma'lumotlari mavjud bo'lganda)
            # DEX uchun OHLCV ma'lumotlarini olish murakkabroq bo'lishi mumkin (masalan, The Graph orqali).
            # Hozircha mock data ishlatiladi.
            mock_ohlcv_data = [{"timestamp": datetime.now().timestamp() * 1000, "open": current_price - 10, "high": current_price + 10, "low": current_price - 20, "close": current_price, "volume": 100}]
            indicators = await calculate_technical_indicators(mock_ohlcv_data)
            logging.info(f"Indikatorlar: {indicators}")

            # 6. Signal generator: indikator + sentiment + order flow asosida savdo signali yaratish
            # Bu yerda savdo logikasi joylashadi
            # Misol shartlar: Ijobiy sentiment, RSI past (overbought emas), katta buyurtmalar, ijobiy imbalance
            sentiment_summary = None
            if sentiment_gemini == "positive" or sentiment_hf == "positive":
                sentiment_summary = "positive"
            elif sentiment_gemini == "negative" or sentiment_hf == "negative":
                sentiment_summary = "negative"
            else:
                sentiment_summary = "neutral"

            if sentiment_summary == "positive" and \
               indicators["rsi"] is not None and indicators["rsi"] < 40 and \
               large_orders["large_bids"] and imbalance > 0.05: # Imbalance chegarasi biroz pasaytirildi
                logging.info("LONG savdo signali! Savdo amalga oshirilmoqda...")
                await execute_trade("BUY", FROM_TOKEN_ADDRESS, TO_TOKEN_ADDRESS, AMOUNT, CHAIN_ID, price=current_price * 1.001, stop_loss=current_price * 0.995)
            # Misol shartlar: Salbiy sentiment, RSI yuqori (oversold emas), katta sotish buyurtmalari, salbiy imbalance
            elif sentiment_summary == "negative" and \
                 indicators["rsi"] is not None and indicators["rsi"] > 60 and \
                 large_orders["large_asks"] and imbalance < -0.05: # Imbalance chegarasi biroz pasaytirildi
                logging.info("SHORT savdo signali! Savdo amalga oshirilmoqda...")
                await execute_trade("SELL", TO_TOKEN_ADDRESS, FROM_TOKEN_ADDRESS, AMOUNT * current_price, CHAIN_ID, price=current_price * 0.999, take_profit=current_price * 0.995)
            else:
                logging.info("Savdo signali mavjud emas. Kutish...")

            await asyncio.sleep(60) # Har 60 soniyada bir marta ishga tushirish

        except Exception as e:
            error_logger.error(f"Botning asosiy siklida kutilmagan xato: {e}", exc_info=True)
            logging.error("15 soniyadan keyin qayta urinish...")
            await asyncio.sleep(15)

async def main():
    """
    Asosiy ishga tushirish funksiyasi.
    Botni ishga tushiradi va backtest/optimallashtirish funksiyalarini boshqaradi.
    """
    logging.info("Bot ishga tushirildi! Ma'lumotlarni olish va tahlil qilishni boshlamoqda...")

    # Backtestni ishga tushirish misoli (ishga tushirishdan oldin # belgisini olib tashlang)
    # logging.info("Backtest ishga tushirilmoqda...")
    # run_backtest_csv(os.path.join(os.path.dirname(__file__), "sample_historical_data.csv")) # Mavjud CSV fayl yo'lini kiriting
    # run_backtest_tick_level(os.path.join(os.path.dirname(__file__), "sample_tick_data.csv")) # Mavjud CSV fayl yo'lini kiriting

    # Strategiyani optimallashtirish misoli (ishga tushirishdan oldin # belgisini olib tashlang)
    # logging.info("Strategiya optimallashtirilmoqda...")
    # optimize_strategy(os.path.join(LOGS_DIR, "results.csv"))

    # Asosiy botni ishga tushirish
    await bot_main_loop()

if __name__ == "__main__":
    # Loyihani ishga tushirish uchun asosiy nuqta
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Bot foydalanuvchi tomonidan to'xtatildi.")
    except Exception as e:
        error_logger.critical(f"Botning ishga tushirilishida halokatli xato: {e}", exc_info=True)
