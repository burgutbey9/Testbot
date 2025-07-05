import asyncio
import logging
from datetime import datetime
from modules.orderflow import run_orderflow
from modules.ai_sentiment import run_sentiment
from modules.api_manager import check_apis
from modules.utils import send_telegram_status, setup_logging

# Logging setup
logger = logging.getLogger(__name__)

async def main():
    """
    BurgutScalpingBot asosiy funksiyasi
    DEX Auto-Scalping bot ishga tushirish
    """
    
    # Logging tizimini sozlash
    setup_logging()
    
    logger.info("🚀 BurgutScalpingBot ishga tushmoqda...")
    
    try:
        # API'larni tekshirish
        logger.info("🔍 API'larni tekshirish...")
        await check_apis()
        
        # Telegram orqali start xabarini yuborish
        start_message = f"🚀 BurgutScalpingBot ishga tushdi!\n📅 Vaqt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n✅ Barcha API'lar tekshirildi"
        await send_telegram_status(start_message)
        
        # Asosiy modullarni parallel ishga tushirish
        logger.info("🔄 Asosiy modullarni ishga tushirish...")
        
        # Async tasklar yaratish
        tasks = [
            asyncio.create_task(run_orderflow(), name="OrderFlow"),
            asyncio.create_task(run_sentiment(), name="AI_Sentiment"),
            asyncio.create_task(periodic_status_update(), name="Status_Update")
        ]
        
        # Tasklar parallel ishlashi
        await asyncio.gather(*tasks, return_exceptions=True)
        
    except KeyboardInterrupt:
        logger.info("⏹️ Bot to'xtatildi (Ctrl+C)")
        await send_telegram_status("⏹️ BurgutScalpingBot to'xtatildi")
        
    except Exception as e:
        logger.error(f"❌ Kritik xato: {e}")
        await send_telegram_status(f"❌ Bot xatosi: {str(e)}")
        
    finally:
        # Cleanup
        logger.info("🧹 Resurslarni tozalash...")
        await cleanup_resources()

async def periodic_status_update():
    """
    Har 1 soatda status yuborish
    """
    while True:
        try:
            await asyncio.sleep(3600)  # 1 soat
            
            status_message = f"🔄 Bot ishlayapti\n📅 Vaqt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n✅ Barcha modullar faol"
            await send_telegram_status(status_message)
            
        except Exception as e:
            logger.error(f"Status update xatosi: {e}")
            await asyncio.sleep(300)  # 5 daqiqa kutish

async def cleanup_resources():
    """
    Resurslarni tozalash
    """
    try:
        # Faol tasklar ro'yxatini olish
        tasks = [task for task in asyncio.all_tasks() if not task.done()]
        
        if tasks:
            logger.info(f"🧹 {len(tasks)} ta task tozalanmoqda...")
            
            # Barcha tasklar cancel qilish
            for task in tasks:
                task.cancel()
            
            # Tasklar tugashini kutish
            await asyncio.gather(*tasks, return_exceptions=True)
            
        logger.info("✅ Resurslar tozalandi")
        
    except Exception as e:
        logger.error(f"Cleanup xatosi: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ Bot to'xtatildi")
    except Exception as e:
        print(f"❌ Bot ishga tushmadi: {e}")
