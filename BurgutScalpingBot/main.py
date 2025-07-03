import asyncio
from strategies.ai_strategy import AIScalpingStrategy
from utils.notifier import TelegramNotifier
from utils.api_rotation import APIRotator
from config.config import Config

notifier = TelegramNotifier()
rotator = APIRotator()
strategy = AIScalpingStrategy()

async def main():
    await notifier.send_message("🤖 BurgutScalpingBot ishga tushdi!")
    while True:
        signal = strategy.generate_signal()
        if signal:
            await notifier.send_message(f"Yangi signal: {signal}")
        await asyncio.sleep(Config.CHECK_INTERVAL)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot to'xtatildi.")
