import asyncio
from modules.orderflow import run_orderflow
from modules.ai_sentiment import run_sentiment
from modules.backtest import run_backtest
from modules.trainer import train_model
from modules.api_manager import check_apis
from modules.utils import send_telegram_status

async def main():
    await check_apis()
    await asyncio.gather(
        run_orderflow(),
        run_sentiment(),
        run_backtest(),
        train_model(),
        send_telegram_status(),
    )

if __name__ == "__main__":
    asyncio.run(main())
