from modules.backtest import run_backtest
import asyncio

def test_backtest():
    asyncio.run(run_backtest())
    assert True  # Backtest tugadi.
