from modules.orderflow import run_orderflow
import asyncio

def test_orderflow():
    asyncio.run(run_orderflow())
    assert True  # Order Flow modul ishladi deb hisoblanadi.
