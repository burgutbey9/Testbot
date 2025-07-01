"""
LOGGER.PY
-----------
Barcha loglarni yozib boradi:
- Xatolar (errors.log)
- Savdo harakatlari (trades.log)
- API rotation (api_rotation.log)
"""

from datetime import datetime

def log_error(message: str):
    with open("logs/errors.log", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.utcnow()}] ❌ ERROR: {message}\n")

def log_trade(message: str):
    with open("logs/trades.log", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.utcnow()}] ✅ TRADE: {message}\n")

def log_api_rotation(message: str):
    with open("logs/api_rotation.log", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.utcnow()}] 🔄 API: {message}\n")
