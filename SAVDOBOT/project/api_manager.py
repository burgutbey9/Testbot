"""
API_MANAGER.PY
---------------
Birja API manager:
- Candle ma'lumotlarini olish
- Order joylash
- Balansni olish
- API rotation (kalitlar almashtirish)
"""

import time
import hmac
import hashlib
import requests
import urllib.parse
import os
from logger import log_api_rotation

class APIManager:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        self.key_index = 0
        self.API_KEY = api_keys[self.key_index].split(':')[0]
        self.API_SECRET = api_keys[self.key_index].split(':')[1]
        self.BASE_URL = 'https://fapi.binance.com'  # Binance Futures endpoint

    def rotate_key(self):
        self.key_index += 1
        if self.key_index >= len(self.api_keys):
            print("⚠️ Barcha API kalitlar ishlatildi!")
            return False
        self.API_KEY = self.api_keys[self.key_index].split(':')[0]
        self.API_SECRET = self.api_keys[self.key_index].split(':')[1]
        log_api_rotation(f"API kalit almashtirildi: {self.key_index + 1}/{len(self.api_keys)}")
        return True

    def sign_request(self, params):
        query = urllib.parse.urlencode(params)
        signature = hmac.new(
            self.API_SECRET.encode('utf-8'),
            query.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    def get_candlestick(self, symbol='BTCUSDT', interval='1m', limit=50):
        url = f"{self.BASE_URL}/fapi/v1/klines"
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            import pandas as pd
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close',
                'volume', 'close_time', 'quote_asset_volume',
                'number_of_trades', 'taker_buy_base_asset_volume',
                'taker_buy_quote_asset_volume', 'ignore'
            ])
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df = df.astype(float)
            return df
        else:
            print(f"❌ Candlestick xato: {response.text}")
            return None

    def place_order(self, symbol, side, quantity, order_type='MARKET', price=None):
        url = f"{self.BASE_URL}/fapi/v1/order"
        timestamp = int(time.time() * 1000)
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'quantity': quantity,
            'timestamp': timestamp
        }
        if price and order_type == 'LIMIT':
            params['price'] = price
            params['timeInForce'] = 'GTC'
        params['signature'] = self.sign_request(params)
        headers = {'X-MBX-APIKEY': self.API_KEY}

        response = requests.post(url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:  # Rate limit xatosi
            if self.rotate_key():
                return self.place_order(symbol, side, quantity, order_type, price)
        else:
            print(f"❌ Order xato: {response.text}")
            return None

    def get_balance(self):
        url = f"{self.BASE_URL}/fapi/v2/balance"
        timestamp = int(time.time() * 1000)
        params = {'timestamp': timestamp}
        params['signature'] = self.sign_request(params)
        headers = {'X-MBX-APIKEY': self.API_KEY}

        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            balances = response.json()
            usdt_balance = next((float(b['balance']) for b in balances if b['asset'] == 'USDT'), 0)
            return usdt_balance
        else:
            print(f"❌ Balance xato: {response.text}")
            return 0
