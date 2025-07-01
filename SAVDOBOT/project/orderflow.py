"""
ORDERFLOW.PY
----------------
- Bozor orderflow (hajm va delta) tahlili
- Whale activity va kutilmagan katta hajmni aniqlash
"""

import logging

def analyze_orderflow(data: list):
    """
    data: [{'volume': .., 'buy_volume': .., 'sell_volume': .., 'delta': ..}]
    """
    signals = []
    for candle in data:
        total_volume = candle.get('volume', 0)
        delta = candle.get('delta', 0)
        buy_volume = candle.get('buy_volume', 0)
        sell_volume = candle.get('sell_volume', 0)

        if total_volume > 1000 and abs(delta) > total_volume * 0.7:
            signals.append({
                'whale_detected': True,
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'delta': delta
            })
    return signals


if __name__ == "__main__":
    # Misol
    candles = [
        {'volume': 1500, 'buy_volume': 1000, 'sell_volume': 500, 'delta': 500},
        {'volume': 800, 'buy_volume': 400, 'sell_volume': 400, 'delta': 0}
    ]
    res = analyze_orderflow(candles)
    print("Orderflow signals:", res)
