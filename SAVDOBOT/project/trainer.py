"""
TRAINER.PY
-------------
- Backtest qilish
- AI signalni tekshirish
- Natijani results.csv ga yozish
"""

import csv
from datetime import datetime
from logger import log_error

class Trainer:
    def __init__(self):
        self.results_file = "logs/results.csv"

    def backtest_signal(self, price_data: list, signal_logic):
        """
        price_data: [{'datetime': .., 'open': .., 'high': .., 'low': .., 'close': ..}]
        signal_logic: funksiya — har bir candle uchun signalni tekshiradi
        """
        total_trades = 0
        wins = 0
        losses = 0

        for candle in price_data:
            signal = signal_logic(candle)
            if not signal:
                continue

            entry = candle['close']
            if signal['side'] == 'buy':
                result = 'win' if candle['high'] >= signal['tp'] else 'loss' if candle['low'] <= signal['sl'] else 'hold'
            else:
                result = 'win' if candle['low'] <= signal['tp'] else 'loss' if candle['high'] >= signal['sl'] else 'hold'

            if result == 'win':
                wins += 1
            if result == 'loss':
                losses += 1
            total_trades += 1

            self._write_result(candle['datetime'], signal['side'], entry, signal['sl'], signal['tp'], result)

        return {
            'total': total_trades,
            'wins': wins,
            'losses': losses,
            'winrate': round(100 * wins / total_trades, 2) if total_trades else 0
        }

    def _write_result(self, dt, side, entry, sl, tp, result):
        try:
            with open(self.results_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([dt, side, entry, sl, tp, result])
        except Exception as e:
            log_error(f"Trainer CSV error: {str(e)}")

if __name__ == "__main__":
    trainer = Trainer()
    prices = [
        {'datetime': datetime.utcnow(), 'open': 100, 'high': 105, 'low': 99, 'close': 101},
        {'datetime': datetime.utcnow(), 'open': 101, 'high': 102, 'low': 97, 'close': 100}
    ]

    def simple_logic(candle):
        return {'side': 'buy', 'entry': candle['close'], 'sl': candle['close'] - 2, 'tp': candle['close'] + 2}

    stats = trainer.backtest_signal(prices, simple_logic)
    print("Backtest:", stats)
