"""
RISK_MANAGER.PY
----------------
Pozitsiya hajmi, dinamik stop-loss/take-profit,
favqulodda to‘xtatish (balans kamayishi, ketma-ket yo‘qotish) boshqaruvi.
"""

import logging
import time

from logger import log_error, log_trade

class RiskManager:
    def __init__(self, initial_capital=10.0, max_risk_percent=2.0, emergency_stop_loss=20.0):
        """
        :param initial_capital: boshlang‘ich balans ($)
        :param max_risk_percent: har bir savdo uchun risk foizi
        :param emergency_stop_loss: umumiy balans kamayishi limiti (%)
        """
        self.current_capital = initial_capital
        self.max_risk_percent = max_risk_percent
        self.emergency_stop_loss = emergency_stop_loss
        self.consecutive_losses = 0
        self.max_consecutive_losses = 3

    def calculate_position_size(self, entry_price, stop_loss_price):
        """
        Pozitsiya hajmini hisoblash
        """
        risk_amount = self.current_capital * (self.max_risk_percent / 100)
        risk_per_unit = abs(entry_price - stop_loss_price)

        if risk_per_unit <= 0:
            log_error("❌ Xatolik: Entry va SL bir xil yoki noto‘g‘ri.")
            return 0

        qty = risk_amount / risk_per_unit
        logging.info(f"Pozitsiya hajmi: {qty:.6f} (Risk: ${risk_amount:.2f})")
        return qty

    def set_dynamic_sl_tp(self, entry_price, atr_value, side='buy', risk_reward_ratio=2):
        """
        Dinamik SL/TP ATR asosida
        """
        sl_distance = atr_value * 1.5

        if side.lower() == 'buy':
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + (sl_distance * risk_reward_ratio)
        else:
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - (sl_distance * risk_reward_ratio)

        logging.info(f"SL: {stop_loss:.2f} | TP: {take_profit:.2f}")
        return stop_loss, take_profit

    def check_emergency_stop(self, current_balance, last_trade_result=False):
        """
        Favqulodda to‘xtatish shartlarini tekshirish:
        - umumiy balans 20% dan ko‘p kamayganmi?
        - 3 ketma-ket yo‘qotish bo‘ldimi?
        """
        balance_drop = ((self.current_capital - current_balance) / self.current_capital) * 100

        if balance_drop >= self.emergency_stop_loss:
            log_error(f"❌ Favqulodda STOP! Balans {balance_drop:.2f}% ga kamaydi.")
            return True

        if last_trade_result is False:
            self.consecutive_losses += 1
            log_error(f"⚠️ Ketma-ket yo‘qotish: {self.consecutive_losses}")
        else:
            self.consecutive_losses = 0

        if self.consecutive_losses >= self.max_consecutive_losses:
            log_error("❌ Favqulodda STOP! 3 ketma-ket yo‘qotish.")
            return True

        return False

    def pause_trading(self, pause_minutes=60):
        """
        Savdoni vaqtincha to‘xtatish
        """
        log_error(f"⏸ Savdo {pause_minutes} daqiqa to‘xtatildi.")
        time.sleep(pause_minutes * 60)
        self.consecutive_losses = 0
