import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
import asyncio
from dataclasses import dataclass
from enum import Enum

from .utils import BurgutLogger, DataManager, PerformanceMonitor, PriceValidator

class TradeType(Enum):
    BUY = "BUY"
    SELL = "SELL"

class TradeStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"

@dataclass
class Trade:
    """Savdo ma'lumotlari"""
    trade_id: str
    timestamp: datetime
    trade_type: TradeType
    token_address: str
    token_symbol: str
    amount: float
    price: float
    slippage: float
    gas_fee: float
    status: TradeStatus
    profit_loss: float = 0.0
    close_timestamp: Optional[datetime] = None
    close_price: Optional[float] = None

@dataclass
class BacktestResult:
    """Backtest natijalari"""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_balance: float
    final_balance: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_profit: float
    total_loss: float
    net_profit: float
    max_drawdown: float
    sharpe_ratio: float
    avg_trade_duration: float
    trades: List[Trade]
    daily_pnl: Dict[str, float]
    metrics: Dict[str, Any]

class BacktestEngine:
    """Backtest engine sinfi"""
    
    def __init__(self, strategy, initial_balance: float = 10000.0):
        self.strategy = strategy
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.trades: List[Trade] = []
        self.open_positions: Dict[str, Trade] = {}
        self.daily_pnl: Dict[str, float] = {}
        
        self.logger = BurgutLogger().logger
        self.data_manager = DataManager()
        self.performance_monitor = PerformanceMonitor()
        
        # Backtest sozlamalari
        self.slippage_rate = 0.003  # 0.3% slippage
        self.gas_fee = 0.01  # $0.01 gas fee
        self.commission_rate = 0.0025  # 0.25% commission
        
    def load_historical_data(self, data_source: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Tarixiy ma'lumotlarni yuklash"""
        try:
            # CSV fayldan yuklash
            if data_source.endswith('.csv'):
                df = pd.read_csv(data_source)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
                
            # JSON fayldan yuklash
            elif data_source.endswith('.json'):
                data = self.data_manager.load_json(data_source)
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
                
            else:
                raise ValueError(f"Noto'g'ri ma'lumot manbai: {data_source}")
            
            # Majburiy ustunlarni tekshirish
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Majburiy ustunlar yo'q: {missing_columns}")
            
            self.logger.info(f"Tarixiy ma'lumot yuklandi: {len(df)} qator")
            return df.sort_values('timestamp')
            
        except Exception as e:
            self.logger.error(f"Tarixiy ma'lumot yuklashda xatolik: {e}")
            raise
    
    def calculate_slippage(self, expected_price: float, trade_type: TradeType) -> float:
        """Slippage hisoblash"""
        if trade_type == TradeType.BUY:
            # Buy da narx yuqori bo'ladi
            return expected_price * (1 + self.slippage_rate)
        else:
            # Sell da narx pastroq bo'ladi
            return expected_price * (1 - self.slippage_rate)
    
    def calculate_total_cost(self, amount: float, price: float) -> float:
        """Umumiy xarajat (commission va gas fee bilan)"""
        trade_value = amount * price
        commission = trade_value * self.commission_rate
        return trade_value + commission + self.gas_fee
    
    def execute_trade(self, signal: Dict[str, Any], current_data: pd.Series) -> bool:
        """Savdoni amalga oshirish"""
        try:
            trade_type = TradeType(signal['action'].upper())
            token_address = signal.get('token_address', 'ETH')
            token_symbol = signal.get('token_symbol', 'ETH')
            amount = signal.get('amount', 1.0)
            
            expected_price = current_data['close']
            actual_price = self.calculate_slippage(expected_price, trade_type)
            
            # Narx validatsiyasi
            if not PriceValidator.validate_price(actual_price):
                self.logger.warning(f"Noto'g'ri narx: {actual_price}")
                return False
            
            # Balans tekshiruvi
            total_cost = self.calculate_total_cost(amount, actual_price)
            if trade_type == TradeType.BUY and total_cost > self.current_balance:
                self.logger.warning(f"Balans yetarli emas: ${self.current_balance:.2f} < ${total_cost:.2f}")
                return False
            
            # Savdo yaratish
            trade = Trade(
                trade_id=f"trade_{len(self.trades) + 1}",
                timestamp=current_data['timestamp'],
                trade_type=trade_type,
                token_address=token_address,
                token_symbol=token_symbol,
                amount=amount,
                price=actual_price,
                slippage=PriceValidator.calculate_slippage(expected_price, actual_price),
                gas_fee=self.gas_fee,
                status=TradeStatus.OPEN
            )
            
            # Ochiq pozitsiyani saqlash
            if trade_type == TradeType.BUY:
                self.open_positions[token_address] = trade
                self.current_balance -= total_cost
                
            elif trade_type == TradeType.SELL:
                # Ochiq pozitsiyani yopish
                if token_address in self.open_positions:
                    buy_trade = self.open_positions[token_address]
                    
                    # Profit/Loss hisoblash
                    buy_cost = self.calculate_total_cost(buy_trade.amount, buy_trade.price)
                    sell_revenue = (amount * actual_price) - (amount * actual_price * self.commission_rate) - self.gas_fee
                    
                    profit_loss = sell_revenue - buy_cost
                    trade.profit_loss = profit_loss
                    trade.status = TradeStatus.CLOSED
                    
                    # Buy trade ni yangilash
                    buy_trade.close_timestamp = current_data['timestamp']
                    buy_trade.close_price = actual_price
                    buy_trade.profit_loss = profit_loss
                    buy_trade.status = TradeStatus.CLOSED
                    
                    # Balansni yangilash
                    self.current_balance += sell_revenue
                    
                    # Ochiq pozitsiyani olib tashlash
                    del self.open_positions[token_address]
                    
                    # Performance monitor
                    self.performance_monitor.record_trade(profit_loss > 0, profit_loss)
                    
                    self.logger.info(f"Savdo yakunlandi: {profit_loss:.2f} profit")
                else:
                    self.logger.warning(f"Ochiq pozitsiya topilmadi: {token_address}")
                    return False
            
            self.trades.append(trade)
            self.logger.info(f"Savdo amalga oshirildi: {trade_type.value} {amount} {token_symbol} @ ${actual_price:.6f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Savdo amalga oshirishda xatolik: {e}")
            return False
    
    def update_daily_pnl(self, date: str, pnl: float):
        """Kunlik P&L ni yangilash"""
        if date not in self.daily_pnl:
            self.daily_pnl[date] = 0.0
        self.daily_pnl[date] += pnl
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Backtest ko'rsatkichlarini hisoblash"""
        if not self.trades:
            return {}
        
        closed_trades = [t for t in self.trades if t.status == TradeStatus.CLOSED]
        
        if not closed_trades:
            return {"error": "Yopilgan savdolar yo'q"}
        
        # Asosiy metrikalar
        winning_trades = [t for t in closed_trades if t.profit_loss > 0]
        losing_trades = [t for t in closed_trades if t.profit_loss < 0]
        
        total_profit = sum(t.profit_loss for t in winning_trades)
        total_loss = abs(sum(t.profit_loss for t in losing_trades))
        net_profit = total_profit - total_loss
        
        win_rate = len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0
        
        # Savdo davomiyligi
        durations = []
        for trade in closed_trades:
            if trade.close_timestamp:
                duration = (trade.close_timestamp - trade.timestamp).total_seconds() / 3600  # soat
                durations.append(duration)
        
        avg_duration = np.mean(durations) if durations else 0
        
        # Maksimal pasayish (drawdown)
        balance_history = [self.initial_balance]
        running_balance = self.initial_balance
        
        for trade in closed_trades:
            running_balance += trade.profit_loss
            balance_history.append(running_balance)
        
        peak = self.initial_balance
        max_drawdown = 0
        
        for balance in balance_history:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        # Sharpe ratio (soddalashtirilgan)
        if len(closed_trades) > 1:
            returns = [t.profit_loss / self.initial_balance for t in closed_trades]
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            "total_trades": len(closed_trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "total_profit": total_profit,
            "total_loss": total_loss,
            "net_profit": net_profit,
            "net_profit_percentage": (net_profit / self.initial_balance) * 100,
            "max_drawdown": max_drawdown,
            "avg_trade_duration_hours": avg_duration,
            "sharpe_ratio": sharpe_ratio,
            "profit_factor": total_profit / total_loss if total_loss > 0 else float('inf'),
            "avg_winning_trade": total_profit / len(winning_trades) if winning_trades else 0,
            "avg_losing_trade": total_loss / len(losing_trades) if losing_trades else 0,
            "largest_winning_trade": max([t.profit_loss for t in winning_trades], default=0),
            "largest_losing_trade": min([t.profit_loss for t in losing_trades], default=0),
        }
    
    def run_backtest(self, data_source: str, start_date: datetime, end_date: datetime, 
                     strategy_params: Dict[str, Any] = None) -> BacktestResult:
        """Backtest ni ishga tushirish"""
        try:
            self.logger.info(f"Backtest boshlandi: {start_date} dan {end_date} gacha")
            
            # Ma'lumotlarni yuklash
            df = self.load_historical_data(data_source, start_date, end_date)
            
            if df.empty:
                raise ValueError("Ma'lumotlar bo'sh")
            
            # Strategiya parametrlarini sozlash
            if strategy_params:
                self.strategy.update_parameters(strategy_params)
            
            # Backtest loop
            for idx, row in df.iterrows():
                try:
                    # Strategiya signalini olish
                    signal = self.strategy.generate_signal(row, df.iloc[:idx+1])
                    
                    # Signal mavjud bo'lsa, savdo amalga oshirish
                    if signal and signal.get('action') in ['BUY', 'SELL']:
                        self.execute_trade(signal, row)
                    
                    # Kunlik P&L ni yangilash
                    date_str = row['timestamp'].strftime('%Y-%m-%d')
                    if self.trades:
                        last_trade = self.trades[-1]
                        if last_trade.status == TradeStatus.CLOSED:
                            self.update_daily_pnl(date_str, last_trade.profit_loss)
                    
                except Exception as e:
                    self.logger.error(f"Backtest iteratsiyasida xatolik: {e}")
                    continue
            
            # Ochiq pozitsiyalarni yopish
            for token_address, trade in self.open_positions.items():
                last_price = df.iloc[-1]['close']
                close_price = self.calculate_slippage(last_price, TradeType.SELL)
                
                buy_cost = self.calculate_total_cost(trade.amount, trade.price)
                sell_revenue = (trade.amount * close_price) - (trade.amount * close_price * self.commission_rate) - self.gas_fee
                
                profit_loss = sell_revenue - buy_cost
                trade.profit_loss = profit_loss
                trade.close_timestamp = df.iloc[-1]['timestamp']
                trade.close_price = close_price
                trade.status = TradeStatus.CLOSED
                
                self.current_balance += sell_revenue
                self.performance_monitor.record_trade(profit_loss > 0, profit_loss)
            
            # Natijalarni hisoblash
            metrics = self.calculate_metrics()
            
            # Backtest natijasi
            result = BacktestResult(
                strategy_name=self.strategy.__class__.__name__,
                start_date=start_date,
                end_date=end_date,
                initial_balance=self.initial_balance,
                final_balance=self.current_balance,
                total_trades=len([t for t in self.trades if t.status == TradeStatus.CLOSED]),
                winning_trades=metrics.get('winning_trades', 0),
                losing_trades=metrics.get('losing_trades', 0),
                win_rate=metrics.get('win_rate', 0),
                total_profit=metrics.get('total_profit', 0),
                total_loss=metrics.get('total_loss', 0),
                net_profit=metrics.get('net_profit', 0),
                max_drawdown=metrics.get('max_drawdown', 0),
                sharpe_ratio=metrics.get('sharpe_ratio', 0),
                avg_trade_duration=metrics.get('avg_trade_duration_hours', 0),
                trades=self.trades,
                daily_pnl=self.daily_pnl,
                metrics=metrics
            )
            
            self.logger.info(f"Backtest yakunlandi: {result.net_profit:.2f} net profit")
            return result
            
        except Exception as e:
            self.logger.error(f"Backtest da xatolik: {e}")
            raise
    
    def save_results(self, result: BacktestResult, filename: str = None) -> str:
        """Backtest natijalarini saqlash"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_{result.strategy_name}_{timestamp}.json"
        
        # Natijalarni lug'atga aylantirish
        result_dict = {
            "strategy_name": result.strategy_name,
            "start_date": result.start_date.isoformat(),
            "end_date": result.end_date.isoformat(),
            "initial_balance": result.initial_balance,
            "final_balance": result.final_balance,
            "total_trades": result.total_trades,
            "winning_trades": result.winning_trades,
            "losing_trades": result.losing_trades,
            "win_rate": result.win
