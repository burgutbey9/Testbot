"""
Advanced Backtesting Engine
Kengaytirilgan backtest engine slippage, market impact, latency simulatsiyasi bilan
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"

@dataclass
class BacktestTrade:
    """Backtest trade ma'lumotlari"""
    timestamp: datetime
    symbol: str
    side: OrderSide
    size: float
    price: float
    order_type: OrderType
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
@dataclass
class ExecutedTrade:
    """Executed trade ma'lumotlari"""
    original_trade: BacktestTrade
    executed_price: float
    executed_size: float
    execution_time: datetime
    slippage: float
    market_impact: float
    latency_ms: int
    pnl: float
    fees: float
    
@dataclass
class BacktestResult:
    """Backtest natijalari"""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    volatility: float
    calmar_ratio: float
    sortino_ratio: float
    
class SlippageModel:
    """Slippage simulation model"""
    
    def __init__(self, base_slippage: float = 0.0005, volatility_multiplier: float = 2.0):
        self.base_slippage = base_slippage  # 0.05% base slippage
        self.volatility_multiplier = volatility_multiplier
        
    def calculate_slippage(self, trade: BacktestTrade, market_data: Dict) -> float:
        """Slippage hisoblash"""
        # Volatility based slippage
        volatility = market_data.get('volatility', 0.02)
        volume = market_data.get('volume', 1000000)
        
        # Base slippage
        slippage = self.base_slippage
        
        # Volatility effect
        slippage += volatility * self.volatility_multiplier
        
        # Volume effect (kam volume = ko'p slippage)
        volume_effect = max(0, 1 - volume / 1000000) * 0.002
        slippage += volume_effect
        
        # Order size effect
        size_effect = min(trade.size / 100000, 0.01)  # Max 1% additional slippage
        slippage += size_effect
        
        # Market order vs limit order
        if trade.order_type == OrderType.MARKET:
            slippage *= 1.5  # Market order ko'proq slippage
            
        return slippage
        
class MarketImpactModel:
    """Market impact simulation model"""
    
    def __init__(self, impact_coefficient: float = 0.1):
        self.impact_coefficient = impact_coefficient
        
    def calculate_market_impact(self, trade: BacktestTrade, market_data: Dict) -> float:
        """Market impact hisoblash"""
        avg_volume = market_data.get('avg_volume', 1000000)
        
        # Trade size relative to average volume
        volume_ratio = trade.size / avg_volume
        
        # Square root impact model
        impact = self.impact_coefficient * np.sqrt(volume_ratio)
        
        # Cap maximum impact
        impact = min(impact, 0.05)  # Max 5% impact
        
        return impact
        
class LatencySimulator:
    """Latency simulation"""
    
    def __init__(self, base_latency: float = 50, max_latency: float = 500):
        self.base_latency = base_latency  # 50ms base latency
        self.max_latency = max_latency   # 500ms max latency
        
    def get_execution_latency(self, market_conditions: Dict) -> int:
        """Execution latency simulation"""
        # Base latency
        latency = self.base_latency
        
        # Volatility effect
        volatility = market_conditions.get('volatility', 0.02)
        latency += volatility * 1000  # High volatility = higher latency
        
        # Random network delay
        random_delay = np.random.exponential(20)  # Exponential distribution
        latency += random_delay
        
        # Cap latency
        latency = min(latency, self.max_latency)
        
        return int(latency)
        
class RealisticPriceSimulator:
    """Realistic price movement simulation"""
    
    def __init__(self):
        self.price_cache = {}
        
    def get_realistic_price(self, timestamp: datetime, symbol: str, 
                          base_price: float, latency_ms: int) -> float:
        """Realistic price with latency effect"""
        # Simulate price movement during latency
        latency_seconds = latency_ms / 1000.0
        
        # Random walk during latency
        price_change = np.random.normal(0, 0.001 * np.sqrt(latency_seconds))
        
        # Adjust price
        new_price = base_price * (1 + price_change)
        
        return new_price
        
class AdvancedBacktester:
    """Advanced backtesting engine"""
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 commission_rate: float = 0.001,
                 slippage_model: Optional[SlippageModel] = None,
                 market_impact_model: Optional[MarketImpactModel] = None,
                 latency_simulator: Optional[LatencySimulator] = None):
        
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_model = slippage_model or SlippageModel()
        self.market_impact_model = market_impact_model or MarketImpactModel()
        self.latency_simulator = latency_simulator or LatencySimulator()
        self.price_simulator = RealisticPriceSimulator()
        
        # State variables
        self.current_capital = initial_capital
        self.positions = {}
        self.executed_trades = []
        self.equity_curve = []
        self.logger = logging.getLogger(__name__)
        
    def run_backtest(self, trades: List[BacktestTrade], 
                    market_data: pd.DataFrame) -> BacktestResult:
        """Backtest ishga tushirish"""
        self.logger.info(f"Starting backtest with {len(trades)} trades")
        
        # Reset state
        self.current_capital = self.initial_capital
        self.positions = {}
        self.executed_trades = []
        self.equity_curve = []
        
        # Process trades
        for trade in trades:
            executed_trade = self._execute_trade(trade, market_data)
            if executed_trade:
                self.executed_trades.append(executed_trade)
                self._update_equity_curve(executed_trade)
                
        # Calculate final results
        return self._calculate_results()
        
    def _execute_trade(self, trade: BacktestTrade, market_data: pd.DataFrame) -> Optional[ExecutedTrade]:
        """Trade'ni execute qilish"""
        try:
            # Market data for this timestamp
            market_row = self._get_market_data(trade.timestamp, market_data)
            if market_row is None:
                return None
                
            market_conditions = {
                'volatility': market_row.get('volatility', 0.02),
                'volume': market_row.get('volume', 1000000),
                'avg_volume': market_row.get('avg_volume', 1000000),
                'bid_ask_spread': market_row.get('spread', 0.001)
            }
            
            # Calculate slippage
            slippage = self.slippage_model.calculate_slippage(trade, market_conditions)
            
            # Calculate market impact
            market_impact = self.market_impact_model.calculate_market_impact(trade, market_conditions)
            
            # Calculate latency
            latency_ms = self.latency_simulator.get_execution_latency(market_conditions)
            
            # Get realistic execution price
            base_price = trade.price
            if trade.side == OrderSide.BUY:
                executed_price = base_price * (1 + slippage + market_impact)
            else:
                executed_price = base_price * (1 - slippage - market_impact)
                
            # Apply latency effect
            executed_price = self.price_simulator.get_realistic_price(
                trade.timestamp, trade.symbol, executed_price, latency_ms
            )
            
            # Calculate fees
            fees = trade.size * executed_price * self.commission_rate
            
            # Calculate PnL
            pnl = self._calculate_pnl(trade, executed_price, fees)
            
            # Update capital
            self.current_capital += pnl
            
            # Create executed trade
            executed_trade = ExecutedTrade(
                original_trade=trade,
                executed_price=executed_price,
                executed_size=trade.size,
                execution_time=trade.timestamp + timedelta(milliseconds=latency_ms),
                slippage=slippage,
                market_impact=market_impact,
                latency_ms=latency_ms,
                pnl=pnl,
                fees=fees
            )
            
            return executed_trade
            
        except Exception as e:
            self.logger.error(f"Trade execution error: {e}")
            return None
            
    def _get_market_data(self, timestamp: datetime, market_data: pd.DataFrame) -> Optional[Dict]:
        """Market data olish"""
        try:
            # Find closest timestamp
            market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])
            closest_idx = (market_data['timestamp'] - timestamp).abs().idxmin()
            return market_data.iloc[closest_idx].to_dict()
        except:
            return None
            
    def _calculate_pnl(self, trade: BacktestTrade, executed_price: float, fees: float) -> float:
        """PnL hisoblash"""
        if trade.symbol not in self.positions:
            self.positions[trade.symbol] = {'size': 0, 'avg_price': 0}
            
        position = self.positions[trade.symbol]
        
        if trade.side == OrderSide.BUY:
            # Long position
            if position['size'] >= 0:
                # Increasing long position
                new_size = position['size'] + trade.size
                new_avg_price = ((position['avg_price'] * position['size']) + 
                               (executed_price * trade.size)) / new_size
                position['size'] = new_size
                position['avg_price'] = new_avg_price
                return -fees  # Only fees for opening position
            else:
                # Closing short position
                close_size = min(abs(position['size']), trade.size)
                pnl = close_size * (position['avg_price'] - executed_price)
                position['size'] += close_size
                return pnl - fees
        else:
            # Short position
            if position['size'] <= 0:
                # Increasing short position
                new_size = position['size'] - trade.size
                new_avg_price = ((position['avg_price'] * abs(position['size'])) + 
                               (executed_price * trade.size)) / abs(new_size)
                position['size'] = new_size
                position['avg_price'] = new_avg_price
                return -fees  # Only fees for opening position
            else:
                # Closing long position
                close_size = min(position['size'], trade.size)
                pnl = close_size * (executed_price - position['avg_price'])
                position['size'] -= close_size
                return pnl - fees
                
    def _update_equity_curve(self, executed_trade: ExecutedTrade):
        """Equity curve yangilash"""
        self.equity_curve.append({
            'timestamp': executed_trade.execution_time,
            'equity': self.current_capital,
            'pnl': executed_trade.pnl,
            'trade_id': len(self.executed_trades)
        })
        
    def _calculate_results(self) -> BacktestResult:
        """Final results hisoblash"""
        if not self.executed_trades:
            return BacktestResult(
                total_return=0, sharpe_ratio=0, max_drawdown=0, win_rate=0,
                profit_factor=0, total_trades=0, avg_trade_duration=0,
                volatility=0, calmar_ratio=0, sortino_ratio=0
            )
            
        # PnL series
        pnls = [trade.pnl for trade in self.executed_trades]
        
        # Total return
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        
        # Sharpe ratio
        returns = pd.Series(pnls)
        sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Max drawdown
        equity_series = pd.Series([e['equity'] for e in self.equity_curve])
        max_equity = equity_series.expanding().max()
        drawdown = (equity_series - max_equity) / max_equity
        max_drawdown = abs(drawdown.min())
        
        # Win rate
        winning_trades = [pnl for pnl in pnls if pnl > 0]
        win_rate = len(winning_trades) / len(pnls) if pnls else 0
        
        # Profit factor
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum([pnl for pnl in pnls if pnl < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average trade duration
        if len(self.executed_trades) > 1:
            durations = []
            for i in range(1, len(self.executed_trades)):
                duration = (self.executed_trades[i].execution_time - 
                          self.executed_trades[i-1].execution_time).total_seconds()
                durations.append(duration)
            avg_trade_duration = np.mean(durations) if durations else 0
        else:
            avg_trade_duration = 0
            
        # Volatility
        volatility = returns.std() * np.sqrt(365) if len(returns) > 1 else 0
        
        # Calmar ratio
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
        
        # Sortino ratio
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std() if len(negative_returns) > 0 else 0
        sortino_ratio = returns.mean() / downside_deviation if downside_deviation > 0 else 0
        
        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=
