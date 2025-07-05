"""
Position Manager - Pozitsiyalarni boshqarish va risk managementni nazorat qilish
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import math
import json

class PositionStatus(Enum):
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    FAILED = "failed"

class PositionType(Enum):
    LONG = "long"
    SHORT = "short"

@dataclass
class Position:
    symbol: str
    position_type: PositionType
    size: float
    entry_price: float
    current_price: float
    timestamp: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: PositionStatus = PositionStatus.OPEN
    trade_id: str = ""
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    max_profit: float = 0.0
    max_loss: float = 0.0
    hold_time: int = 0  # seconds
    
    def update_pnl(self, current_price: float):
        """PnL ni yangilash"""
        self.current_price = current_price
        
        if self.position_type == PositionType.LONG:
            self.unrealized_pnl = (current_price - self.entry_price) * self.size
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.size
        
        # Max profit/loss tracking
        if self.unrealized_pnl > self.max_profit:
            self.max_profit = self.unrealized_pnl
        elif self.unrealized_pnl < self.max_loss:
            self.max_loss = self.unrealized_pnl
            
        self.hold_time = int((datetime.now() - self.timestamp).total_seconds())
    
    def get_pnl_percentage(self) -> float:
        """PnL ni foizda qaytarish"""
        if self.entry_price == 0:
            return 0.0
        return (self.unrealized_pnl / (self.entry_price * self.size)) * 100

class PositionManager:
    def __init__(self, config: Dict):
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.position_history: List[Position] = []
        self.logger = logging.getLogger(__name__)
        
        # Risk parameters
        self.max_positions = config.get('max_positions', 5)
        self.max_position_size = config.get('max_position_size', 1000)
        self.max_total_exposure = config.get('max_total_exposure', 5000)
        self.max_drawdown = config.get('max_drawdown', 0.05)  # 5%
        self.position_timeout = config.get('position_timeout', 3600)  # 1 hour
        
        # Performance tracking
        self.total_realized_pnl = 0.0
        self.total_fees = 0.0
        self.win_count = 0
        self.loss_count = 0
        self.start_balance = config.get('start_balance', 10000)
        
        # Emergency stop
        self.emergency_stop_active = False
        self.emergency_stop_reason = ""
        
        # Correlation tracking
        self.correlation_matrix = {}
        
    async def open_position(self, symbol: str, position_type: PositionType, 
                          size: float, entry_price: float, 
                          stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None) -> Tuple[bool, str]:
        """Yangi pozitsiya ochish"""
        try:
            # Pre-checks
            if self.emergency_stop_active:
                return False, f"Emergency stop active: {self.emergency_stop_reason}"
            
            # Risk checks
            risk_check, risk_msg = self._check_risk_limits(symbol, size, entry_price)
            if not risk_check:
                return False, risk_msg
            
            # Position size validation
            if size > self.max_position_size:
                return False, f"Position size {size} exceeds max limit {self.max_position_size}"
            
            # Correlation check
            if not self._check_correlation_risk(symbol, position_type):
                return False, "High correlation risk detected"
            
            # Create position
            trade_id = f"{symbol}_{position_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            position = Position(
                symbol=symbol,
                position_type=position_type,
                size=size,
                entry_price=entry_price,
                current_price=entry_price,
                timestamp=datetime.now(),
                stop_loss=stop_loss,
                take_profit=take_profit,
                trade_id=trade_id
            )
            
            self.positions[trade_id] = position
            
            self.logger.info(f"Position opened: {trade_id} - {symbol} {position_type.value} "
                           f"Size: {size} Price: {entry_price}")
            
            return True, trade_id
            
        except Exception as e:
            self.logger.error(f"Error opening position: {str(e)}")
            return False, str(e)
    
    async def close_position(self, trade_id: str, exit_price: float, 
                           reason: str = "manual") -> Tuple[bool, str]:
        """Pozitsiyani yopish"""
        try:
            if trade_id not in self.positions:
                return False, f"Position {trade_id} not found"
            
            position = self.positions[trade_id]
            
            if position.status != PositionStatus.OPEN:
                return False, f"Position {trade_id} is not open"
            
            # Final PnL calculation
            position.update_pnl(exit_price)
            position.realized_pnl = position.unrealized_pnl
            position.status = PositionStatus.CLOSED
            
            # Update statistics
            self.total_realized_pnl += position.realized_pnl
            
            if position.realized_pnl > 0:
                self.win_count += 1
            else:
                self.loss_count += 1
            
            # Move to history
            self.position_history.append(position)
            del self.positions[trade_id]
            
            self.logger.info(f"Position closed: {trade_id} - PnL: {position.realized_pnl:.2f} "
                           f"Reason: {reason}")
            
            # Emergency stop check
            await self._check_emergency_stop()
            
            return True, f"Position closed with PnL: {position.realized_pnl:.2f}"
            
        except Exception as e:
            self.logger.error(f"Error closing position: {str(e)}")
            return False, str(e)
    
    async def update_positions(self, price_data: Dict[str, float]):
        """Barcha pozitsiyalarni yangilash"""
        try:
            for trade_id, position in self.positions.items():
                if position.symbol in price_data:
                    current_price = price_data[position.symbol]
                    position.update_pnl(current_price)
                    
                    # Check stop loss / take profit
                    await self._check_exit_conditions(trade_id, position)
                    
                    # Check timeout
                    await self._check_position_timeout(trade_id, position)
            
            # Update correlation matrix
            self._update_correlation_matrix(price_data)
            
        except Exception as e:
            self.logger.error(f"Error updating positions: {str(e)}")
    
    async def _check_exit_conditions(self, trade_id: str, position: Position):
        """Stop loss va take profit tekshirish"""
        try:
            should_close = False
            reason = ""
            
            if position.stop_loss and position.position_type == PositionType.LONG:
                if position.current_price <= position.stop_loss:
                    should_close = True
                    reason = "stop_loss"
            elif position.stop_loss and position.position_type == PositionType.SHORT:
                if position.current_price >= position.stop_loss:
                    should_close = True
                    reason = "stop_loss"
            
            if position.take_profit and position.position_type == PositionType.LONG:
                if position.current_price >= position.take_profit:
                    should_close = True
                    reason = "take_profit"
            elif position.take_profit and position.position_type == PositionType.SHORT:
                if position.current_price <= position.take_profit:
                    should_close = True
                    reason = "take_profit"
            
            if should_close:
                await self.close_position(trade_id, position.current_price, reason)
                
        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {str(e)}")
    
    async def _check_position_timeout(self, trade_id: str, position: Position):
        """Pozitsiya timeout tekshirish"""
        try:
            if position.hold_time > self.position_timeout:
                await self.close_position(trade_id, position.current_price, "timeout")
                
        except Exception as e:
            self.logger.error(f"Error checking position timeout: {str(e)}")
    
    def _check_risk_limits(self, symbol: str, size: float, price: float) -> Tuple[bool, str]:
        """Risk limitlarini tekshirish"""
        try:
            # Max positions check
            if len(self.positions) >= self.max_positions:
                return False, f"Max positions limit reached: {self.max_positions}"
            
            # Total exposure check
            current_exposure = sum(pos.size * pos.current_price for pos in self.positions.values())
            new_exposure = current_exposure + (size * price)
            
            if new_exposure > self.max_total_exposure:
                return False, f"Total exposure limit exceeded: {new_exposure} > {self.max_total_exposure}"
            
            # Drawdown check
            current_drawdown = self.get_current_drawdown()
            if current_drawdown > self.max_drawdown:
                return False, f"Max drawdown exceeded: {current_drawdown:.2%} > {self.max_drawdown:.2%}"
            
            return True, "Risk checks passed"
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {str(e)}")
            return False, str(e)
    
    def _check_correlation_risk(self, symbol: str, position_type: PositionType) -> bool:
        """Korrelyatsiya riskini tekshirish"""
        try:
            if not self.correlation_matrix:
                return True
            
            # Check correlation with existing positions
            for trade_id, position in self.positions.items():
                if position.symbol == symbol:
                    continue
                
                correlation = self.correlation_matrix.get(f"{symbol}_{position.symbol}", 0)
                
                # High correlation risk
                if correlation > 0.8 and position.position_type == position_type:
                    return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking correlation risk: {str(e)}")
            return True
    
    def _update_correlation_matrix(self, price_data: Dict[str, float]):
        """Korrelyatsiya matrisasini yangilash"""
        try:
            # Simple correlation calculation (moving window)
            symbols = list(price_data.keys())
            
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols[i+1:], i+1):
                    key = f"{symbol1}_{symbol2}"
                    
                    # Simplified correlation (should be improved with proper calculation)
                    price1 = price_data[symbol1]
                    price2 = price_data[symbol2]
                    
                    # Store for correlation analysis
                    if key not in self.correlation_matrix:
                        self.correlation_matrix[key] = []
                    
                    self.correlation_matrix[key].append((price1, price2))
                    
                    # Keep only last 100 data points
                    if len(self.correlation_matrix[key]) > 100:
                        self.correlation_matrix[key] = self.correlation_matrix[key][-100:]
            
        except Exception as e:
            self.logger.error(f"Error updating correlation matrix: {str(e)}")
    
    async def _check_emergency_stop(self):
        """Emergency stop holatini tekshirish"""
        try:
            current_drawdown = self.get_current_drawdown()
            
            # Large drawdown
            if current_drawdown > self.max_drawdown * 2:
                self.emergency_stop_active = True
                self.emergency_stop_reason = f"Large drawdown: {current_drawdown:.2%}"
                
                # Close all positions
                for trade_id, position in list(self.positions.items()):
                    await self.close_position(trade_id, position.current_price, "emergency_stop")
                
                self.logger.critical(f"Emergency stop activated: {self.emergency_stop_reason}")
            
            # Consecutive losses
            recent_trades = self.position_history[-10:] if len(self.position_history) >= 10 else self.position_history
            if len(recent_trades) >= 5:
                consecutive_losses = sum(1 for trade in recent_trades if trade.realized_pnl < 0)
                if consecutive_losses >= 5:
                    self.emergency_stop_active = True
                    self.emergency_stop_reason = f"5 consecutive losses detected"
                    
        except Exception as e:
            self.logger.error(f"Error checking emergency stop: {str(e)}")
    
    def get_current_drawdown(self) -> float:
        """Joriy drawdown hisobla"""
        try:
            if self.start_balance == 0:
                return 0.0
            
            current_balance = self.start_balance + self.total_realized_pnl
            current_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
            total_balance = current_balance + current_unrealized
            
            peak_balance = max(self.start_balance, total_balance)
            drawdown = (peak_balance - total_balance) / peak_balance
            
            return max(0, drawdown)
            
        except Exception as e:
            self.logger.error(f"Error calculating drawdown: {str(e)}")
            return 0.0
    
    def get_portfolio_stats(self) -> Dict:
        """Portfolio statistikalari"""
        try:
            total_trades = len(self.position_history)
            win_rate = (self.win_count / total_trades * 100) if total_trades > 0 else 0
            
            current_balance = self.start_balance + self.total_realized_pnl
            current_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
            total_balance = current_balance + current_unrealized
            
            return {
                'total_balance': total_balance,
                'realized_pnl': self.total_realized_pnl,
                'unrealized_pnl': current_unrealized,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'win_count': self.win_count,
                'loss_count': self.loss_count,
                'open_positions': len(self.positions),
                'max_drawdown': self.get_current_drawdown(),
                'total_fees': self.total_fees,
                'roi': ((total_balance - self.start_balance) / self.start_balance * 100) if self.start_balance > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio stats: {str(e)}")
            return {}
    
    def get_position_details(self, trade_id: str) -> Optional[Dict]:
        """Pozitsiya tafsilotlari"""
        try:
            if trade_id in self.positions:
                position = self.positions[trade_id]
                return {
                    'trade_id': trade_id,
                    'symbol': position.symbol,
                    'type': position.position_type.value,
                    'size': position.size,
                    'entry_price': position.entry_price,
                    'current_price': position.current_price,
                    'unrealized_pnl': position.unrealized_pnl,
                    'pnl_percentage': position.get_pnl_percentage(),
                    'hold_time': position.hold_time,
                    'max_profit': position.max_profit,
                    'max_loss': position.max_loss,
                    'stop_loss': position.stop_loss,
                    'take_profit': position.take_profit,
                    'status': position.status.value
                }
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting position details: {str(e)}")
            return None
    
    def close_all_positions(self, reason: str = "manual_close_all"):
        """Barcha pozitsiyalarni yopish"""
        try:
            closed_count = 0
            for trade_id, position in list(self.positions.items()):
                success, msg = asyncio.run(self.close_position(trade_id, position.current_price, reason))
                if success:
                    closed_count += 1
            
            self.logger.info(f"Closed {closed_count} positions. Reason: {reason}")
            return closed_count
            
        except Exception as e:
            self.logger.error(f"Error closing all positions: {str(e)}")
            return 0
    
    def reset_emergency_stop(self):
        """Emergency stop ni reset qilish"""
        self.emergency_stop_active = False
        self.emergency_stop_reason = ""
        self.logger.info("Emergency stop reset")
    
    def export_positions_to_json(self) -> str:
        """Pozitsiyalarni JSON formatda eksport qilish"""
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'open_positions': [],
                'position_history': [],
                'portfolio_stats': self.get_portfolio_stats()
            }
            
            # Open positions
            for trade_id, position in self.positions.items():
                export_data['open_positions'].append({
                    'trade_id': trade_id,
                    'symbol': position.symbol,
                    'type': position.position_type.value,
                    'size': position.size,
                    'entry_price': position.entry_price,
                    'current_price': position.current_price,
                    'unrealized_pnl': position.unrealized_pnl,
                    'timestamp': position.timestamp.isoformat()
                })
            
            # Position history (last 50)
            for position in self.position_history[-50:]:
                export_data['position_history'].append({
                    'trade_id': position.trade_id,
                    'symbol': position.symbol,
                    'type': position.position_type.value,
                    'size': position.size,
                    'entry_price': position.entry_price,
                    'exit_price': position.current_price,
                    'realized_pnl': position.realized_pnl,
                    'hold_time': position.hold_time,
                    'timestamp': position.timestamp.isoformat()
                })
            
            return json.dumps(export_data, indent=2)
            
        except Exception as e:
            self.logger.error(f"Error exporting positions: {str(e)}")
            return "{}"
