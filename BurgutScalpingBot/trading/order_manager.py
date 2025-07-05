# trading/order_manager.py
import asyncio
import time
import logging
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from decimal import Decimal
import json
from datetime import datetime, timedelta

from utils.advanced_logger import TradingLogger
from utils.rate_limiter import RateLimiter
from utils.health_checker import HealthChecker
from config.validators import ConfigValidator


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Order data structure"""
    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: Decimal = Decimal('0')
    avg_fill_price: Optional[Decimal] = None
    timestamp: datetime = None
    client_order_id: Optional[str] = None
    time_in_force: str = "GTC"  # Good Till Cancelled
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active"""
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]
    
    @property
    def remaining_quantity(self) -> Decimal:
        """Get remaining quantity to fill"""
        return self.quantity - self.filled_quantity
    
    @property
    def fill_percentage(self) -> float:
        """Get fill percentage"""
        if self.quantity == 0:
            return 0.0
        return float(self.filled_quantity / self.quantity * 100)


class OrderManager:
    """Advanced order management with fallback mechanisms"""
    
    def __init__(self, config: Dict, exchange_client=None):
        self.config = config
        self.exchange_client = exchange_client
        self.logger = TradingLogger()
        self.rate_limiter = RateLimiter(
            max_calls=config.get('max_orders_per_minute', 30),
            window=60
        )
        self.health_checker = HealthChecker()
        
        # Order tracking
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.failed_orders: List[Dict] = []
        
        # Risk management
        self.max_position_size = Decimal(str(config.get('max_position_size', 1000)))
        self.max_daily_trades = config.get('max_daily_trades', 100)
        self.daily_trade_count = 0
        self.last_trade_date = datetime.now().date()
        
        # Slippage protection
        self.max_slippage = Decimal(str(config.get('max_slippage', 0.001)))  # 0.1%
        
        # Order timeout settings
        self.order_timeout = config.get('order_timeout_seconds', 30)
        self.max_retries = config.get('max_order_retries', 3)
        
        # Circuit breaker
        self.circuit_breaker_errors = 0
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_reset_time = 300  # 5 minutes
        self.circuit_breaker_triggered = False
        self.circuit_breaker_time = None
        
        self.logger.log_system_event("OrderManager initialized", {
            'max_position_size': float(self.max_position_size),
            'max_daily_trades': self.max_daily_trades,
            'order_timeout': self.order_timeout
        })
    
    async def place_order(self, order: Order) -> Tuple[bool, str]:
        """Place order with comprehensive error handling"""
        try:
            # Circuit breaker check
            if self.circuit_breaker_triggered:
                if time.time() - self.circuit_breaker_time > self.circuit_breaker_reset_time:
                    self.reset_circuit_breaker()
                else:
                    return False, "Circuit breaker is active - order rejected"
            
            # Rate limiting check
            allowed, message = self.rate_limiter.check_rate_limit("place_order")
            if not allowed:
                self.logger.log_error(f"Rate limit exceeded for place_order: {message}")
                return False, message
            
            # Pre-order validation
            validation_result = await self.validate_order(order)
            if not validation_result[0]:
                return False, validation_result[1]
            
            # Check daily trade limit
            if not self.check_daily_trade_limit():
                return False, "Daily trade limit exceeded"
            
            # Submit order with retry logic
            success, result = await self.submit_order_with_retry(order)
            
            if success:
                self.active_orders[order.id] = order
                self.daily_trade_count += 1
                self.logger.log_trade_decision(
                    signal={'type': 'order_placed'},
                    decision=f"Order {order.id} placed successfully",
                    reasoning=f"Order type: {order.type.value}, Side: {order.side.value}, Quantity: {order.quantity}"
                )
                
                # Start order monitoring
                asyncio.create_task(self.monitor_order(order))
                return True, f"Order {order.id} placed successfully"
            else:
                self.circuit_breaker_errors += 1
                if self.circuit_breaker_errors >= self.circuit_breaker_threshold:
                    self.trigger_circuit_breaker()
                
                self.failed_orders.append({
                    'order': order.__dict__,
                    'error': result,
                    'timestamp': datetime.now()
                })
                return False, result
                
        except Exception as e:
            self.logger.log_error(f"Critical error in place_order: {str(e)}")
            self.circuit_breaker_errors += 1
            if self.circuit_breaker_errors >= self.circuit_breaker_threshold:
                self.trigger_circuit_breaker()
            return False, f"Critical error: {str(e)}"
    
    async def validate_order(self, order: Order) -> Tuple[bool, str]:
        """Comprehensive order validation"""
        try:
            # Basic validation
            if order.quantity <= 0:
                return False, "Order quantity must be positive"
            
            if order.type == OrderType.LIMIT and order.price is None:
                return False, "Limit order requires price"
            
            if order.type in [OrderType.STOP_LOSS, OrderType.TRAILING_STOP] and order.stop_price is None:
                return False, "Stop order requires stop price"
            
            # Position size validation
            current_position = await self.get_current_position(order.symbol)
            if order.side == OrderSide.BUY:
                new_position = current_position + order.quantity
            else:
                new_position = current_position - order.quantity
            
            if abs(new_position) > self.max_position_size:
                return False, f"Order would exceed maximum position size: {self.max_position_size}"
            
            # Market price validation for limit orders
            if order.type == OrderType.LIMIT:
                current_price = await self.get_current_price(order.symbol)
                if current_price is None:
                    return False, "Unable to get current market price"
                
                # Check for reasonable price (prevent fat finger errors)
                price_diff = abs(order.price - current_price) / current_price
                if price_diff > 0.1:  # 10% difference
                    return False, f"Order price deviates too much from market price: {price_diff:.2%}"
            
            # Symbol validation
            if not await self.validate_symbol(order.symbol):
                return False, f"Invalid or unsupported symbol: {order.symbol}"
            
            return True, "Order validation passed"
            
        except Exception as e:
            self.logger.log_error(f"Error in order validation: {str(e)}")
            return False, f"Validation error: {str(e)}"
    
    async def submit_order_with_retry(self, order: Order) -> Tuple[bool, str]:
        """Submit order with exponential backoff retry"""
        for attempt in range(self.max_retries):
            try:
                # Mock exchange API call - replace with actual exchange client
                if self.exchange_client:
                    response = await self.exchange_client.place_order(
                        symbol=order.symbol,
                        side=order.side.value,
                        type=order.type.value,
                        quantity=float(order.quantity),
                        price=float(order.price) if order.price else None,
                        stop_price=float(order.stop_price) if order.stop_price else None,
                        time_in_force=order.time_in_force
                    )
                    
                    if response.get('status') == 'success':
                        order.status = OrderStatus.SUBMITTED
                        order.client_order_id = response.get('client_order_id')
                        return True, "Order submitted successfully"
                    else:
                        error_msg = response.get('error', 'Unknown error')
                        if attempt < self.max_retries - 1:
                            wait_time = 2 ** attempt  # Exponential backoff
                            await asyncio.sleep(wait_time)
                            continue
                        return False, error_msg
                else:
                    # Simulation mode
                    await asyncio.sleep(0.1)  # Simulate network delay
                    order.status = OrderStatus.SUBMITTED
                    return True, "Order submitted (simulation mode)"
                    
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                return False, f"Failed after {self.max_retries} attempts: {str(e)}"
        
        return False, "Max retries exceeded"
    
    async def monitor_order(self, order: Order):
        """Monitor order status and handle timeouts"""
        start_time = time.time()
        
        while order.is_active and time.time() - start_time < self.order_timeout:
            try:
                # Check order status
                updated_order = await self.get_order_status(order.id)
                if updated_order:
                    order.status = updated_order.status
                    order.filled_quantity = updated_order.filled_quantity
                    order.avg_fill_price = updated_order.avg_fill_price
                    
                    if order.status == OrderStatus.FILLED:
                        self.logger.log_trade_execution(
                            order_id=order.id,
                            symbol=order.symbol,
                            side=order.side.value,
                            quantity=float(order.filled_quantity),
                            price=float(order.avg_fill_price) if order.avg_fill_price else None
                        )
                        break
                    elif order.status in [OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                        self.logger.log_system_event(f"Order {order.id} status: {order.status.value}")
                        break
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                self.logger.log_error(f"Error monitoring order {order.id}: {str(e)}")
                break
        
        # Handle timeout
        if order.is_active and time.time() - start_time >= self.order_timeout:
            self.logger.log_system_event(f"Order {order.id} timed out, attempting cancellation")
            await self.cancel_order(order.id)
        
        # Move to history
        if order.id in self.active_orders:
            self.order_history.append(self.active_orders.pop(order.id))
    
    async def cancel_order(self, order_id: str) -> Tuple[bool, str]:
        """Cancel an active order"""
        try:
            if order_id not in self.active_orders:
                return False, "Order not found in active orders"
            
            order = self.active_orders[order_id]
            
            # Mock exchange API call
            if self.exchange_client:
                response = await self.exchange_client.cancel_order(order_id)
                if response.get('status') == 'success':
                    order.status = OrderStatus.CANCELLED
                    return True, "Order cancelled successfully"
                else:
                    return False, response.get('error', 'Cancellation failed')
            else:
                # Simulation mode
                order.status = OrderStatus.CANCELLED
                return True, "Order cancelled (simulation mode)"
                
        except Exception as e:
            self.logger.log_error(f"Error cancelling order {order_id}: {str(e)}")
            return False, f"Cancellation error: {str(e)}"
    
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> Dict:
        """Cancel all active orders for a symbol or all symbols"""
        results = {'cancelled': [], 'failed': []}
        
        orders_to_cancel = []
        for order_id, order in self.active_orders.items():
            if symbol is None or order.symbol == symbol:
                orders_to_cancel.append(order_id)
        
        for order_id in orders_to_cancel:
            success, message = await self.cancel_order(order_id)
            if success:
                results['cancelled'].append(order_id)
            else:
                results['failed'].append({'order_id': order_id, 'error': message})
        
        self.logger.log_system_event(f"Bulk cancellation completed", results)
        return results
    
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get current order status from exchange"""
        try:
            if self.exchange_client:
                response = await self.exchange_client.get_order_status(order_id)
                if response.get('status') == 'success':
                    order_data = response.get('order', {})
                    # Convert response to Order object
                    order = Order(
                        id=order_data.get('id'),
                        symbol=order_data.get('symbol'),
                        side=OrderSide(order_data.get('side')),
                        type=OrderType(order_data.get('type')),
                        quantity=Decimal(str(order_data.get('quantity'))),
                        price=Decimal(str(order_data.get('price'))) if order_data.get('price') else None,
                        status=OrderStatus(order_data.get('status')),
                        filled_quantity=Decimal(str(order_data.get('filled_quantity', 0))),
                        avg_fill_price=Decimal(str(order_data.get('avg_fill_price'))) if order_data.get('avg_fill_price') else None
                    )
                    return order
            return None
            
        except Exception as e:
            self.logger.log_error(f"Error getting order status for {order_id}: {str(e)}")
            return None
    
    async def get_current_position(self, symbol: str) -> Decimal:
        """Get current position size for symbol"""
        try:
            if self.exchange_client:
                response = await self.exchange_client.get_position(symbol)
                if response.get('status') == 'success':
                    return Decimal(str(response.get('position', 0)))
            return Decimal('0')  # Simulation mode
            
        except Exception as e:
            self.logger.log_error(f"Error getting position for {symbol}: {str(e)}")
            return Decimal('0')
    
    async def get_current_price(self, symbol: str) -> Optional[Decimal]:
        """Get current market price for symbol"""
        try:
            if self.exchange_client:
                response = await self.exchange_client.get_ticker(symbol)
                if response.get('status') == 'success':
                    return Decimal(str(response.get('price')))
            return Decimal('50000')  # Mock price for simulation
            
        except Exception as e:
            self.logger.log_error(f"Error getting price for {symbol}: {str(e)}")
            return None
    
    async def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol is tradeable"""
        try:
            if self.exchange_client:
                response = await self.exchange_client.get_symbol_info(symbol)
                return response.get('status') == 'success' and response.get('active', False)
            return True  # Simulation mode
            
        except Exception as e:
            self.logger.log_error(f"Error validating symbol {symbol}: {str(e)}")
            return False
    
    def check_daily_trade_limit(self) -> bool:
        """Check if daily trade limit is exceeded"""
        current_date = datetime.now().date()
        
        # Reset counter for new day
        if current_date != self.last_trade_date:
            self.daily_trade_count = 0
            self.last_trade_date = current_date
        
        return self.daily_trade_count < self.max_daily_trades
    
    def trigger_circuit_breaker(self):
        """Trigger circuit breaker to prevent cascade failures"""
        self.circuit_breaker_triggered = True
        self.circuit_breaker_time = time.time()
        self.logger.log_system_event("Circuit breaker triggered", {
            'error_count': self.circuit_breaker_errors,
            'threshold': self.circuit_breaker_threshold
        })
    
    def reset_circuit_breaker(self):
        """Reset circuit breaker"""
        self.circuit_breaker_triggered = False
        self.circuit_breaker_time = None
        self.circuit_breaker_errors = 0
        self.logger.log_system_event("Circuit breaker reset")
    
    def get_order_statistics(self) -> Dict:
        """Get order execution statistics"""
        total_orders = len(self.order_history)
        filled_orders = [o for o in self.order_history if o.status == OrderStatus.FILLED]
        cancelled_orders = [o for o in self.order_history if o.status == OrderStatus.CANCELLED]
        
        return {
            'total_orders': total_orders,
            'filled_orders': len(filled_orders),
            'cancelled_orders': len(cancelled_orders),
            'active_orders': len(self.active_orders),
            'fill_rate': len(filled_orders) / total_orders if total_orders > 0 else 0,
            'daily_trade_count': self.daily_trade_count,
            'circuit_breaker_triggered': self.circuit_breaker_triggered,
            'failed_orders_count': len(self.failed_orders)
        }
    
    def get_active_orders(self) -> List[Order]:
        """Get list of active orders"""
        return list(self.active_orders.values())
    
    def get_order_history(self, limit: int = 100) -> List[Order]:
        """Get order history with limit"""
        return self.order_history[-limit:]
    
    async def emergency_cancel_all(self) -> Dict:
        """Emergency function to cancel all orders"""
        self.logger.log_system_event("Emergency cancellation initiated")
        result = await self.cancel_all_orders()
        self.trigger_circuit_breaker()
        return result
    
    def __del__(self):
        """Cleanup on destruction"""
        if hasattr(self, 'logger'):
            self.logger.log_system_event("OrderManager destroyed", {
                'active_orders': len(self.active_orders),
                'order_history': len(self.order_history)
            })
