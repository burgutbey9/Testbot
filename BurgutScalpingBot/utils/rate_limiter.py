import time
import asyncio
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
from functools import wraps
import logging
from datetime import datetime, timedelta

class RateLimiter:
    def __init__(self, max_calls: int = 10, window_seconds: int = 60):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.calls = defaultdict(deque)  # API name -> deque of timestamps
        self.blocked_until = defaultdict(float)  # API name -> timestamp when unblocked
        self.logger = logging.getLogger('rate_limiter')
        
    def check_rate_limit(self, api_name: str) -> Tuple[bool, str]:
        """Check if API call is within rate limit"""
        current_time = time.time()
        
        # Check if currently blocked
        if current_time < self.blocked_until[api_name]:
            remaining_time = self.blocked_until[api_name] - current_time
            return False, f"{api_name} blocked for {remaining_time:.1f}s"
        
        # Clean old calls outside window
        call_times = self.calls[api_name]
        while call_times and current_time - call_times[0] > self.window_seconds:
            call_times.popleft()
        
        # Check if we can make another call
        if len(call_times) >= self.max_calls:
            # Calculate when we can make next call
            oldest_call = call_times[0]
            reset_time = oldest_call + self.window_seconds
            self.blocked_until[api_name] = reset_time
            
            return False, f"{api_name} rate limit exceeded. Reset in {reset_time - current_time:.1f}s"
        
        # Add current call
        call_times.append(current_time)
        remaining_calls = self.max_calls - len(call_times)
        
        return True, f"{api_name} OK. {remaining_calls} calls remaining in window"
    
    def get_status(self, api_name: str) -> Dict[str, any]:
        """Get current rate limit status for API"""
        current_time = time.time()
        call_times = self.calls[api_name]
        
        # Clean old calls
        while call_times and current_time - call_times[0] > self.window_seconds:
            call_times.popleft()
        
        return {
            'api_name': api_name,
            'calls_used': len(call_times),
            'calls_remaining': self.max_calls - len(call_times),
            'window_seconds': self.window_seconds,
            'reset_time': call_times[0] + self.window_seconds if call_times else current_time,
            'blocked_until': self.blocked_until[api_name] if current_time < self.blocked_until[api_name] else None
        }
    
    def reset_api_limits(self, api_name: str):
        """Reset rate limits for specific API"""
        if api_name in self.calls:
            self.calls[api_name].clear()
        if api_name in self.blocked_until:
            del self.blocked_until[api_name]
        self.logger.info(f"Rate limits reset for {api_name}")

class ExponentialBackoff:
    def __init__(self, base_delay: float = 1.0, max_delay: float = 300.0, multiplier: float = 2.0):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.attempt_count = defaultdict(int)
        self.last_attempt = defaultdict(float)
        
    def get_delay(self, operation_name: str) -> float:
        """Get delay for next retry attempt"""
        current_time = time.time()
        
        # Reset attempts if enough time has passed
        if current_time - self.last_attempt[operation_name] > self.max_delay:
            self.attempt_count[operation_name] = 0
        
        self.attempt_count[operation_name] += 1
        self.last_attempt[operation_name] = current_time
        
        # Calculate exponential backoff delay
        delay = min(
            self.base_delay * (self.multiplier ** (self.attempt_count[operation_name] - 1)),
            self.max_delay
        )
        
        return delay
    
    def reset_attempts(self, operation_name: str):
        """Reset attempt count for successful operation"""
        if operation_name in self.attempt_count:
            del self.attempt_count[operation_name]
        if operation_name in self.last_attempt:
            del self.last_attempt[operation_name]

class APICallManager:
    def __init__(self):
        self.rate_limiters = {}
        self.backoff_handlers = {}
        self.logger = logging.getLogger('api_manager')
        
        # Default rate limits for different APIs
        self.default_limits = {
            'binance': {'max_calls': 1200, 'window': 60},  # 1200 calls per minute
            'coinbase': {'max_calls': 10, 'window': 1},    # 10 calls per second
            'uniswap': {'max_calls': 5, 'window': 1},      # 5 calls per second
            'telegram': {'max_calls': 30, 'window': 1},    # 30 calls per second
            'openai': {'max_calls': 60, 'window': 60},     # 60 calls per minute
            'default': {'max_calls': 10, 'window': 60}     # Default fallback
        }
        
        self.setup_rate_limiters()
    
    def setup_rate_limiters(self):
        """Setup rate limiters for all APIs"""
        for api_name, limits in self.default_limits.items():
            self.rate_limiters[api_name] = RateLimiter(
                max_calls=limits['max_calls'],
                window_seconds=limits['window']
            )
            self.backoff_handlers[api_name] = ExponentialBackoff()
    
    def get_rate_limiter(self, api_name: str) -> RateLimiter:
        """Get rate limiter for specific API"""
        if api_name not in self.rate_limiters:
            # Create default rate limiter for unknown API
            limits = self.default_limits['default']
            self.rate_limiters[api_name] = RateLimiter(
                max_calls=limits['max_calls'],
                window_seconds=limits['window']
            )
            self.backoff_handlers[api_name] = ExponentialBackoff()
        
        return self.rate_limiters[api_name]
    
    async def execute_with_rate_limit(self, api_name: str, operation_name: str, func, *args, **kwargs):
        """Execute function with rate limiting and exponential backoff"""
        rate_limiter = self.get_rate_limiter(api_name)
        backoff_handler = self.backoff_handlers[api_name]
        
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            # Check rate limit
            can_call, message = rate_limiter.check_rate_limit(api_name)
            
            if not can_call:
                # Rate limited, wait and retry
                delay = backoff_handler.get_delay(f"{api_name}_{operation_name}")
                self.logger.warning(f"Rate limited: {message}. Waiting {delay:.1f}s")
                await asyncio.sleep(delay)
                retry_count += 1
                continue
            
            try:
                # Execute the function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Success, reset backoff
                backoff_handler.reset_attempts(f"{api_name}_{operation_name}")
                return result
                
            except Exception as e:
                retry_count += 1
                delay = backoff_handler.get_delay(f"{api_name}_{operation_name}")
                
                self.logger.error(f"API call failed: {api_name}.{operation_name} - {str(e)}")
                
                if retry_count >= max_retries:
                    self.logger.error(f"Max retries exceeded for {api_name}.{operation_name}")
                    raise e
                
                self.logger.info(f"Retrying in {delay:.1f}s (attempt {retry_count}/{max_retries})")
                await asyncio.sleep(delay)
        
        raise Exception(f"Max retries exceeded for {api_name}.{operation_name}")
    
    def get_all_status(self) -> Dict[str, Dict]:
        """Get status of all rate limiters"""
        status = {}
        for api_name, rate_limiter in self.rate_limiters.items():
            status[api_name] = rate_limiter.get_status(api_name)
        return status

# Decorator for rate limiting
def rate_limit(api_name: str, operation_name: str = None):
    """Decorator to apply rate limiting to functions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            manager = APICallManager()
            return await manager.execute_with_rate_limit(api_name, op_name, func, *args, **kwargs)
        return wrapper
    return decorator

# Global API manager instance
api_manager = APICallManager()

# Example usage decorators
@rate_limit('binance', 'get_price')
async def get_binance_price(symbol: str):
    """Example function with rate limiting"""
    # Your API call logic here
    pass

@rate_limit('telegram', 'send_message')
async def send_telegram_message(chat_id: str, message: str):
    """Example Telegram function with rate limiting"""
    # Your Telegram API call logic here
    pass

# Circuit breaker pattern
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self.logger = logging.getLogger('circuit_breaker')
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker pattern"""
        if self.state == 'OPEN':
            if self.should_attempt_reset():
                self.state = 'HALF_OPEN'
                self.logger.info("Circuit breaker: Attempting reset")
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise e
    
    def should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        
        return (time.time() - self.last_failure_time) >= self.recovery_timeout
    
    def on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            self.logger.error(f"Circuit breaker OPEN after {self.failure_count} failures")

# Global circuit breaker instance
circuit_breaker = CircuitBreaker()
