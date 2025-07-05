import asyncio
import time
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from abc import ABC, abstractmethod

class StrategyType(Enum):
    SCALPING = "scalping"
    SWING = "swing"
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"
    TREND_FOLLOWING = "trend_following"

class SignalStrength(Enum):
    WEAK = 1
    MEDIUM = 2
    STRONG = 3
    VERY_STRONG = 4

@dataclass
class TradingSignal:
    strategy_name: str
    signal_type: str  # 'buy', 'sell', 'hold'
    strength: SignalStrength
    confidence: float  # 0-1
    price: float
    size: float
    timestamp: datetime
    reasoning: str
    ai_features: Dict[str, Any]
    risk_score: float  # 0-1
    expected_duration: int  # seconds
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)

class BaseStrategy(ABC):
    def __init__(self, name: str, params: Dict):
        self.name = name
        self.params = params
        self.active = True
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'avg_trade_duration': 0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'last_updated': datetime.now()
        }
        self.trade_history = []
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
    @abstractmethod
    async def generate_signal(self, market_data: Dict) -> Optional[TradingSignal]:
        """Generate trading signal based on market data"""
        pass
    
    @abstractmethod
    def update_performance(self, trade_result: Dict):
        """Update strategy performance metrics"""
        pass
    
    def get_performance(self) -> Dict:
        return self.performance_metrics.copy()
    
    def is_active(self) -> bool:
        return self.active
    
    def activate(self):
        self.active = True
        self.logger.info(f"Strategy {self.name} activated")
    
    def deactivate(self):
        self.active = False
        self.logger.info(f"Strategy {self.name} deactivated")

class ScalpingStrategy(BaseStrategy):
    def __init__(self, params: Dict):
        super().__init__("AI_Scalping", params)
        self.min_spread = params.get('min_spread', 0.0001)
        self.max_position_time = params.get('max_position_time', 300)  # 5 minutes
        self.sentiment_threshold = params.get('sentiment_threshold', 0.6)
        self.orderflow_threshold = params.get('orderflow_threshold', 0.7)
        
    async def generate_signal(self, market_data: Dict) -> Optional[TradingSignal]:
        try:
            # Extract market data
            price = market_data.get('price', 0)
            spread = market_data.get('spread', 0)
            volume = market_data.get('volume', 0)
            
            # AI features
            ai_sentiment = market_data.get('ai_sentiment', 0)
            orderflow_imbalance = market_data.get('orderflow_imbalance', 0)
            price_momentum = market_data.get('price_momentum', 0)
            
            # Check basic conditions
            if spread < self.min_spread:
                return None
                
            # AI-based signal generation
            signal_strength = self._calculate_signal_strength(
                ai_sentiment, orderflow_imbalance, price_momentum
            )
            
            if signal_strength < 0.5:
                return None
            
            # Determine signal type
            if ai_sentiment > self.sentiment_threshold and orderflow_imbalance > 0:
                signal_type = 'buy'
            elif ai_sentiment < (1 - self.sentiment_threshold) and orderflow_imbalance < 0:
                signal_type = 'sell'
            else:
                return None
            
            # Calculate position size based on volatility
            volatility = market_data.get('volatility', 0.02)
            base_size = self.params.get('base_position_size', 1000)
            position_size = base_size * (1 - volatility)  # Reduce size in high volatility
            
            # Set stop loss and take profit
            atr = market_data.get('atr', price * 0.001)  # Average True Range
            if signal_type == 'buy':
                stop_loss = price - (atr * 2)
                take_profit = price + (atr * 1.5)
            else:
                stop_loss = price + (atr * 2)
                take_profit = price - (atr * 1.5)
            
            # Create signal
            signal = TradingSignal(
                strategy_name=self.name,
                signal_type=signal_type,
                strength=self._get_signal_strength_enum(signal_strength),
                confidence=signal_strength,
                price=price,
                size=position_size,
                timestamp=datetime.now(),
                reasoning=f"AI Sentiment: {ai_sentiment:.2f}, OrderFlow: {orderflow_imbalance:.2f}",
                ai_features={
                    'sentiment': ai_sentiment,
                    'orderflow_imbalance': orderflow_imbalance,
                    'price_momentum': price_momentum,
                    'volatility': volatility,
                    'spread': spread
                },
                risk_score=self._calculate_risk_score(market_data),
                expected_duration=self.max_position_time,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {str(e)}")
            return None
    
    def _calculate_signal_strength(self, sentiment: float, orderflow: float, momentum: float) -> float:
        """Calculate combined signal strength"""
        # Weighted combination of factors
        weights = {
            'sentiment': 0.4,
            'orderflow': 0.4,
            'momentum': 0.2
        }
        
        # Normalize values
        sentiment_score = abs(sentiment - 0.5) * 2  # 0-1 scale
        orderflow_score = abs(orderflow)  # Already -1 to 1
        momentum_score = abs(momentum)
        
        combined_score = (
            sentiment_score * weights['sentiment'] +
            orderflow_score * weights['orderflow'] +
            momentum_score * weights['momentum']
        )
        
        return min(combined_score, 1.0)
    
    def _get_signal_strength_enum(self, score: float) -> SignalStrength:
        """Convert numeric score to SignalStrength enum"""
        if score >= 0.8:
            return SignalStrength.VERY_STRONG
        elif score >= 0.6:
            return SignalStrength.STRONG
        elif score >= 0.4:
            return SignalStrength.MEDIUM
        else:
            return SignalStrength.WEAK
    
    def _calculate_risk_score(self, market_data: Dict) -> float:
        """Calculate risk score for the trade"""
        volatility = market_data.get('volatility', 0.02)
        spread = market_data.get('spread', 0.0001)
        volume = market_data.get('volume', 1000)
        
        # Higher volatility = higher risk
        volatility_risk = min(volatility * 10, 1.0)
        
        # Wider spread = higher risk
        spread_risk = min(spread * 1000, 1.0)
        
        # Lower volume = higher risk
        volume_risk = max(0, 1 - (volume / 10000))
        
        # Combined risk score
        risk_score = (volatility_risk + spread_risk + volume_risk) / 3
        return min(risk_score, 1.0)
    
    def update_performance(self, trade_result: Dict):
        """Update strategy performance metrics"""
        pnl = trade_result.get('pnl', 0)
        duration = trade_result.get('duration', 0)
        
        self.performance_metrics['total_trades'] += 1
        self.performance_metrics['total_pnl'] += pnl
        
        if pnl > 0:
            self.performance_metrics['winning_trades'] += 1
        
        # Update averages
        self.performance_metrics['win_rate'] = (
            self.performance_metrics['winning_trades'] / 
            self.performance_metrics['total_trades']
        )
        
        # Update average trade duration
        current_avg = self.performance_metrics['avg_trade_duration']
        total_trades = self.performance_metrics['total_trades']
        self.performance_metrics['avg_trade_duration'] = (
            (current_avg * (total_trades - 1) + duration) / total_trades
        )
        
        # Store trade history
        self.trade_history.append({
            'pnl': pnl,
            'duration': duration,
            'timestamp': datetime.now()
        })
        
        # Keep only last 100 trades
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]
        
        # Calculate Sharpe ratio and max drawdown
        if len(self.trade_history) >= 10:
            self._calculate_advanced_metrics()
        
        self.performance_metrics['last_updated'] = datetime.now()
    
    def _calculate_advanced_metrics(self):
        """Calculate advanced performance metrics"""
        if not self.trade_history:
            return
        
        pnls = [trade['pnl'] for trade in self.trade_history]
        
        # Sharpe ratio
        if len(pnls) > 1:
            returns = np.array(pnls)
            if np.std(returns) != 0:
                self.performance_metrics['sharpe_ratio'] = (
                    np.mean(returns) / np.std(returns)
                )
        
        # Max drawdown
        cumulative_pnl = np.cumsum(pnls)
        peak = np.maximum.accumulate(cumulative_pnl)
        drawdown = (cumulative_pnl - peak) / peak
        self.performance_metrics['max_drawdown'] = abs(np.min(drawdown))

class StrategyManager:
    def __init__(self, config: Dict):
        self.config = config
        self.strategies: Dict[str, BaseStrategy] = {}
        self.active_signals: List[TradingSignal] = []
        self.signal_history: List[TradingSignal] = []
        self.performance_tracker = {}
        
        # Circuit breaker settings
        self.circuit_breaker_active = False
        self.max_consecutive_losses = config.get('max_consecutive_losses', 5)
        self.consecutive_losses = 0
        self.min_win_rate = config.get('min_win_rate', 0.3)
        self.performance_check_interval = config.get('performance_check_interval', 3600)  # 1 hour
        
        # Strategy switching
        self.strategy_switching_enabled = config.get('strategy_switching_enabled', True)
        self.performance_window = config.get('performance_window', 24)  # hours
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize strategies
        self._initialize_strategies()
        
        # Start background tasks
        asyncio.create_task(self._performance_monitor())
    
    def _initialize_strategies(self):
        """Initialize all trading strategies"""
        # Scalping strategy
        scalping_params = self.config.get('scalping_params', {
            'min_spread': 0.0001,
            'max_position_time': 300,
            'sentiment_threshold': 0.6,
            'orderflow_threshold': 0.7,
            'base_position_size': 1000
        })
        
        self.strategies['scalping'] = ScalpingStrategy(scalping_params)
        
        # Add more strategies here as needed
        self.logger.info(f"Initialized {len(self.strategies)} strategies")
    
    async def generate_signals(self, market_data: Dict) -> List[TradingSignal]:
        """Generate signals from all active strategies"""
        if self.circuit_breaker_active:
            self.logger.warning("Circuit breaker active - no signals generated")
            return []
        
        signals = []
        
        for strategy_name, strategy in self.strategies.items():
            if strategy.is_active():
                try:
                    signal = await strategy.generate_signal(market_data)
                    if signal:
                        signals.append(signal)
                        self.logger.info(f"Signal generated by {strategy_name}: {signal.signal_type}")
                except Exception as e:
                    self.logger.error(f"Error generating signal from {strategy_name}: {str(e)}")
        
        # Filter and rank signals
        filtered_signals = self._filter_signals(signals)
        ranked_signals = self._rank_signals(filtered_signals)
        
        # Store signals
        self.active_signals = ranked_signals
        self.signal_history.extend(ranked_signals)
        
        # Keep signal history manageable
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]
        
        return ranked_signals
    
    def _filter_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Filter signals based on quality criteria"""
        filtered = []
        
        for signal in signals:
            # Minimum confidence threshold
            if signal.confidence < 0.5:
                continue
            
            # Risk score threshold
            if signal.risk_score > 0.8:
                continue
            
            # Avoid conflicting signals
            if not self._check_signal_conflicts(signal):
                continue
            
            filtered.append(signal)
        
        return filtered
    
    def _rank_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Rank signals by quality score"""
        def signal_score(signal: TradingSignal) -> float:
            # Combine confidence and inverse risk score
            quality_score = signal.confidence * (1 - signal.risk_score)
            
            # Boost score for stronger signals
            strength_multiplier = {
                SignalStrength.WEAK: 0.8,
                SignalStrength.MEDIUM: 1.0,
                SignalStrength.STRONG: 1.2,
                SignalStrength.VERY_STRONG: 1.5
            }
            
            return quality_score * strength_multiplier[signal.strength]
        
        return sorted(signals, key=signal_score, reverse=True)
    
    def _check_signal_conflicts(self, new_signal: TradingSignal) -> bool:
        """Check if new signal conflicts with existing signals"""
        for existing_signal in self.active_signals:
            # Same asset, opposite direction
            if (existing_signal.signal_type != new_signal.signal_type and
                abs(existing_signal.price - new_signal.price) < new_signal.price * 0.001):
                return False
        
        return True
    
    def update_strategy_performance(self, strategy_name: str, trade_result: Dict):
        """Update performance for specific strategy"""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].update_performance(trade_result)
            
            # Check for consecutive losses
            if trade_result.get('pnl', 0) < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
            
            # Activate circuit breaker if needed
            if self.consecutive_losses >= self.max_consecutive_losses:
                self._activate_circuit_breaker()
    
    def _activate_circuit_breaker(self):
        """Activate circuit breaker to stop trading"""
        self.circuit_breaker_active = True
        self.logger.critical(f"Circuit breaker activated after {self.consecutive_losses} consecutive losses")
        
        # Deactivate all strategies
        for strategy in self.strategies.values():
            strategy.deactivate()
    
    def _deactivate_circuit_breaker(self):
        """Deactivate circuit breaker and resume trading"""
        self.circuit_breaker_active = False
        self.consecutive_losses = 0
        self.logger.info("Circuit breaker deactivated - resuming trading")
        
        # Reactivate strategies
        for strategy in self.strategies.values():
            strategy.activate()
    
    async def _performance_monitor(self):
        """Background task to monitor strategy performance"""
        while True:
            try:
                await asyncio.sleep(self.performance_check_interval)
                
                # Check overall performance
                overall_performance = self.get_overall_performance()
                
                # Check win rate
                if overall_performance['win_rate'] < self.min_win_rate:
                    if not self.circuit_breaker_active:
                        self.logger.warning(f"Low win rate detected: {overall_performance['win_rate']:.2%}")
                        self._activate_circuit_breaker()
                
                # Auto-recovery logic
                if self.circuit_breaker_active:
                    # Check if we should try to recover
                    if self._should_attempt_recovery():
                        self._deactivate_circuit_breaker()
                
                # Strategy switching logic
                if self.strategy_switching_enabled:
                    await self._evaluate_strategy_switching()
                
            except Exception as e:
                self.logger.error(f"Error in performance monitor: {str(e)}")
    
    def _should_attempt_recovery(self) -> bool:
        """Determine if we should attempt to recover from circuit breaker"""
        # Simple time-based recovery (can be enhanced)
        if hasattr(self, 'circuit_breaker_start_time'):
            time_since_activation = time.time() - self.circuit_breaker_start_time
            return time_since_activation > 3600  # 1 hour cooldown
        
        return False
    
    async def _evaluate_strategy_switching(self):
        """Evaluate if we should switch strategies based on performance"""
        if len(self.strategies) <= 1:
            return
        
        # Get performance for each strategy
        performance_scores = {}
        
        for name, strategy in self.strategies.items():
            perf = strategy.get_performance()
            # Simple scoring: win_rate * total_pnl
            score = perf['win_rate'] * max(perf['total_pnl'], 0)
            performance_scores[name] = score
        
        # Find best performing strategy
        best_strategy = max(performance_scores, key=performance_scores.get)
        worst_strategy = min(performance_scores, key=performance_scores.get)
        
        # If performance gap is significant, switch
        if performance_scores[best_strategy] > performance_scores[worst_strategy] * 2:
            self.logger.info(f"Switching from {worst_strategy} to {best_strategy}")
            self.strategies[worst_strategy].deactivate()
            self.strategies[best_strategy].activate()
    
    def get_overall_performance(self) -> Dict:
        """Get combined performance metrics from all strategies"""
        total_trades = sum(s.performance_metrics['total_trades'] for s in self.strategies.values())
        total_winning = sum(s.performance_metrics['winning_trades'] for s in self.strategies.values())
        total_pnl = sum(s.performance_metrics['total_pnl'] for s in self.strategies.values())
        
        if total_trades == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl_per_trade': 0.0,
                'active_strategies': len([s for s in self.strategies.values() if s.is_active()]),
                'circuit_breaker_active': self.circuit_breaker_active
            }
        
        return {
            'total_trades': total_trades,
            'win_rate': total_winning / total_trades,
            'total_pnl': total_pnl,
            'avg_pnl_per_trade': total_pnl / total_trades,
            'active_strategies': len([s for s in self.strategies.values() if s.is_active()]),
            'circuit_breaker_active': self.circuit_breaker_active
        }
    
    def get_strategy_performance(self, strategy_name: str) -> Optional[Dict]:
        """Get performance for specific strategy"""
        if strategy_name in self.strategies:
            return self.strategies[strategy_name].get_performance()
        return None
    
    def get_all_strategies_performance(self) -> Dict[str, Dict]:
        """Get performance for all strategies"""
        return {
            name: strategy.get_performance() 
            for name, strategy in self.strategies.items()
        }
    
    def get_active_signals(self) -> List[TradingSignal]:
        """Get currently active signals"""
        return self.active_signals.copy()
    
    def get_signal_history(self, limit: int = 100) -> List[TradingSignal]:
        """Get signal history"""
        return self.signal_history[-limit:]
    
    def clear_active_signals(self):
        """Clear active signals (called after execution)"""
        self.active_signals.clear()
    
    def add_custom_strategy(self, strategy: BaseStrategy):
        """Add a custom strategy"""
        self.strategies[strategy.name] = strategy
        self.logger.info(f"Added custom strategy: {strategy.name}")
    
    def remove_strategy(self, strategy_name: str):
        """Remove a strategy"""
        if strategy_name in self.strategies:
            del self.strategies[strategy_name]
            self.logger.info(f"Removed strategy: {strategy_name}")
    
    def get_status(self) -> Dict:
        """Get current status of strategy manager"""
        return {
            'total_strategies': len(self.strategies),
            'active_strategies': len([s for s in self.strategies.values() if s.is_active()]),
            'active_signals': len(self.active_signals),
            'circuit_breaker_active': self.circuit_breaker_active,
            'consecutive_losses': self.consecutive_losses,
            'overall_performance': self.get_overall_performance(),
            'strategy_list': list(self.strategies.keys())
        }

# Usage example
if __name__ == "__main__":
    async def main():
        config = {
            'max_consecutive_losses': 3,
            'min_win_rate': 0.4,
            'performance_check_interval': 300,  # 5 minutes
            'strategy_switching_enabled': True,
            'scalping_params': {
                'min_spread': 0.0001,
                'max_position_time': 300,
                'sentiment_threshold': 0.6,
                'orderflow_threshold': 0.7,
                'base_position_size': 1000
            }
        }
        
        manager = StrategyManager(config)
        
        # Sample market data
        market_data = {
            'price': 50000,
            'spread': 0.0002,
            'volume': 5000,
            'ai_sentiment': 0.75,
            'orderflow_imbalance': 0.3,
            'price_momentum': 0.1,
            'volatility': 0.02,
            'atr': 500
        }
        
        # Generate signals
        signals = await manager.generate_signals(market_data)
        
        for signal in signals:
            print(f"Signal: {signal.signal_type} at {signal.price} with confidence {signal.confidence:.2f}")
        
        # Simulate trade result
        trade_result = {
            'pnl': 25.50,
            'duration': 180
        }
        
        manager.update_strategy_performance('scalping', trade_result)
        
        # Get status
        status = manager.get_status()
        print(f"Manager Status: {status}")
        
        await asyncio.sleep(2)
    
    asyncio.run(main())
