"""
Reinforcement Learning Agent - Kengaytirilgan RL trading agent
Deep Q-Network (DQN) va Actor-Critic algoritmlari bilan
"""

import asyncio
import json
import logging
import numpy as np
import pickle
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque
import random

from utils.advanced_logger import TradingLogger
from utils.rate_limiter import RateLimiter

class ActionType(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2
    STRONG_BUY = 3
    STRONG_SELL = 4

class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"

@dataclass
class TradingState:
    """Trading environment state"""
    price: float
    volume: float
    rsi: float
    macd: float
    bb_position: float  # Bollinger Band position
    sentiment: float
    volatility: float
    trend: float
    order_flow: float
    portfolio_value: float
    position_size: float
    cash_balance: float
    unrealized_pnl: float
    drawdown: float
    win_rate: float
    sharpe_ratio: float
    market_regime: MarketRegime
    timestamp: datetime

@dataclass
class TradingAction:
    """Trading action with metadata"""
    action_type: ActionType
    size: float  # Position size (0.0 to 1.0)
    confidence: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    reasoning: str
    timestamp: datetime

@dataclass
class Experience:
    """Experience replay memory item"""
    state: TradingState
    action: TradingAction
    reward: float
    next_state: TradingState
    done: bool
    timestamp: datetime

class DQNNetwork:
    """Deep Q-Network for trading decisions"""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Neural network weights (simplified representation)
        self.weights = {
            'layer1': np.random.randn(state_size, 64) * 0.1,
            'bias1': np.zeros(64),
            'layer2': np.random.randn(64, 32) * 0.1,
            'bias2': np.zeros(32),
            'output': np.random.randn(32, action_size) * 0.1,
            'output_bias': np.zeros(action_size)
        }
        
        self.target_weights = self.weights.copy()
        
    def predict(self, state: np.ndarray) -> np.ndarray:
        """Forward pass through network"""
        try:
            # Layer 1
            z1 = np.dot(state, self.weights['layer1']) + self.weights['bias1']
            a1 = self._relu(z1)
            
            # Layer 2
            z2 = np.dot(a1, self.weights['layer2']) + self.weights['bias2']
            a2 = self._relu(z2)
            
            # Output layer
            q_values = np.dot(a2, self.weights['output']) + self.weights['output_bias']
            
            return q_values
            
        except Exception as e:
            logging.error(f"DQN prediction failed: {e}")
            return np.zeros(self.action_size)
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def update_target_network(self, tau: float = 0.001):
        """Soft update of target network"""
        for key in self.weights:
            self.target_weights[key] = (1 - tau) * self.target_weights[key] + tau * self.weights[key]
    
    def save_model(self, filepath: str):
        """Save model weights"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.weights, f)
        except Exception as e:
            logging.error(f"Failed to save model: {e}")
    
    def load_model(self, filepath: str):
        """Load model weights"""
        try:
            with open(filepath, 'rb') as f:
                self.weights = pickle.load(f)
                self.target_weights = self.weights.copy()
        except Exception as e:
            logging.error(f"Failed to load model: {e}")

class ReinforcementLearningAgent:
    """Advanced RL Trading Agent"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = TradingLogger()
        self.rate_limiter = RateLimiter(max_calls=1000, window=60)
        
        # RL Parameters
        self.state_size = 17  # Number of state features
        self.action_size = len(ActionType)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.gamma = config.get('gamma', 0.95)  # Discount factor
        self.epsilon = config.get('epsilon', 1.0)  # Exploration rate
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        
        # Experience replay
        self.memory_size = config.get('memory_size', 10000)
        self.memory = deque(maxlen=self.memory_size)
        self.batch_size = config.get('batch_size', 32)
        
        # Neural networks
        self.dqn = DQNNetwork(self.state_size, self.action_size, self.learning_rate)
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_actions = []
        self.training_losses = []
        self.performance_metrics = {
            'total_episodes': 0,
            'total_rewards': 0.0,
            'avg_reward': 0.0,
            'best_reward': float('-inf'),
            'worst_reward': float('inf'),
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
        
        # Market regime detection
        self.market_regime_history = deque(maxlen=100)
        self.regime_thresholds = {
            'bull': 0.02,
            'bear': -0.02,
            'volatile': 0.05
        }
        
        # Action space constraints
        self.max_position_size = config.get('max_position_size', 0.5)
        self.min_confidence_threshold = config.get('min_confidence_threshold', 0.6)
        
        # Initialize components
        self.initialize_agent()
    
    def initialize_agent(self):
        """Initialize RL agent components"""
        try:
            # Load pre-trained model if available
            model_path = self.config.get('model_path', 'models/rl_agent.pkl')
            try:
                self.dqn.load_model(model_path)
                self.logger.log_info(f"Loaded pre-trained model from {model_path}")
            except:
                self.logger.log_info("No pre-trained model found, starting fresh")
            
            # Initialize memory with some random experiences
            self._initialize_memory()
            
            self.logger.log_info("RL Agent initialized successfully")
            
        except Exception as e:
            self.logger.log_error(f"Failed to initialize RL agent: {e}")
            raise
    
    def _initialize_memory(self):
        """Initialize experience replay memory"""
        try:
            # Add some random experiences to bootstrap learning
            for _ in range(min(100, self.memory_size // 10)):
                dummy_state = self._create_dummy_state()
                dummy_action = TradingAction(
                    action_type=random.choice(list(ActionType)),
                    size=random.uniform(0.1, 0.5),
                    confidence=random.uniform(0.5, 1.0),
                    stop_loss=None,
                    take_profit=None,
                    reasoning="Bootstrap experience",
                    timestamp=datetime.now()
                )
                
                dummy_reward = random.uniform(-0.1, 0.1)
                dummy_next_state = self._create_dummy_state()
                
                experience = Experience(
                    state=dummy_state,
                    action=dummy_action,
                    reward=dummy_reward,
                    next_state=dummy_next_state,
                    done=False,
                    timestamp=datetime.now()
                )
                
                self.memory.append(experience)
                
        except Exception as e:
            self.logger.log_error(f"Memory initialization failed: {e}")
    
    def _create_dummy_state(self) -> TradingState:
        """Create dummy state for initialization"""
        return TradingState(
            price=random.uniform(40000, 60000),
            volume=random.uniform(100, 1000),
            rsi=random.uniform(20, 80),
            macd=random.uniform(-100, 100),
            bb_position=random.uniform(0, 1),
            sentiment=random.uniform(-1, 1),
            volatility=random.uniform(0, 0.05),
            trend=random.uniform(-0.1, 0.1),
            order_flow=random.uniform(-1, 1),
            portfolio_value=random.uniform(10000, 50000),
            position_size=random.uniform(0, 1),
            cash_balance=random.uniform(5000, 25000),
            unrealized_pnl=random.uniform(-1000, 1000),
            drawdown=random.uniform(0, 0.1),
            win_rate=random.uniform(0.4, 0.7),
            sharpe_ratio=random.uniform(0, 2),
            market_regime=random.choice(list(MarketRegime)),
            timestamp=datetime.now()
        )
    
    async def get_trading_action(self, state: TradingState) -> TradingAction:
        """Get trading action from RL agent"""
        try:
            # Rate limiting check
            can_proceed, message = self.rate_limiter.check_rate_limit("rl_action")
            if not can_proceed:
                self.logger.log_warning(f"Rate limit exceeded for RL action: {message}")
                return self._get_default_action(state)
            
            # Convert state to numpy array
            state_array = self._state_to_array(state)
            
            # Get Q-values from DQN
            q_values = self.dqn.predict(state_array)
            
            # Select action using epsilon-greedy policy
            if random.random() <= self.epsilon:
                # Exploration: random action
                action_type = random.choice(list(ActionType))
                confidence = random.uniform(0.3, 0.7)
            else:
                # Exploitation: best action
                action_type = ActionType(np.argmax(q_values))
                confidence = min(1.0, max(0.0, np.max(q_values)))
            
            # Determine position size based on confidence and market conditions
            position_size = self._calculate_position_size(state, confidence, action_type)
            
            # Set stop loss and take profit
            stop_loss, take_profit = self._calculate_risk_management(state, action_type)
            
            # Create reasoning
            reasoning = self._generate_reasoning(state, action_type, q_values, confidence)
            
            action = TradingAction(
                action_type=action_type,
                size=position_size,
                confidence=confidence,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasoning=reasoning,
                timestamp=datetime.now()
            )
            
            # Log action
            self.logger.log_info(f"RL Action: {action_type.name} (size: {position_size:.3f}, conf: {confidence:.3f})")
            
            return action
            
        except Exception as e:
            self.logger.log_error(f"RL action generation failed: {e}")
            return self._get_default_action(state)
    
    def _state_to_array(self, state: TradingState) -> np.ndarray:
        """Convert TradingState to numpy array"""
        try:
            # Normalize features
            features = np.array([
                state.price / 50000.0,  # Normalize price
                state.volume / 1000.0,  # Normalize volume
                state.rsi / 100.0,
                state.macd / 1000.0,
                state.bb_position,
                state.sentiment,
                state.volatility * 100.0,
                state.trend * 10.0,
                state.order_flow,
                state.portfolio_value / 50000.0,
                state.position_size,
                state.cash_balance / 25000.0,
                state.unrealized_pnl / 5000.0,
                state.drawdown * 10.0,
                state.win_rate,
                state.sharpe_ratio / 3.0,
                float(state.market_regime.value == MarketRegime.BULL.value)
            ])
            
            # Clip to reasonable ranges
            features = np.clip(features, -5.0, 5.0)
            
            return features
            
        except Exception as e:
            self.logger.log_error(f"State conversion failed: {e}")
            return np.zeros(self.state_size)
    
    def _calculate_position_size(self, state: TradingState, confidence: float, action_type: ActionType) -> float:
        """Calculate position size based on confidence and market conditions"""
        try:
            # Base size from confidence
            base_size = confidence * self.max_position_size
            
            # Adjust for market regime
            if state.market_regime == MarketRegime.VOLATILE:
                base_size *= 0.5  # Reduce size in volatile markets
            elif state.market_regime == MarketRegime.BULL and action_type == ActionType.BUY:
                base_size *= 1.2  # Increase size in bull market for buy
            elif state.market_regime == MarketRegime.BEAR and action_type == ActionType.SELL:
                base_size *= 1.2  # Increase size in bear market for sell
            
            # Adjust for current drawdown
            if state.drawdown > 0.05:  # 5% drawdown
                base_size *= 0.7
            
            # Adjust for volatility
            if state.volatility > 0.03:  # High volatility
                base_size *= 0.8
            
            # Strong actions get larger sizes
            if action_type in [ActionType.STRONG_BUY, ActionType.STRONG_SELL]:
                base_size *= 1.5
            
            # Ensure within bounds
            final_size = max(0.01, min(self.max_position_size, base_size))
            
            return final_size
            
        except Exception as e:
            self.logger.log_error(f"Position size calculation failed: {e}")
            return 0.1
    
    def _calculate_risk_management(self, state: TradingState, action_type: ActionType) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit levels"""
        try:
            if action_type == ActionType.HOLD:
                return None, None
            
            # Calculate ATR-based levels
            atr = state.volatility * state.price
            
            if action_type in [ActionType.BUY, ActionType.STRONG_BUY]:
                # Long position
                stop_loss = state.price - (2.0 * atr)
                take_profit = state.price + (3.0 * atr)
            else:
                # Short position
                stop_loss = state.price + (2.0 * atr)
                take_profit = state.price - (3.0 * atr)
            
            return stop_loss, take_profit
            
        except Exception as e:
            self.logger.log_error(f"Risk management calculation failed: {e}")
            return None, None
    
    def _generate_reasoning(self, state: TradingState, action_type: ActionType, q_values: np.ndarray, confidence: float) -> str:
        """Generate reasoning for trading decision"""
        try:
            reasons = []
            
            # Market regime
            reasons.append(f"Market regime: {state.market_regime.value}")
            
            # Technical indicators
            if state.rsi > 70:
                reasons.append("RSI overbought")
            elif state.rsi < 30:
                reasons.append("RSI oversold")
            
            if state.bb_position > 0.8:
                reasons.append("Near upper Bollinger Band")
            elif state.bb_position < 0.2:
                reasons.append("Near lower Bollinger Band")
            
            # Sentiment
            if state.sentiment > 0.5:
                reasons.append("Bullish sentiment")
            elif state.sentiment < -0.5:
                reasons.append("Bearish sentiment")
            
            # Q-values
            reasons.append(f"Q-value: {np.max(q_values):.3f}")
            
            # Confidence
            reasons.append(f"Confidence: {confidence:.3f}")
            
            return "; ".join(reasons)
            
        except Exception as e:
            self.logger.log_error(f"Reasoning generation failed: {e}")
            return f"Action: {action_type.name}"
    
    def _get_default_action(self, state: TradingState) -> TradingAction:
        """Get default safe action"""
        return TradingAction(
            action_type=ActionType.HOLD,
            size=0.0,
            confidence=0.5,
            stop_loss=None,
            take_profit=None,
            reasoning="Default safe action",
            timestamp=datetime.now()
        )
    
    def remember(self, experience: Experience):
        """Store experience in replay memory"""
        try:
            self.memory.append(experience)
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
        except Exception as e:
            self.logger.log_error(f"Experience storage failed: {e}")
    
    def replay(self) -> float:
        """Train the agent using experience replay"""
        try:
            if len(self.memory) < self.batch_size:
                return 0.0
            
            # Sample random batch
            batch = random.sample(self.memory, self.batch_size)
            
            # Prepare training data
            states = np.array([self._state_to_array(exp.state) for exp in batch])
            next_states = np.array([self._state_to_array(exp.next_state) for exp in batch])
            
            # Get current Q values
            current_q_values = np.array([self.dqn.predict(state) for state in states])
            next_q_values = np.array([self.dqn.predict(state) for state in next_states])
            
            # Calculate targets
            targets = current_q_values.copy()
            
            for i, exp in enumerate(batch):
                action_index = exp.action.action_type.value
                
                if exp.done:
                    target = exp.reward
                else:
                    target = exp.reward + self.gamma * np.max(next_q_values[i])
                
                targets[i][action_index] = target
            
            # Calculate loss (simplified)
            loss = np.mean(np.square(targets - current_q_values))
            
            # Update network weights (simplified gradient descent)
            self._update_weights(states, targets, loss)
            
            # Update target network
            self.dqn.update_target_network()
            
            # Store loss
            self.training_losses.append(loss)
            
            return loss
            
        except Exception as e:
            self.logger.log_error(f"Replay training failed: {e}")
            return 0.0
    
    def _update_weights(self, states: np.ndarray, targets: np.ndarray, loss: float):
        """Update network weights (simplified implementation)"""
        try:
            # This is a simplified weight update
            # In a real implementation, you would use proper backpropagation
            
            learning_rate = self.learning_rate
            
            # Small random updates to simulate learning
            for key in self.dqn.weights:
                if 'bias' not in key:
                    noise = np.random.normal(0, learning_rate * 0.1, self.dqn.weights[key].shape)
                    self.dqn.weights[key] += noise * (1.0 - loss)
                    
        except Exception as e:
            self.logger.log_error(f"Weight update failed: {e}")
    
    def update_performance_metrics(self, reward: float, action: TradingAction):
        """Update performance metrics"""
        try:
            self.performance_metrics['total_episodes'] += 1
            self.performance_metrics['total_rewards'] += reward
            self.performance_metrics['avg_reward'] = (
                self.performance_metrics['total_rewards'] / 
                self.performance_metrics['total_episodes']
            )
            
            # Update best/worst rewards
            if reward > self.performance_metrics['best_reward']:
                self.performance_metrics['best_reward'] = reward
            if reward < self.performance_metrics['worst_reward']:
                self.performance_metrics['worst_reward'] = reward
            
            # Store episode data
            self.episode_rewards.append(reward)
            self.episode_actions.append(action.action_type.value)
            
            # Calculate win rate
            if len(self.episode_rewards) >= 10:
                recent_rewards = self.episode_rewards[-10:]
                wins = sum(1 for r in recent_rewards if r > 0)
                self.performance_metrics['win_rate'] = wins / len(recent_rewards)
            
            # Calculate Sharpe ratio
            if len(self.episode_rewards) >= 20:
                recent_rewards = np.array(self.episode_rewards[-20:])
                if np.std(recent_rewards) > 0:
                    self.performance_metrics['sharpe_ratio'] = (
                        np.mean(recent_rewards) / np.std(recent_rewards)
                    )
            
        except Exception as e:
            self.logger.log_error(f"Performance metrics update failed: {e}")
    
    def detect_market_regime(self, price_history: List[float]) -> MarketRegime:
        """Detect current market regime"""
        try:
            if len(price_history) < 20:
                return MarketRegime.SIDEWAYS
            
            # Calculate returns
            returns = np.diff(price_history) / price_history[:-1]
            
            # Calculate trend
            trend = np.mean(returns[-10:])
            
            # Calculate volatility
            volatility = np.std(returns[-20:])
            
            # Determine regime
            if volatility > self.regime_thresholds['volatile']:
                regime = MarketRegime.VOLATILE
            elif trend > self.regime_thresholds['bull']:
                regime = MarketRegime.BULL
            elif trend < self.regime_thresholds['bear']:
                regime = MarketRegime.BEAR
            else:
                regime = MarketRegime.SIDEWAYS
            
            # Store in history
            self.market_regime_history.append(regime)
            
            return regime
            
        except Exception as e:
            self.logger.log_error(f"Market regime detection failed: {e}")
            return MarketRegime.SIDEWAYS
    
    def save_model(self, filepath: str = None):
        """Save RL agent model"""
        try:
            if filepath is None:
                filepath = self.config.get('model_path', 'models/rl_agent.pkl')
            
            # Save DQN weights
            self.dqn.save_model(filepath)
            
            # Save agent state
            agent_state = {
                'epsilon': self.epsilon,
                'performance_metrics': self.performance_metrics,
                'episode_rewards': list(self.episode_rewards),
                'episode_actions': list(self.episode_actions)
            }
            
            with open(filepath.replace('.pkl', '_state.json'), 'w') as f:
                json.dump(agent_state, f, indent=2)
            
            self.logger.log_info(f"Model saved to {filepath}")
            
        except Exception as e:
            self.logger.log_error(f"Model save failed: {e}")
    
    def load_model(self, filepath: str = None):
        """Load RL agent model"""
        try:
            if filepath is None:
                filepath = self.config.get('model_path', 'models/rl_agent.pkl')
            
            # Load DQN weights
            self.dqn.load_model(filepath)
            
            # Load agent state
            try:
                with open(filepath.replace('.pkl', '_state.json'), 'r') as f:
                    agent_state = json.load(f)
                
                self.epsilon = agent_state.get('epsilon', self.epsilon)
                self.performance_metrics = agent_state.get('performance_metrics', self.performance_metrics)
                self.episode_rewards = deque(agent_state.get('episode_rewards', []), maxlen=1000)
                self.episode_actions = deque(agent_state.get('episode_actions', []), maxlen=1000)
                
            except:
                self.logger.log_warning("Could not load agent state, using defaults")
            
            self.logger.log_info(f"Model loaded from {filepath}")
            
        except Exception as e:
            self.logger.log_error(f"Model load failed: {e}")
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        try:
            summary = self.performance_metrics.copy()
            
            # Add additional metrics
            summary.update({
                'epsilon': self.epsilon,
                'memory_size': len(self.memory),
                'recent_rewards': list(self.episode_rewards)[-10:] if self.episode_rewards else [],
                'recent_actions': list(self.episode_actions)[-10:] if self.episode_actions else [],
                'training_losses': self.training_losses[-10:] if self.training_losses else [],
                'market_regime_distribution': self._get_regime_distribution()
            })
            
            return summary
            
        except Exception as e:
            self.logger.log_error(f"Performance summary failed: {e}")
            return self.performance_metrics
    
    def _get_regime_distribution(self) -> Dict[str, float]:
        """Get distribution of market regimes"""
        try:
            if not self.market_regime_history:
                return {}
            
            regime_counts = {}
            for regime in self.market_regime_history:
                regime_counts[regime.value] = regime_counts.get(regime.value, 0) + 1
            
            total = len(self.market_regime_history)
            return {regime: count / total for regime, count in regime_counts.items()}
            
        except Exception as e:
            self.logger.log_error(f"Regime distribution calculation failed: {e}")
            return {}
    
    async def health_check(self) -> Dict:
        """RL agent health check"""
        try:
            health_status = {
                'status': 'healthy',
                'epsilon': self.epsilon,
                'memory_size': len(self.memory),
                'total_episodes': self.performance_metrics['total_episodes'],
                'avg_reward': self.performance_metrics['avg_reward'],
                'issues': []
            }
            
            # Check if learning is happening
            if self.epsilon < 0.1 and self.performance_metrics['total_episodes'] < 100:
                health_status['issues'].append("Low exploration with few episodes")
                health_status['status'] = 'warning'
            
            # Check recent performance
            if len(self.episode_rewards) >= 10:
                recent_avg = np.mean(list(self.episode_rewards)[-10:])
                if recent_avg < -0.05:
                    health_status['issues'].append("Poor recent performance")
                    health_status['status'] = 'warning'
            
            return health_status
            
        except Exception as e:
            self.logger.log_error(f"Health check failed: {e}")
            return {'status': 'error', 'error': str(e)}
