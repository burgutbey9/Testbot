"""
Advanced Risk Management System - Kengaytirilgan risk boshqaruv tizimi
"""

import asyncio
import logging
import math
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import json

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RiskType(Enum):
    MARKET = "market"
    LIQUIDITY = "liquidity"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    DRAWDOWN = "drawdown"
    CONCENTRATION = "concentration"
    LEVERAGE = "leverage"
    OPERATIONAL = "operational"

@dataclass
class RiskMetric:
    risk_type: RiskType
    current_value: float
    threshold: float
    risk_level: RiskLevel
    description: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def is_breached(self) -> bool:
        return self.current_value > self.threshold

@dataclass
class RiskLimit:
    name: str
    limit_type: str
    max_value: float
    current_value: float = 0.0
    warning_threshold: float = 0.8  # 80% of max
    breached: bool = False
    breach_count: int = 0
    last_breach: Optional[datetime] = None
    
    def check_breach(self, value: float) -> bool:
        self.current_value = value
        was_breached = self.breached
        self.breached = value > self.max_value
        
        if self.breached and not was_breached:
            self.breach_count += 1
            self.last_breach = datetime.now()
            
        return self.breached
    
    def get_utilization(self) -> float:
        return (self.current_value / self.max_value) if self.max_value > 0 else 0.0

class RiskManager:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Risk limits
        self.risk_limits = self._initialize_risk_limits()
        
        # Risk metrics history
        self.risk_history = defaultdict(deque)
        self.max_history_length = 1000
        
        # Volatility tracking
        self.price_history = defaultdict(deque)
        self.volatility_window = 20  # 20 periods
        
        # Correlation matrix
        self.correlation_matrix = {}
        self.correlation_window = 50
        
        # Portfolio metrics
        self.portfolio_value_history = deque(maxlen=100)
        self.daily_returns = deque(maxlen=252)  # 1 year
        
        # Risk parameters
        self.max_portfolio_risk = config.get('max_portfolio_risk', 0.02)  # 2% VaR
        self.max_single_position_risk = config.get('max_single_position_risk', 0.01)  # 1%
        self.max_sector_exposure = config.get('max_sector_exposure', 0.30)  # 30%
        self.max_correlation_exposure = config.get('max_correlation_exposure', 0.60)  # 60%
        self.confidence_level = config.get('confidence_level', 0.95)  # 95% VaR
        
        # Circuit breaker
        self.circuit_breaker_active = False
        self.circuit_breaker_threshold = config.get('circuit_breaker_threshold', 0.05)  # 5%
        self.circuit_breaker_cooldown = config.get('circuit_breaker_cooldown', 3600)  # 1 hour
        self.circuit_breaker_triggered_at = None
        
        # Risk alerts
        self.risk_alerts = []
        self.alert_cooldown = {}
        
        # Stress testing
        self.stress_scenarios = self._initialize_stress_scenarios()
        
    def _initialize_risk_limits(self) -> Dict[str, RiskLimit]:
        """Risk limitlarini boshlang'ich sozlash"""
        limits = {}
        
        # Portfolio level limits
        limits['max_drawdown'] = RiskLimit(
            name="Maximum Drawdown",
            limit_type="portfolio",
            max_value=self.config.get('max_drawdown', 0.05),  # 5%
            warning_threshold=0.8
        )
        
        limits['max_daily_loss'] = RiskLimit(
            name="Maximum Daily Loss",
            limit_type="portfolio",
            max_value=self.config.get('max_daily_loss', 0.02),  # 2%
            warning_threshold=0.8
        )
        
        limits['max_positions'] = RiskLimit(
            name="Maximum Positions",
            limit_type="portfolio",
            max_value=self.config.get('max_positions', 10),
            warning_threshold=0.8
        )
        
        limits['max_leverage'] = RiskLimit(
            name="Maximum Leverage",
            limit_type="portfolio",
            max_value=self.config.get('max_leverage', 3.0),
            warning_threshold=0.8
        )
        
        limits['max_concentration'] = RiskLimit(
            name="Maximum Single Asset Concentration",
            limit_type="position",
            max_value=self.config.get('max_concentration', 0.20),  # 20%
            warning_threshold=0.8
        )
        
        limits['max_correlation_risk'] = RiskLimit(
            name="Maximum Correlation Risk",
            limit_type="portfolio",
            max_value=self.config.get('max_correlation_risk', 0.70),  # 70%
            warning_threshold=0.8
        )
        
        return limits
    
    def _initialize_stress_scenarios(self) -> Dict[str, Dict]:
        """Stress test scenariylarini boshlang'ich sozlash"""
        return {
            'market_crash': {
                'description': 'Market crash scenario (-20% in all assets)',
                'shock_factor': -0.20,
                'correlation_increase': 0.30
            },
            'flash_crash': {
                'description': 'Flash crash scenario (-10% in 5 minutes)',
                'shock_factor': -0.10,
                'time_horizon': 300  # 5 minutes
            },
            'high_volatility': {
                'description': 'High volatility scenario (3x normal vol)',
                'volatility_multiplier': 3.0,
                'correlation_increase': 0.20
            },
            'liquidity_crisis': {
                'description': 'Liquidity crisis (spreads widen 5x)',
                'spread_multiplier': 5.0,
                'impact_multiplier': 2.0
            }
        }
    
    async def evaluate_trade_risk(self, symbol: str, position_type: str, 
                                 size: float, price: float, 
                                 portfolio_value: float) -> Tuple[bool, str, Dict]:
        """Savdo riskini baholash"""
        try:
            risk_assessment = {
                'approved': True,
                'risk_level': RiskLevel.LOW,
                'risk_score': 0.0,
                'warnings': [],
                'blocking_factors': [],
                'recommendations': []
            }
            
            # Position size risk
            position_value = size * price
            position_risk = position_value / portfolio_value if portfolio_value > 0 else 0
            
            if position_risk > self.max_single_position_risk:
                risk_assessment['blocking_factors'].append(
                    f"Position size risk {position_risk:.2%} exceeds limit {self.max_single_position_risk:.2%}"
                )
                risk_assessment['approved'] = False
            
            # Concentration risk
            concentration_risk = await self._calculate_concentration_risk(symbol, position_value, portfolio_value)
            if concentration_risk > self.risk_limits['max_concentration'].max_value:
                risk_assessment['blocking_factors'].append(
                    f"Concentration risk {concentration_risk:.2%} exceeds limit"
                )
                risk_assessment['approved'] = False
            
            # Correlation risk
            correlation_risk = await self._calculate_correlation_risk(symbol, position_type, position_value)
            if correlation_risk > self.max_correlation_exposure:
                risk_assessment['warnings'].append(
                    f"High correlation risk detected: {correlation_risk:.2%}"
                )
                risk_assessment['risk_level'] = RiskLevel.HIGH
            
            # Volatility risk
            volatility_risk = await self._calculate_volatility_risk(symbol, size)
            if volatility_risk > 0.10:  # 10% volatility threshold
                risk_assessment['warnings'].append(
                    f"High volatility risk: {volatility_risk:.2%}"
                )
            
            # Liquidity risk
            liquidity_risk = await self._calculate_liquidity_risk(symbol, size)
            if liquidity_risk > 0.05:  # 5% liquidity risk threshold
                risk_assessment['warnings'].append(
                    f"Liquidity risk detected: {liquidity_risk:.2%}"
                )
            
            # Circuit breaker check
            if self.circuit_breaker_active:
                risk_assessment['blocking_factors'].append("Circuit breaker is active")
                risk_assessment['approved'] = False
            
            # Calculate overall risk score
            risk_assessment['risk_score'] = (
                position_risk * 0.3 +
                concentration_risk * 0.2 +
                correlation_risk * 0.2 +
                volatility_risk * 0.15 +
                liquidity_risk * 0.15
            )
            
            # Determine risk level
            if risk_assessment['risk_score'] > 0.08:
                risk_assessment['risk_level'] = RiskLevel.CRITICAL
            elif risk_assessment['risk_score'] > 0.05:
                risk_assessment['risk_level'] = RiskLevel.HIGH
            elif risk_assessment['risk_score'] > 0.02:
                risk_assessment['risk_level'] = RiskLevel.MEDIUM
            
            # Generate recommendations
            if risk_assessment['risk_score'] > 0.05:
                risk_assessment['recommendations'].append("Consider reducing position size")
            
            if correlation_risk > 0.50:
                risk_assessment['recommendations'].append("High correlation with existing positions")
            
            if volatility_risk > 0.08:
                risk_assessment['recommendations'].append("Consider wider stop losses due to high volatility")
            
            return risk_assessment['approved'], "Risk assessment completed", risk_assessment
            
        except Exception as e:
            self.logger.error(f"Error evaluating trade risk: {str(e)}")
            return False, str(e), {}
    
    async def _calculate_concentration_risk(self, symbol: str, position_value: float, 
                                          portfolio_value: float) -> float:
        """Konsentratsiya riskini hisoblash"""
        try:
            if portfolio_value == 0:
                return 0.0
            
            # Current concentration would be
            concentration = position_value / portfolio_value
            
            # Check existing positions in same symbol
            # This should be integrated with position manager
            # For now, assume no existing positions
            
            return concentration
            
        except Exception as e:
            self.logger.error(f"Error calculating concentration risk: {str(e)}")
            return 0.0
    
    async def _calculate_correlation_risk(self, symbol: str, position_type: str, 
                                        position_value: float) -> float:
        """Korrelyatsiya riskini hisoblash"""
        try:
            if not self.correlation_matrix:
                return 0.0
            
            # Check correlation with existing positions
            # This is a simplified version - should be integrated with position manager
            max_correlation = 0.0
            
            for key, correlation in self.correlation_matrix.items():
                if symbol in key:
                    correlation_value = abs(correlation)
                    max_correlation = max(max_correlation, correlation_value)
            
            return max_correlation
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation risk: {str(e)}")
            return 0.0
    
    async def _calculate_volatility_risk(self, symbol: str, size: float) -> float:
        """Volatillik riskini hisoblash"""
        try:
            if symbol not in self.price_history or len(self.price_history[symbol]) < 2:
                return 0.05  # Default volatility assumption
            
            prices = list(self.price_history[symbol])
            returns = []
            
            for i in range(1, len(prices)):
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)
            
            if len(returns) < 2:
                return 0.05
            
            # Calculate volatility (standard deviation of returns)
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            # Risk is volatility * position size factor
            position_risk = volatility * min(size / 1000, 1.0)  # Normalize by size
            
            return position_risk
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility risk: {str(e)}")
            return 0.05
    
    async def _calculate_liquidity_risk(self, symbol: str, size: float) -> float:
        """Likvidlik riskini hisoblash"""
        try:
            # This should be integrated with order book data
            # For now, use size-based heuristic
            
            if size > 10000:  # Large orders
                return 0.08
            elif size > 5000:  # Medium orders
                return 0.04
            elif size > 1000:  # Small orders
                return 0.02
            else:
                return 0.01
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity risk: {str(e)}")
            return 0.02
    
    async def update_portfolio_metrics(self, portfolio_value: float, 
                                     positions: Dict, price_data: Dict[str, float]):
        """Portfolio metrikalarini yangilash"""
        try:
            # Update portfolio value history
            self.portfolio_value_history.append({
                'value': portfolio_value,
                'timestamp': datetime.now()
            })
            
            # Calculate daily returns
            if len(self.portfolio_value_history) > 1:
                prev_value = self.portfolio_value_history[-2]['value']
                daily_return = (portfolio_value - prev_value) / prev_value
                self.daily_returns.append(daily_return)
            
            # Update price history
            for symbol, price in price_data.items():
                if symbol not in self.price_history:
                    self.price_history[symbol] = deque(maxlen=self.volatility_window)
                self.price_history[symbol].append(price)
            
            # Update correlation matrix
            await self._update_correlation_matrix(price_data)
            
            # Check risk limits
            await self._check_risk_limits(portfolio_value, positions)
            
            # Check circuit breaker
            await self._check_circuit_breaker(portfolio_value)
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio metrics: {str(e)}")
    
    async def _update_correlation_matrix(self, price_data: Dict[str, float]):
        """Korrelyatsiya matrisasini yangilash"""
        try:
            symbols = list(price_data.keys())
            
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols[i+1:], i+1):
                    if (symbol1 in self.price_history and 
                        symbol2 in self.price_history and
                        len(self.price_history[symbol1]) > 10 and
                        len(self.price_history[symbol2]) > 10):
                        
                        prices1 = np.array(list(self.price_history[symbol1]))
                        prices2 = np.array(list(self.price_history[symbol2]))
                        
                        # Calculate returns
                        returns1 = np.diff(prices1) / prices1[:-1]
                        returns2 = np.diff(prices2) / prices2[:-1]
                        
                        # Calculate correlation
                        if len(returns1) > 5 and len(returns2) > 5:
                            correlation = np.corrcoef(returns1, returns2)[0, 1]
                            if not np.isnan(correlation):
                                self.correlation_matrix[f"{symbol1}_{symbol2}"] = correlation
            
        except Exception as e:
            self.logger.error(f"Error updating correlation matrix: {str(e)}")
    
    async def _check_risk_limits(self, portfolio_value: float, positions: Dict):
        """Risk limitlarini tekshirish"""
        try:
            # Check drawdown
            if len(self.portfolio_value_history) > 1:
                peak_value = max(item['value'] for item in self.portfolio_value_history)
                drawdown = (peak_value - portfolio_value) / peak_value
                
                if self.risk_limits['max_drawdown'].check_breach(drawdown):
                    await self._trigger_risk_alert(
                        RiskType.DRAWDOWN,
                        f"Maximum drawdown breached: {drawdown:.2%}",
                        RiskLevel.CRITICAL
                    )
            
            # Check daily loss
            if len(self.daily_returns) > 0:
                daily_loss = abs(min(self.daily_returns[-1], 0))
                if self.risk_limits['max_daily_loss'].check_breach(daily_loss):
                    await self._trigger_risk_alert(
                        RiskType.MARKET,
                        f"Daily loss limit breached: {daily_loss:.2%}",
                        RiskLevel.HIGH
                    )
            
            # Check position count
            position_count = len(positions)
            if self.risk_limits['max_positions'].check_breach(position_count):
                await self._trigger_risk_alert(
                    RiskType.CONCENTRATION,
                    f"Maximum positions breached: {position_count}",
                    RiskLevel.MEDIUM
                )
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {str(e)}")
    
    async def _check_circuit_breaker(self, portfolio_value: float):
        """Circuit breaker tekshirish"""
        try:
            if len(self.portfolio_value_history) < 2:
                return
            
            start_value = self.portfolio_value_history[0]['value']
            current_loss = (start_value - portfolio_value) / start_value
            
            if current_loss > self.circuit_breaker_threshold and not self.circuit_breaker_active:
                self.circuit_breaker_active = True
                self.circuit_breaker_triggered_at = datetime.now()
                
                await self._trigger_risk_alert(
                    RiskType.MARKET,
                    f"Circuit breaker triggered: {current_loss:.2%} loss",
                    RiskLevel.CRITICAL
                )
                
                self.logger.critical(f"Circuit breaker activated due to {current_loss:.2%} loss")
            
            # Check cooldown
            if (self.circuit_breaker_active and 
                self.circuit_breaker_triggered_at and
                datetime.now() - self.circuit_breaker_triggered_at > timedelta(seconds=self.circuit_breaker_cooldown)):
                
                self.circuit_breaker_active = False
                self.circuit_breaker_triggered_at = None
                self.logger.info("Circuit breaker deactivated after cooldown")
            
        except Exception as e:
            self.logger.error(f"Error checking circuit breaker: {str(e)}")
    
    async def _trigger_risk_alert(self, risk_type: RiskType, message: str, 
                                 risk_level: RiskLevel):
        """Risk alert trigger qilish"""
        try:
            # Check cooldown
            alert_key = f"{risk_type.value}_{risk_level.value}"
            now = datetime.now()
            
            if alert_key in self.alert_cooldown:
                if now - self.alert_cooldown[alert_key] < timedelta(minutes=15):
                    return  # Still in cooldown
            
            # Create alert
            alert = {
                'risk_type': risk_type.value,
                'risk_level': risk_level.value,
                'message': message,
                'timestamp': now.isoformat(),
                'portfolio_value': self.portfolio_value_history[-1]['value'] if self.portfolio_value_history else 0
            }
            
            self.risk_alerts.append(alert)
            
            # Keep only last 100 alerts
            if len(self.risk_alerts) > 100:
                self.risk_alerts = self.risk_alerts[-100:]
            
            # Set cooldown
            self.alert_cooldown[alert_key] = now
            
            self.logger.warning(f"Risk alert triggered: {message}")
            
        except Exception as e:
            self.logger.error(f"Error triggering risk alert: {str(e)}")
    
    async def run_stress_test(self, scenario: str, positions: Dict, 
                            price_data: Dict[str, float]) -> Dict:
        """Stress test o'tkazish"""
        try:
            if scenario not in self.stress_scenarios:
                return {'error': f'Unknown scenario: {scenario}'}
            
            scenario_config = self.stress_scenarios[scenario]
            results = {
                'scenario': scenario,
                'description': scenario_config['description'],
                'original_portfolio_value': 0,
                'stressed_portfolio_value': 0,
                'portfolio_loss': 0,
                'portfolio_loss_pct': 0,
                'position_impacts': {},
                'risk_metrics': {}
            }
            
            # Calculate original portfolio value
            original_value = sum(pos.get('size', 0) * pos.get('current_price', 0) 
                               for pos in positions.values())
            results['original_portfolio_value'] = original_value
            
            # Apply stress scenario
            stressed_value = 0
            for pos_id, position in positions.items():
                symbol = position.get('symbol', '')
                size = position.get('size', 0)
                current_price = position.get('current_price', 0)
                
                # Apply shock
                if 'shock_factor' in scenario_config:
                    shocked_price = current_price * (1 + scenario_config['shock_factor'])
                elif 'volatility_multiplier' in scenario_config:
                    # Use historical volatility if available
                    volatility = await self._calculate_volatility_risk(symbol, size)
                    shock = volatility * scenario_config['volatility_multiplier']
                    shocked_price = current_price * (1 - shock)
                else:
                    shocked_price = current_price
                
                stressed_position_value = size * shocked_price
                stressed_value += stressed_position_value
                
                # Track individual position impact
                position_loss = (size * current_price) - stressed_position_value
                results['position_impacts'][pos_id] = {
                    'symbol': symbol,
                    'original_value': size * current_price,
                    'stressed_value': stressed_position_value,
                    'loss': position_loss,
                    'loss_pct': (position_loss / (size * current_price)) * 100 if current_price > 0 else 0
                }
            
            results['stressed_portfolio_value'] = stressed_value
            results['portfolio_loss'] = original_value - stressed_value
            results['portfolio_loss_pct'] = (results['portfolio_loss'] / original_value) * 100 if original_value > 0 else 0
            
            # Calculate risk metrics
            results['risk_metrics'] = {
                'var_95': results['portfolio_loss_pct'] * 0.95,
                'expected_shortfall': results['portfolio_loss_pct'] * 1.2,
                'max_individual_loss': max(
                    (pos['loss_pct'] for pos in results['position_impacts'].values()),
                    default=0
                )
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error running stress test: {str(e)}")
            return {'error': str(e)}
    
    def calculate_var(self, confidence_level: float = 0.95) -> float:
        """Value at Risk (VaR) hisoblash"""
        try:
            if len(self.daily_returns) < 30:
                return 0.0
            
            returns = np.array(list(self.daily_returns))
            var = np.percentile(returns, (1 - confidence_level) * 100)
            
            return abs(var)
            
        except Exception as e:
            self.logger.error(f"Error calculating VaR: {str(e)}")
            return 0.0
    
    def calculate_expected_shortfall(self, confidence_level: float = 0.95) -> float:
        """Expected Shortfall (ES) hisoblash"""
        try:
            if len(self.daily_returns) < 30:
                return 0.0
            
            returns = np.array(list(self.daily_returns))
            var = np.percentile(returns, (1 - confidence_level) * 100)
            
            # Expected shortfall is the mean of returns below VaR
            tail_returns = returns[returns <= var]
            
            if len(tail_returns) == 0:
                return abs(var)
            
            expected_shortfall = np.mean(tail_returns)
            
            return abs(expected_shortfall)
            
        except Exception as e:
            self.logger.error(f"Error calculating Expected Shortfall: {str(e)}")
            return 0.0
    
    def get_risk_summary(self) -> Dict:
        """Risk summary report"""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'circuit_breaker_active': self.circuit_breaker_active,
                'risk_limits': {},
                'risk_metrics': {},
                'recent_alerts': [],
                'correlation_summary': {}
            }
            
            # Risk limits status
            for name, limit in self.risk_limits.items():
                summary['risk_limits'][name] = {
                    'current_value': limit.current_value,
                    'max_value': limit.max_value,
                    'utilization': limit.get_utilization(),
                    'breached': limit.breached,
                    'breach_count': limit.breach_count
                }
            
            # Risk metrics
            summary['risk_metrics'] = {
                'var_95': self.calculate_var(0.95),
                'expected_shortfall': self.calculate_expected_shortfall(0.95),
                'portfolio_volatility': np.std(list(self.daily_returns)) * np.sqrt(252) if len(self.daily_returns) > 1 else 0
            }
            
            # Recent alerts (last 10)
            summary['recent_alerts'] = self.risk_alerts[-10:] if self.risk_alerts else []
            
            # Correlation summary
            if self.correlation_matrix:
                correlations = list(self.correlation_matrix.values())
                summary['correlation_summary'] = {
                    'max_correlation': max(correlations),
                    'min_correlation': min(correlations),
                    'avg_correlation': np.mean(correlations),
                    'high_correlation_pairs': len([c for c in correlations if c > 0.8])
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating risk summary: {str(e)}")
            return {}
    
    def export_risk_report(self) -> str:
        """Risk report ni JSON formatda eksport qilish"""
        try:
            report = {
                'risk_manager_report': {
                    'generated_at': datetime.now().isoformat(),
                    'risk_summary': self.get_risk_summary(),
                    'stress_test_scenarios': list(self.stress_scenarios.keys()),
                    'risk_limits_config': {
                        name: {
                            'max_value': limit.max_value,
                            'warning_threshold': limit.warning_threshold,
                            'breach_count': limit.breach_count
                        }
                        for name, limit in self.risk_limits.items()
                    },
                    'correlation_matrix': dict(self.correlation_matrix),
                    'alert_history': self.risk_alerts
                }
            }
            
            return json.dumps(report, indent=2)
            
        except Exception as e:
            self.logger.error(f"Error exporting risk report: {str(e)}")
            return "{}"
    
    def reset_circuit_breaker(self):
        """Circuit breaker ni reset qilish"""
        self.circuit_breaker_active = False
        self.circuit_breaker_triggered_at = None
        self.logger.info("Circuit breaker manually reset")
    
    def add_custom_risk_limit(self, name: str, limit_type: str, max_value: float):
        """Custom risk limit qo'shish"""
        self.risk_limits[name] = RiskLimit(
            name=name,
            limit_type=limit_type,
            max_value=max_value
        )
        self.logger.info(f"Custom risk limit added: {name} = {max_value}")
