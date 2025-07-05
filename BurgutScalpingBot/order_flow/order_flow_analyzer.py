"""
Order Flow Analyzer - Buyurtma oqimini tahlil qilish moduli
Katta orderlar, whale faoliyati, mikrostruktura patternlarini aniqlash
"""

import asyncio
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import logging
from datetime import datetime, timedelta
import statistics

from utils.advanced_logger import TradingLogger
from utils.rate_limiter import RateLimiter

@dataclass
class OrderData:
    """Buyurtma ma'lumotlari struktura"""
    timestamp: float
    price: float
    size: float
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market', 'limit', 'stop'
    exchange: str
    order_id: str = ""
    
class OrderFlowAnalyzer:
    """Order Flow tahlil qiluvchi asosiy klass"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = TradingLogger()
        self.rate_limiter = RateLimiter(max_calls=100, window=60)
        
        # Order flow parametrlari
        self.whale_threshold = config.get('whale_threshold', 100000)  # $100k
        self.large_order_threshold = config.get('large_order_threshold', 50000)  # $50k
        self.analysis_window = config.get('analysis_window', 300)  # 5 daqiqa
        
        # Ma'lumotlar saqlash
        self.order_history = deque(maxlen=10000)
        self.trade_history = deque(maxlen=10000)
        self.volume_profile = defaultdict(float)
        
        # Pattern detection
        self.pattern_cache = {}
        self.last_analysis = 0
        
        # Metrics
        self.metrics = {
            'total_volume': 0,
            'buy_volume': 0,
            'sell_volume': 0,
            'large_orders_count': 0,
            'whale_activity_score': 0,
            'order_flow_imbalance': 0,
            'aggressive_ratio': 0
        }
        
        self.logger.log_info("OrderFlowAnalyzer initialized")
    
    async def process_order_data(self, order_data: OrderData) -> Dict[str, Any]:
        """Order ma'lumotlarini qayta ishlash"""
        try:
            # Rate limiting check
            can_process, msg = self.rate_limiter.check_rate_limit("order_processing")
            if not can_process:
                self.logger.log_warning(f"Rate limit exceeded: {msg}")
                return {'status': 'rate_limited'}
            
            # Orderni saqlash
            self.order_history.append(order_data)
            
            # Metrikalarni yangilash
            self.update_metrics(order_data)
            
            # Pattern detection
            patterns = await self.detect_patterns(order_data)
            
            # Whale activity detection
            whale_activity = self.detect_whale_activity([order_data])
            
            # Order flow imbalance
            imbalance = self.calculate_order_flow_imbalance()
            
            # Microstructure analysis
            microstructure = self.analyze_microstructure()
            
            analysis_result = {
                'timestamp': time.time(),
                'order_data': order_data,
                'patterns': patterns,
                'whale_activity': whale_activity,
                'imbalance': imbalance,
                'microstructure': microstructure,
                'metrics': self.metrics.copy(),
                'signals': self.generate_signals()
            }
            
            self.logger.log_trade_decision(
                signal=analysis_result,
                decision="order_flow_analysis",
                reasoning="Real-time order flow processing"
            )
            
            return analysis_result
            
        except Exception as e:
            self.logger.log_error(f"Order processing error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def update_metrics(self, order: OrderData):
        """Metrikalarni yangilash"""
        order_value = order.price * order.size
        
        self.metrics['total_volume'] += order_value
        
        if order.side == 'buy':
            self.metrics['buy_volume'] += order_value
        else:
            self.metrics['sell_volume'] += order_value
        
        # Large order detection
        if order_value >= self.large_order_threshold:
            self.metrics['large_orders_count'] += 1
        
        # Whale activity score
        if order_value >= self.whale_threshold:
            self.metrics['whale_activity_score'] += order_value / self.whale_threshold
    
    def detect_whale_activity(self, orders: List[OrderData]) -> Dict[str, Any]:
        """Whale faoliyatini aniqlash"""
        whale_orders = []
        total_whale_volume = 0
        
        for order in orders:
            order_value = order.price * order.size
            if order_value >= self.whale_threshold:
                whale_orders.append({
                    'timestamp': order.timestamp,
                    'price': order.price,
                    'size': order.size,
                    'side': order.side,
                    'value': order_value,
                    'exchange': order.exchange
                })
                total_whale_volume += order_value
        
        if not whale_orders:
            return {
                'detected': False,
                'impact': 'none',
                'orders': [],
                'total_volume': 0
            }
        
        # Impact assessment
        recent_volume = sum(o.price * o.size for o in list(self.order_history)[-100:])
        impact_ratio = total_whale_volume / max(recent_volume, 1)
        
        if impact_ratio > 0.3:
            impact = 'high'
        elif impact_ratio > 0.1:
            impact = 'medium'
        else:
            impact = 'low'
        
        return {
            'detected': True,
            'impact': impact,
            'orders': whale_orders,
            'total_volume': total_whale_volume,
            'impact_ratio': impact_ratio,
            'order_count': len(whale_orders)
        }
    
    def calculate_order_flow_imbalance(self) -> Dict[str, Any]:
        """Order flow nomutanosibligini hisoblash"""
        if len(self.order_history) < 10:
            return {'direction': 'neutral', 'strength': 0, 'ratio': 0.5}
        
        # So'nggi N ta orderni tahlil qilish
        recent_orders = list(self.order_history)[-100:]
        
        buy_volume = sum(o.price * o.size for o in recent_orders if o.side == 'buy')
        sell_volume = sum(o.price * o.size for o in recent_orders if o.side == 'sell')
        
        total_volume = buy_volume + sell_volume
        
        if total_volume == 0:
            return {'direction': 'neutral', 'strength': 0, 'ratio': 0.5}
        
        buy_ratio = buy_volume / total_volume
        sell_ratio = sell_volume / total_volume
        
        # Direction and strength
        if buy_ratio > 0.6:
            direction = 'bullish'
            strength = (buy_ratio - 0.5) * 2
        elif sell_ratio > 0.6:
            direction = 'bearish'
            strength = (sell_ratio - 0.5) * 2
        else:
            direction = 'neutral'
            strength = abs(buy_ratio - sell_ratio) * 2
        
        self.metrics['order_flow_imbalance'] = strength if direction == 'bullish' else -strength
        
        return {
            'direction': direction,
            'strength': strength,
            'ratio': buy_ratio,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'total_volume': total_volume
        }
    
    async def detect_patterns(self, current_order: OrderData) -> Dict[str, Any]:
        """Mikrostruktura patternlarini aniqlash"""
        patterns = {}
        
        # Iceberg orders detection
        patterns['iceberg_orders'] = self.detect_iceberg_orders()
        
        # Spoofing attempts
        patterns['spoofing_attempts'] = self.detect_spoofing()
        
        # Momentum ignition
        patterns['momentum_ignition'] = self.detect_momentum_ignition()
        
        # Layering
        patterns['layering'] = self.detect_layering()
        
        # Order clustering
        patterns['order_clustering'] = self.detect_order_clustering(current_order)
        
        return patterns
    
    def detect_iceberg_orders(self) -> Dict[str, Any]:
        """Iceberg orderlarini aniqlash"""
        if len(self.order_history) < 50:
            return {'detected': False, 'confidence': 0}
        
        recent_orders = list(self.order_history)[-50:]
        
        # Bir xil narx darajasida takrorlanuvchi orderlar
        price_frequency = defaultdict(int)
        size_consistency = defaultdict(list)
        
        for order in recent_orders:
            price_level = round(order.price, 2)
            price_frequency[price_level] += 1
            size_consistency[price_level].append(order.size)
        
        # Iceberg pattern detection
        iceberg_candidates = []
        
        for price, freq in price_frequency.items():
            if freq >= 5:  # Kamida 5 ta order
                sizes = size_consistency[price]
                size_std = statistics.stdev(sizes) if len(sizes) > 1 else 0
                avg_size = statistics.mean(sizes)
                
                # Bir xil o'lchamdagi orderlar
                if size_std < avg_size * 0.1:  # 10% dan kam farq
                    iceberg_candidates.append({
                        'price': price,
                        'frequency': freq,
                        'avg_size': avg_size,
                        'size_consistency': 1 - (size_std / avg_size) if avg_size > 0 else 0
                    })
        
        if iceberg_candidates:
            confidence = min(len(iceberg_candidates) * 0.2, 1.0)
            return {
                'detected': True,
                'confidence': confidence,
                'candidates': iceberg_candidates
            }
        
        return {'detected': False, 'confidence': 0}
    
    def detect_spoofing(self) -> Dict[str, Any]:
        """Spoofing holatlarini aniqlash"""
        if len(self.order_history) < 20:
            return {'detected': False, 'confidence': 0}
        
        recent_orders = list(self.order_history)[-20:]
        
        # Katta orderlarning tez bekor qilinishi
        large_orders = [o for o in recent_orders if o.price * o.size >= self.large_order_threshold]
        
        if len(large_orders) < 2:
            return {'detected': False, 'confidence': 0}
        
        # Pattern: Katta order -> Kichik orderlar -> Katta order bekor
        spoofing_score = 0
        
        for i, order in enumerate(large_orders[:-1]):
            next_order = large_orders[i + 1]
            
            # Qarama-qarshi tomonlar
            if order.side != next_order.side:
                time_diff = next_order.timestamp - order.timestamp
                
                # Qisqa vaqt oralig'i
                if time_diff < 60:  # 1 daqiqa
                    spoofing_score += 0.3
                
                # Narx darajasi yaqinligi
                price_diff = abs(order.price - next_order.price) / order.price
                if price_diff < 0.01:  # 1% dan kam
                    spoofing_score += 0.2
        
        confidence = min(spoofing_score, 1.0)
        detected = confidence > 0.3
        
        return {
            'detected': detected,
            'confidence': confidence,
            'score': spoofing_score
        }
    
    def detect_momentum_ignition(self) -> Dict[str, Any]:
        """Momentum ignition patternini aniqlash"""
        if len(self.order_history) < 30:
            return {'detected': False, 'confidence': 0}
        
        recent_orders = list(self.order_history)[-30:]
        
        # Ketma-ket bir tomonga katta orderlar
        consecutive_large = []
        current_streak = []
        
        for order in recent_orders:
            if order.price * order.size >= self.large_order_threshold:
                if not current_streak or current_streak[-1].side == order.side:
                    current_streak.append(order)
                else:
                    if len(current_streak) >= 3:
                        consecutive_large.append(current_streak)
                    current_streak = [order]
        
        if len(current_streak) >= 3:
            consecutive_large.append(current_streak)
        
        if consecutive_large:
            # Eng uzun streakni topish
            longest_streak = max(consecutive_large, key=len)
            confidence = min(len(longest_streak) * 0.2, 1.0)
            
            return {
                'detected': True,
                'confidence': confidence,
                'streak_length': len(longest_streak),
                'direction': longest_streak[0].side
            }
        
        return {'detected': False, 'confidence': 0}
    
    def detect_layering(self) -> Dict[str, Any]:
        """Layering patternini aniqlash"""
        if len(self.order_history) < 40:
            return {'detected': False, 'confidence': 0}
        
        recent_orders = list(self.order_history)[-40:]
        
        # Bir tomonda ko'p orderlar, ikkinchi tomonda kam
        price_levels = defaultdict(lambda: {'buy': 0, 'sell': 0})
        
        for order in recent_orders:
            price_level = round(order.price, 2)
            price_levels[price_level][order.side] += order.size
        
        # Asymmetric layering detection
        layering_score = 0
        
        for price, sides in price_levels.items():
            buy_size = sides['buy']
            sell_size = sides['sell']
            
            if buy_size > 0 and sell_size > 0:
                ratio = max(buy_size, sell_size) / min(buy_size, sell_size)
                if ratio > 5:  # 5:1 dan ortiq nisbat
                    layering_score += 0.1
        
        confidence = min(layering_score, 1.0)
        detected = confidence > 0.3
        
        return {
            'detected': detected,
            'confidence': confidence,
            'score': layering_score
        }
    
    def detect_order_clustering(self, current_order: OrderData) -> Dict[str, Any]:
        """Order clustering patternini aniqlash"""
        if len(self.order_history) < 10:
            return {'detected': False, 'confidence': 0}
        
        recent_orders = list(self.order_history)[-10:]
        
        # Vaqt va narx bo'yicha clustering
        time_clusters = []
        price_clusters = []
        
        for order in recent_orders:
            time_diff = abs(current_order.timestamp - order.timestamp)
            price_diff = abs(current_order.price - order.price) / current_order.price
            
            if time_diff < 30:  # 30 sekund
                time_clusters.append(order)
            
            if price_diff < 0.005:  # 0.5% narx oralig'i
                price_clusters.append(order)
        
        # Clustering score
        time_score = min(len(time_clusters) * 0.1, 1.0)
        price_score = min(len(price_clusters) * 0.1, 1.0)
        
        confidence = (time_score + price_score) / 2
        detected = confidence > 0.3
        
        return {
            'detected': detected,
            'confidence': confidence,
            'time_cluster_size': len(time_clusters),
            'price_cluster_size': len(price_clusters)
        }
    
    def analyze_microstructure(self) -> Dict[str, Any]:
        """Mikrostruktura tahlili"""
        if len(self.order_history) < 50:
            return {'status': 'insufficient_data'}
        
        recent_orders = list(self.order_history)[-50:]
        
        # Bid-ask spread simulation
        buy_prices = [o.price for o in recent_orders if o.side == 'buy']
        sell_prices = [o.price for o in recent_orders if o.side == 'sell']
        
        if not buy_prices or not sell_prices:
            return {'status': 'incomplete_data'}
        
        # Price impact analysis
        avg_buy_price = statistics.mean(buy_prices)
        avg_sell_price = statistics.mean(sell_prices)
        spread = abs(avg_sell_price - avg_buy_price)
        
        # Order size distribution
        order_sizes = [o.size for o in recent_orders]
        avg_order_size = statistics.mean(order_sizes)
        order_size_std = statistics.stdev(order_sizes) if len(order_sizes) > 1 else 0
        
        # Aggressive vs passive ratio
        market_orders = [o for o in recent_orders if o.order_type == 'market']
        aggressive_ratio = len(market_orders) / len(recent_orders)
        
        self.metrics['aggressive_ratio'] = aggressive_ratio
        
        return {
            'status': 'complete',
            'spread': spread,
            'avg_buy_price': avg_buy_price,
            'avg_sell_price': avg_sell_price,
            'avg_order_size': avg_order_size,
            'order_size_volatility': order_size_std,
            'aggressive_ratio': aggressive_ratio,
            'market_depth': len(set(round(o.price, 2) for o in recent_orders))
        }
    
    def generate_signals(self) -> Dict[str, Any]:
        """Order flow asosida signallar generatsiya qilish"""
        signals = {
            'bullish_signals': [],
            'bearish_signals': [],
            'neutral_signals': [],
            'confidence': 0
        }
        
        # Imbalance-based signals
        if self.metrics['order_flow_imbalance'] > 0.3:
            signals['bullish_signals'].append('strong_buy_imbalance')
        elif self.metrics['order_flow_imbalance'] < -0.3:
            signals['bearish_signals'].append('strong_sell_imbalance')
        
        # Whale activity signals
        if self.metrics['whale_activity_score'] > 2:
            signals['bullish_signals'].append('whale_accumulation')
        
        # Aggressive trading signals
        if self.metrics['aggressive_ratio'] > 0.7:
            signals['bullish_signals'].append('aggressive_buying')
        elif self.metrics['aggressive_ratio'] < 0.3:
            signals['bearish_signals'].append('aggressive_selling')
        
        # Pattern-based signals
        if self.pattern_cache.get('momentum_ignition', {}).get('detected', False):
            direction = self.pattern_cache['momentum_ignition'].get('direction', 'neutral')
            if direction == 'buy':
                signals['bullish_signals'].append('momentum_ignition_buy')
            elif direction == 'sell':
                signals['bearish_signals'].append('momentum_ignition_sell')
        
        # Calculate overall confidence
        total_signals = len(signals['bullish_signals']) + len(signals['bearish_signals'])
        if total_signals > 0:
            signals['confidence'] = min(total_signals * 0.2, 1.0)
        
        return signals
    
    def get_order_book_pressure(self) -> Dict[str, Any]:
        """Order book bosimini hisoblash"""
        if len(self.order_history) < 20:
            return {'pressure': 'neutral', 'score': 0}
        
        recent_orders = list(self.order_history)[-20:]
        
        # Volume-weighted pressure
        buy_pressure = sum(o.size for o in recent_orders if o.side == 'buy')
        sell_pressure = sum(o.size for o in recent_orders if o.side == 'sell')
        
        total_pressure = buy_pressure + sell_pressure
        
        if total_pressure == 0:
            return {'pressure': 'neutral', 'score': 0}
        
        pressure_ratio = (buy_pressure - sell_pressure) / total_pressure
        
        if pressure_ratio > 0.2:
            pressure = 'bullish'
        elif pressure_ratio < -0.2:
            pressure = 'bearish'
        else:
            pressure = 'neutral'
        
        return {
            'pressure': pressure,
            'score': pressure_ratio,
            'buy_pressure': buy_pressure,
            'sell_pressure': sell_pressure
        }
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Tahlil xulosasi"""
        return {
            'timestamp': time.time(),
            'total_orders_analyzed': len(self.order_history),
            'metrics': self.metrics.copy(),
            'patterns': self.pattern_cache.copy(),
            'signals': self.generate_signals(),
            'order_book_pressure': self.get_order_book_pressure(),
            'whale_activity': self.metrics['whale_activity_score'],
            'imbalance': self.metrics['order_flow_imbalance'],
            'status': 'active'
        }
    
    def reset_analysis(self):
        """Tahlilni qayta tiklash"""
        self.order_history.clear()
        self.trade_history.clear()
        self.volume_profile.clear()
        self.pattern_cache.clear()
        
        # Metrikalarni qayta tiklash
        for key in self.metrics:
            self.metrics[key] = 0
        
        self.logger.log_info("Order flow analysis reset completed")

# Qo'shimcha utility funksiyalar
def calculate_vwap(orders: List[OrderData]) -> float:
    """Volume Weighted Average Price hisoblash"""
    if not orders:
        return 0.0
    
    total_value = sum(o.price * o.size for o in orders)
    total_volume = sum(o.size for o in orders)
    
    return total_value / total_volume if total_volume > 0 else 0.0

def calculate_order_arrival_rate(orders: List[OrderData], window_seconds: int = 60) -> float:
    """Order kelish tezligini hisoblash"""
    if not orders:
        return 0.0
    
    current_time = time.time()
    recent_orders = [o for o in orders if current_time - o.timestamp <= window_seconds]
    
    return len(recent_orders) / window_seconds

def detect_price_levels(orders: List[OrderData], tolerance: float = 0.001) -> List[Dict[str, Any]]:
    """Muhim narx darajalarini aniqlash"""
    if not orders:
        return []
    
    price_volume = defaultdict(float)
    
    for order in orders:
        price_level = round(order.price / (1 + tolerance)) * (1 + tolerance)
        price_volume[price_level] += order.size
    
    # Eng ko'p volume bo'lgan narx darajalarini topish
    sorted_levels = sorted(price_volume.items(), key=lambda x: x[1], reverse=True)
    
    return [
        {
            'price': price,
            'volume': volume,
            'significance': volume / max(price_volume.values())
        }
        for price, volume in sorted_levels[:10]
    ]
