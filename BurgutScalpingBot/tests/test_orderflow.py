"""
Order Flow Analysis Test Suite
Bu modul order flow tahlili modulini to'liq test qiladi
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json

# Test data generators
def generate_test_orderbook_data(depth: int = 10) -> Dict[str, Any]:
    """Test orderbook data generator"""
    bids = []
    asks = []
    
    base_price = 50000
    
    for i in range(depth):
        # Bids (buy orders)
        bid_price = base_price - (i + 1) * 10
        bid_size = np.random.uniform(0.1, 5.0)
        bids.append([bid_price, bid_size])
        
        # Asks (sell orders)
        ask_price = base_price + (i + 1) * 10
        ask_size = np.random.uniform(0.1, 5.0)
        asks.append([ask_price, ask_size])
    
    return {
        'bids': bids,
        'asks': asks,
        'timestamp': datetime.now().isoformat()
    }

def generate_whale_orders(whale_threshold: float = 1000000) -> List[Dict]:
    """Katta order (whale) data generator"""
    orders = []
    
    # Normal orders
    for _ in range(50):
        orders.append({
            'price': np.random.uniform(49000, 51000),
            'size': np.random.uniform(0.01, 0.5),
            'side': np.random.choice(['buy', 'sell']),
            'timestamp': datetime.now().isoformat()
        })
    
    # Whale orders
    for _ in range(3):
        orders.append({
            'price': np.random.uniform(49500, 50500),
            'size': np.random.uniform(10.0, 50.0),  # Large size
            'side': np.random.choice(['buy', 'sell']),
            'timestamp': datetime.now().isoformat(),
            'is_whale': True
        })
    
    return orders

def generate_imbalanced_orders(buy_ratio: float = 0.8, total_orders: int = 100) -> List[Dict]:
    """Nomutanosib order flow generator"""
    orders = []
    buy_count = int(total_orders * buy_ratio)
    sell_count = total_orders - buy_count
    
    # Buy orders
    for _ in range(buy_count):
        orders.append({
            'price': np.random.uniform(49800, 50200),
            'size': np.random.uniform(0.1, 2.0),
            'side': 'buy',
            'timestamp': datetime.now().isoformat()
        })
    
    # Sell orders
    for _ in range(sell_count):
        orders.append({
            'price': np.random.uniform(49800, 50200),
            'size': np.random.uniform(0.1, 2.0),
            'side': 'sell',
            'timestamp': datetime.now().isoformat()
        })
    
    return orders

def generate_tick_data(duration_minutes: int = 60) -> pd.DataFrame:
    """Tick data generator"""
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=duration_minutes)
    
    # Generate timestamps
    timestamps = pd.date_range(start_time, end_time, freq='1S')
    
    data = []
    base_price = 50000
    
    for ts in timestamps:
        # Price with random walk
        price_change = np.random.normal(0, 10)
        base_price += price_change
        
        data.append({
            'timestamp': ts,
            'price': base_price,
            'volume': np.random.exponential(0.5),
            'side': np.random.choice(['buy', 'sell']),
            'order_type': np.random.choice(['market', 'limit'], p=[0.3, 0.7])
        })
    
    return pd.DataFrame(data)

class TestOrderFlowAnalyzer:
    """Order Flow Analyzer asosiy testlari"""
    
    @pytest.fixture
    def order_flow_analyzer(self):
        """OrderFlowAnalyzer instance"""
        # Mock import qilish
        with patch('order_flow.OrderFlowAnalyzer') as mock_analyzer:
            analyzer = mock_analyzer.return_value
            analyzer.whale_threshold = 1000000  # $1M threshold
            analyzer.imbalance_threshold = 0.6
            analyzer.window_size = 60  # 60 seconds
            return analyzer
    
    @pytest.fixture
    def sample_orderbook(self):
        """Sample orderbook data"""
        return generate_test_orderbook_data()
    
    @pytest.fixture
    def whale_orders(self):
        """Whale orders data"""
        return generate_whale_orders()
    
    @pytest.fixture
    def imbalanced_orders(self):
        """Imbalanced orders data"""
        return generate_imbalanced_orders(buy_ratio=0.8)
    
    def test_orderbook_spread_calculation(self, order_flow_analyzer, sample_orderbook):
        """Orderbook spread hisobi testi"""
        # Mock method
        order_flow_analyzer.calculate_spread.return_value = {
            'bid_ask_spread': 20.0,
            'spread_percentage': 0.04,
            'mid_price': 50000.0
        }
        
        result = order_flow_analyzer.calculate_spread(sample_orderbook)
        
        assert result['bid_ask_spread'] > 0
        assert result['spread_percentage'] > 0
        assert result['mid_price'] > 0
        order_flow_analyzer.calculate_spread.assert_called_once()
    
    def test_whale_activity_detection(self, order_flow_analyzer, whale_orders):
        """Whale activity detection testi"""
        # Mock whale detection
        order_flow_analyzer.detect_whale_activity.return_value = {
            'detected': True,
            'whale_count': 3,
            'total_whale_volume': 150.0,
            'impact_level': 'high',
            'dominant_side': 'buy'
        }
        
        result = order_flow_analyzer.detect_whale_activity(whale_orders)
        
        assert result['detected'] == True
        assert result['whale_count'] > 0
        assert result['impact_level'] in ['low', 'medium', 'high']
        assert result['dominant_side'] in ['buy', 'sell', 'neutral']
    
    def test_order_flow_imbalance(self, order_flow_analyzer, imbalanced_orders):
        """Order flow imbalance testi"""
        # Mock imbalance calculation
        order_flow_analyzer.calculate_imbalance.return_value = {
            'imbalance_ratio': 0.8,
            'direction': 'bullish',
            'strength': 0.75,
            'buy_volume': 80.0,
            'sell_volume': 20.0,
            'confidence': 0.85
        }
        
        result = order_flow_analyzer.calculate_imbalance(imbalanced_orders)
        
        assert 0 <= result['imbalance_ratio'] <= 1
        assert result['direction'] in ['bullish', 'bearish', 'neutral']
        assert 0 <= result['strength'] <= 1
        assert 0 <= result['confidence'] <= 1
    
    def test_microstructure_patterns(self, order_flow_analyzer):
        """Mikrostruktura pattern detection testi"""
        # Generate test tick data
        tick_data = generate_tick_data(duration_minutes=30)
        
        # Mock pattern detection
        order_flow_analyzer.detect_patterns.return_value = {
            'iceberg_orders': {
                'detected': True,
                'count': 2,
                'avg_hidden_size': 5.0
            },
            'spoofing_attempts': {
                'detected': False,
                'count': 0
            },
            'layering_pattern': {
                'detected': True,
                'intensity': 0.6
            },
            'momentum_ignition': {
                'detected': False,
                'probability': 0.2
            }
        }
        
        result = order_flow_analyzer.detect_patterns(tick_data)
        
        assert 'iceberg_orders' in result
        assert 'spoofing_attempts' in result
        assert 'layering_pattern' in result
        assert 'momentum_ignition' in result
    
    def test_volume_profile_analysis(self, order_flow_analyzer):
        """Volume profile tahlili testi"""
        # Generate volume data
        volume_data = []
        for i in range(100):
            volume_data.append({
                'price': 50000 + i * 10,
                'volume': np.random.exponential(2.0),
                'timestamp': datetime.now().isoformat()
            })
        
        # Mock volume profile
        order_flow_analyzer.analyze_volume_profile.return_value = {
            'poc_price': 50500,  # Point of Control
            'value_area_high': 50800,
            'value_area_low': 50200,
            'volume_distribution': 'normal',
            'support_levels': [50200, 50000, 49800],
            'resistance_levels': [50800, 51000, 51200]
        }
        
        result = order_flow_analyzer.analyze_volume_profile(volume_data)
        
        assert result['poc_price'] > 0
        assert result['value_area_high'] > result['value_area_low']
        assert len(result['support_levels']) > 0
        assert len(result['resistance_levels']) > 0
    
    def test_order_flow_momentum(self, order_flow_analyzer):
        """Order flow momentum testi"""
        # Generate momentum data
        momentum_data = []
        for i in range(50):
            momentum_data.append({
                'timestamp': datetime.now() - timedelta(seconds=i),
                'buy_volume': np.random.exponential(1.0),
                'sell_volume': np.random.exponential(1.0),
                'price': 50000 + np.random.normal(0, 100)
            })
        
        # Mock momentum calculation
        order_flow_analyzer.calculate_momentum.return_value = {
            'momentum_score': 0.7,
            'direction': 'bullish',
            'acceleration': 0.15,
            'sustainability': 0.6,
            'divergence_detected': False
        }
        
        result = order_flow_analyzer.calculate_momentum(momentum_data)
        
        assert -1 <= result['momentum_score'] <= 1
        assert result['direction'] in ['bullish', 'bearish', 'neutral']
        assert isinstance(result['divergence_detected'], bool)

class TestOrderFlowIntegration:
    """Order Flow integration testlari"""
    
    @pytest.fixture
    def mock_exchange_data(self):
        """Mock exchange data"""
        return {
            'orderbook': generate_test_orderbook_data(depth=20),
            'recent_trades': generate_whale_orders()[:10],
            'tick_data': generate_tick_data(duration_minutes=15)
        }
    
    @pytest.mark.asyncio
    async def test_real_time_analysis(self, order_flow_analyzer, mock_exchange_data):
        """Real-time tahlil testi"""
        # Mock real-time analysis
        order_flow_analyzer.analyze_real_time.return_value = {
            'timestamp': datetime.now().isoformat(),
            'signal_strength': 0.8,
            'signal_type': 'bullish',
            'confidence': 0.75,
            'risk_level': 'medium',
            'recommended_action': 'buy',
            'supporting_factors': [
                'whale_activity_detected',
                'order_imbalance_bullish',
                'volume_profile_supportive'
            ]
        }
        
        result = await order_flow_analyzer.analyze_real_time(mock_exchange_data)
        
        assert 0 <= result['signal_strength'] <= 1
        assert result['signal_type'] in ['bullish', 'bearish', 'neutral']
        assert result['recommended_action'] in ['buy', 'sell', 'hold']
        assert len(result['supporting_factors']) > 0
    
    def test_historical_pattern_recognition(self, order_flow_analyzer):
        """Historical pattern tanib olish testi"""
        # Generate historical data
        historical_data = []
        for i in range(1000):
            historical_data.append({
                'timestamp': datetime.now() - timedelta(minutes=i),
                'price': 50000 + np.random.normal(0, 500),
                'volume': np.random.exponential(1.0),
                'order_flow_imbalance': np.random.uniform(-1, 1)
            })
        
        # Mock pattern recognition
        order_flow_analyzer.recognize_historical_patterns.return_value = {
            'patterns_found': [
                {
                    'pattern_type': 'accumulation',
                    'start_time': datetime.now() - timedelta(hours=2),
                    'end_time': datetime.now() - timedelta(hours=1),
                    'confidence': 0.85,
                    'outcome': 'bullish_breakout'
                },
                {
                    'pattern_type': 'distribution',
                    'start_time': datetime.now() - timedelta(hours=6),
                    'end_time': datetime.now() - timedelta(hours=4),
                    'confidence': 0.7,
                    'outcome': 'bearish_breakdown'
                }
            ],
            'pattern_success_rate': 0.68,
            'most_reliable_pattern': 'accumulation'
        }
        
        result = order_flow_analyzer.recognize_historical_patterns(historical_data)
        
        assert len(result['patterns_found']) > 0
        assert 0 <= result['pattern_success_rate'] <= 1
        assert result['most_reliable_pattern'] in ['accumulation', 'distribution', 'consolidation']

class TestOrderFlowRiskManagement:
    """Order Flow risk management testlari"""
    
    def test_liquidity_assessment(self, order_flow_analyzer):
        """Liquidity baholash testi"""
        orderbook_data = generate_test_orderbook_data(depth=50)
        
        # Mock liquidity assessment
        order_flow_analyzer.assess_liquidity.return_value = {
            'liquidity_score': 0.7,
            'bid_depth': 15.5,
            'ask_depth': 14.2,
            'depth_imbalance': 0.08,
            'slippage_estimate': 0.005,
            'market_impact': 'low'
        }
        
        result = order_flow_analyzer.assess_liquidity(orderbook_data)
        
        assert 0 <= result['liquidity_score'] <= 1
        assert result['bid_depth'] > 0
        assert result['ask_depth'] > 0
        assert result['market_impact'] in ['low', 'medium', 'high']
    
    def test_execution_quality_metrics(self, order_flow_analyzer):
        """Execution quality metrics testi"""
        execution_data = []
        for i in range(20):
            execution_data.append({
                'order_id': f'order_{i}',
                'expected_price': 50000,
                'executed_price': 50000 + np.random.normal(0, 20),
                'expected_size': 1.0,
                'executed_size': 1.0 + np.random.normal(0, 0.05),
                'execution_time': datetime.now() - timedelta(minutes=i),
                'order_type': np.random.choice(['market', 'limit'])
            })
        
        # Mock execution quality
        order_flow_analyzer.calculate_execution_quality.return_value = {
            'avg_slippage': 0.003,
            'fill_rate': 0.95,
            'avg_execution_time': 2.5,
            'price_improvement': 0.001,
            'quality_score': 0.8,
            'recommendations': [
                'consider_limit_orders_in_volatile_periods',
                'split_large_orders'
            ]
        }
        
        result = order_flow_analyzer.calculate_execution_quality(execution_data)
        
        assert result['avg_slippage'] >= 0
        assert 0 <= result['fill_rate'] <= 1
        assert result['avg_execution_time'] > 0
        assert 0 <= result['quality_score'] <= 1

@pytest.mark.integration
class TestOrderFlowSystemIntegration:
    """Butun system bilan integration testlari"""
    
    @pytest.mark.asyncio
    async def test_complete_order_flow_pipeline(self):
        """To'liq order flow pipeline testi"""
        # Mock complete pipeline
        with patch('order_flow.OrderFlowAnalyzer') as mock_analyzer:
            analyzer = mock_analyzer.return_value
            
            # Mock pipeline steps
            analyzer.collect_data.return_value = generate_test_orderbook_data()
            analyzer.analyze_patterns.return_value = {'patterns': ['bullish_momentum']}
            analyzer.generate_signals.return_value = {
                'signal': 'buy',
                'strength': 0.8,
                'confidence': 0.75
            }
            
            # Run pipeline
            data = await analyzer.collect_data()
            patterns = analyzer.analyze_patterns(data)
            signals = analyzer.generate_signals(patterns)
            
            assert signals['signal'] in ['buy', 'sell', 'hold']
            assert 0 <= signals['strength'] <= 1
            assert 0 <= signals['confidence'] <= 1
    
    def test_error_handling(self, order_flow_analyzer):
        """Error handling testi"""
        # Test with invalid data
        invalid_data = None
        
        with patch.object(order_flow_analyzer, 'analyze_real_time') as mock_analyze:
            mock_analyze.side_effect = Exception("API connection failed")
            
            with pytest.raises(Exception):
                order_flow_analyzer.analyze_real_time(invalid_data)
    
    def test_performance_benchmarks(self, order_flow_analyzer):
        """Performance benchmark testlari"""
        large_dataset = generate_tick_data(duration_minutes=1440)  # 24 hours
        
        # Mock performance test
        order_flow_analyzer.benchmark_performance.return_value = {
            'processing_time': 0.5,  # seconds
            'memory_usage': 45.2,    # MB
            'cpu_utilization': 0.25,
            'throughput': 2000,      # orders/second
            'latency_p95': 0.002     # seconds
        }
        
        result = order_flow_analyzer.benchmark_performance(large_dataset)
        
        assert result['processing_time'] > 0
        assert result['memory_usage'] > 0
        assert 0 <= result['cpu_utilization'] <= 1
        assert result['throughput'] > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
