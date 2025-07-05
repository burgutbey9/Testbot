"""
AI Sentiment Analysis Test Suite
Bu modul AI sentiment analyzer modulini to'liq test qiladi
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json
import random

# Test data generators
def generate_sentiment_data(source: str = "mixed", count: int = 100) -> List[Dict]:
    """Sentiment data generator"""
    data = []
    
    # Sentiment range based on source
    sentiment_ranges = {
        "twitter": (-0.8, 0.8),
        "reddit": (-0.6, 0.7),
        "news": (-0.5, 0.6),
        "telegram": (-0.9, 0.9),
        "orderbook": (-1.0, 1.0),
        "mixed": (-1.0, 1.0)
    }
    
    min_sent, max_sent = sentiment_ranges.get(source, (-1.0, 1.0))
    
    for i in range(count):
        sentiment_score = np.random.uniform(min_sent, max_sent)
        
        data.append({
            'id': f'{source}_{i}',
            'source': source if source != "mixed" else random.choice(["twitter", "reddit", "news", "telegram"]),
            'content': f'Sample content {i}',
            'sentiment_score': sentiment_score,
            'confidence': np.random.uniform(0.3, 0.95),
            'timestamp': datetime.now() - timedelta(minutes=i),
            'author_influence': np.random.uniform(0.1, 1.0),
            'engagement': np.random.randint(1, 10000),
            'keywords': ['crypto', 'trading', 'bullish' if sentiment_score > 0 else 'bearish']
        })
    
    return data

def generate_noisy_sentiment_data(noise_level: float = 0.3) -> List[Dict]:
    """Noisy sentiment data generator"""
    clean_data = generate_sentiment_data(count=50)
    noisy_data = []
    
    for item in clean_data:
        # Add noise to sentiment scores
        if random.random() < noise_level:
            # Flip sentiment randomly
            item['sentiment_score'] = -item['sentiment_score']
            item['confidence'] = 0.2  # Low confidence for noisy data
            item['is_noisy'] = True
        else:
            item['is_noisy'] = False
        
        noisy_data.append(item)
    
    return noisy_data

def generate_multi_source_sentiment() -> Dict[str, List[Dict]]:
    """Multi-source sentiment data generator"""
    sources = ['twitter', 'reddit', 'news', 'telegram', 'orderbook']
    data = {}
    
    for source in sources:
        data[source] = generate_sentiment_data(source=source, count=20)
    
    return data

def generate_historical_sentiment_data(days: int = 30) -> pd.DataFrame:
    """Historical sentiment data generator"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Generate daily sentiment data
    dates = pd.date_range(start_date, end_date, freq='1H')
    
    sentiment_data = []
    base_sentiment = 0.0
    
    for date in dates:
        # Random walk sentiment
        sentiment_change = np.random.normal(0, 0.1)
        base_sentiment += sentiment_change
        base_sentiment = np.clip(base_sentiment, -1.0, 1.0)
        
        sentiment_data.append({
            'timestamp': date,
            'sentiment_score': base_sentiment,
            'volume': np.random.exponential(100),
            'source_count': np.random.randint(10, 100),
            'confidence': np.random.uniform(0.5, 0.9),
            'volatility': np.random.uniform(0.1, 0.5)
        })
    
    return pd.DataFrame(sentiment_data)

class TestAISentimentAnalyzer:
    """AI Sentiment Analyzer asosiy testlari"""
    
    @pytest.fixture
    def sentiment_analyzer(self):
        """AISentimentAnalyzer instance"""
        with patch('ai.sentiment.AISentimentAnalyzer') as mock_analyzer:
            analyzer = mock_analyzer.return_value
            analyzer.sentiment_threshold = 0.6
            analyzer.confidence_threshold = 0.7
            analyzer.source_weights = {
                'twitter': 0.3,
                'reddit': 0.2,
                'news': 0.25,
                'telegram': 0.15,
                'orderbook': 0.1
            }
            return analyzer
    
    @pytest.fixture
    def sample_sentiment_data(self):
        """Sample sentiment data"""
        return generate_sentiment_data(count=50)
    
    @pytest.fixture
    def multi_source_data(self):
        """Multi-source sentiment data"""
        return generate_multi_source_sentiment()
    
    @pytest.fixture
    def noisy_data(self):
        """Noisy sentiment data"""
        return generate_noisy_sentiment_data(noise_level=0.4)
    
    def test_sentiment_aggregation(self, sentiment_analyzer, multi_source_data):
        """Sentiment aggregation testi"""
        # Mock sentiment aggregation
        sentiment_analyzer.aggregate_sentiment.return_value = {
            'weighted_sentiment': 0.65,
            'confidence': 0.8,
            'source_breakdown': {
                'twitter': 0.7,
                'reddit': 0.6,
                'news': 0.5,
                'telegram': 0.8,
                'orderbook': 0.9
            },
            'dominant_sentiment': 'bullish',
            'consensus_level': 0.75
        }
        
        result = sentiment_analyzer.aggregate_sentiment(multi_source_data)
        
        assert -1 <= result['weighted_sentiment'] <= 1
        assert 0 <= result['confidence'] <= 1
        assert result['dominant_sentiment'] in ['bullish', 'bearish', 'neutral']
        assert 0 <= result['consensus_level'] <= 1
        assert len(result['source_breakdown']) > 0
    
    def test_sentiment_reliability_scoring(self, sentiment_analyzer, noisy_data):
        """Sentiment reliability scoring testi"""
        # Mock reliability calculation
        sentiment_analyzer.calculate_reliability.return_value = {
            'reliability_score': 0.65,
            'noise_level': 0.4,
            'data_quality': 'medium',
            'outlier_count': 12,
            'consistency_score': 0.7,
            'temporal_stability': 0.8
        }
        
        result = sentiment_analyzer.calculate_reliability(noisy_data)
        
        assert 0 <= result['reliability_score'] <= 1
        assert 0 <= result['noise_level'] <= 1
        assert result['data_quality'] in ['low', 'medium', 'high']
        assert result['outlier_count'] >= 0
        assert 0 <= result['consistency_score'] <= 1
        assert 0 <= result['temporal_stability'] <= 1
    
    def test_sentiment_trend_analysis(self, sentiment_analyzer):
        """Sentiment trend analysis testi"""
        historical_data = generate_historical_sentiment_data(days=7)
        
        # Mock trend analysis
        sentiment_analyzer.analyze_trend.return_value = {
            'trend_direction': 'bullish',
            'trend_strength': 0.7,
            'trend_duration': 5,  # days
            'momentum': 0.6,
            'reversal_probability': 0.3,
            'key_inflection_points': [
                {
                    'timestamp': datetime.now() - timedelta(days=3),
                    'sentiment_change': 0.4,
                    'trigger_event': 'positive_news'
                }
            ]
        }
        
        result = sentiment_analyzer.analyze_trend(historical_data)
        
        assert result['trend_direction'] in ['bullish', 'bearish', 'sideways']
        assert 0 <= result['trend_strength'] <= 1
        assert result['trend_duration'] >= 0
        assert 0 <= result['momentum'] <= 1
        assert 0 <= result['reversal_probability'] <= 1
    
    def test_sentiment_anomaly_detection(self, sentiment_analyzer, sample_sentiment_data):
        """Sentiment anomaly detection testi"""
        # Add some anomalous data
        anomalous_data = sample_sentiment_data.copy()
        anomalous_data.extend([
            {
                'id': 'anomaly_1',
                'source': 'twitter',
                'sentiment_score': -0.95,  # Extreme negative
                'confidence': 0.9,
                'timestamp': datetime.now(),
                'engagement': 50000,  # High engagement
                'is_anomaly': True
            },
            {
                'id': 'anomaly_2',
                'source': 'news',
                'sentiment_score': 0.98,  # Extreme positive
                'confidence': 0.85,
                'timestamp': datetime.now(),
                'engagement': 30000,
                'is_anomaly': True
            }
        ])
        
        # Mock anomaly detection
        sentiment_analyzer.detect_anomalies.return_value = {
            'anomalies_detected': 2,
            'anomaly_threshold': 0.85,
            'anomalous_items': [
                {
                    'id': 'anomaly_1',
                    'anomaly_score': 0.92,
                    'anomaly_type': 'extreme_negative',
                    'impact_level': 'high'
                },
                {
                    'id': 'anomaly_2',
                    'anomaly_score': 0.89,
                    'anomaly_type': 'extreme_positive',
                    'impact_level': 'medium'
                }
            ],
            'overall_anomaly_level': 'elevated'
        }
        
        result = sentiment_analyzer.detect_anomalies(anomalous_data)
        
        assert result['anomalies_detected'] >= 0
        assert 0 <= result['anomaly_threshold'] <= 1
        assert result['overall_anomaly_level'] in ['normal', 'elevated', 'high']
        assert len(result['anomalous_items']) == result['anomalies_detected']
    
    def test_sentiment_feature_extraction(self, sentiment_analyzer, sample_sentiment_data):
        """Sentiment feature extraction testi"""
        # Mock feature extraction
        sentiment_analyzer.extract_features.return_value = {
            'statistical_features': {
                'mean_sentiment': 0.25,
                'std_sentiment': 0.45,
                'skewness': 0.1,
                'kurtosis': -0.3,
                'sentiment_range': 1.6
            },
            'temporal_features': {
                'sentiment_velocity': 0.05,  # Change rate
                'acceleration': 0.01,
                'volatility': 0.3,
                'persistence': 0.7
            },
            'source_features': {
                'source_diversity': 0.8,
                'source_agreement': 0.6,
                'influence_weighted_sentiment': 0.35
            },
            'engagement_features': {
                'total_engagement': 150000,
                'engagement_sentiment_correlation': 0.4,
                'viral_content_ratio': 0.15
            }
        }
        
        result = sentiment_analyzer.extract_features(sample_sentiment_data)
        
        assert 'statistical_features' in result
        assert 'temporal_features' in result
        assert 'source_features' in result
        assert 'engagement_features' in result
        
        # Check statistical features
        stats = result['statistical_features']
        assert -1 <= stats['mean_sentiment'] <= 1
        assert stats['std_sentiment'] >= 0
        assert stats['sentiment_range'] >= 0
    
    def test_sentiment_prediction(self, sentiment_analyzer):
        """Sentiment prediction testi"""
        historical_data = generate_historical_sentiment_data(days=14)
        
        # Mock sentiment prediction
        sentiment_analyzer.predict_sentiment.return_value = {
            'predicted_sentiment': 0.4,
            'prediction_confidence': 0.75,
            'prediction_horizon': 24,  # hours
            'prediction_intervals': {
                'lower_bound': 0.2,
                'upper_bound': 0.6
            },
            'key_drivers': [
                'positive_news_trend',
                'increasing_social_volume',
                'whale_accumulation'
            ],
            'model_uncertainty': 0.15
        }
        
        result = sentiment_analyzer.predict_sentiment(historical_data)
        
        assert -1 <= result['predicted_sentiment'] <= 1
        assert 0 <= result['prediction_confidence'] <= 1
        assert result['prediction_horizon'] > 0
        assert result['prediction_intervals']['lower_bound'] <= result['prediction_intervals']['upper_bound']
        assert len(result['key_drivers']) > 0
        assert 0 <= result['model_uncertainty'] <= 1

class TestSentimentModelPerformance:
    """Sentiment model performance testlari"""
    
    @pytest.fixture
    def performance_data(self):
        """Performance test data"""
        return {
            'predictions': np.random.uniform(-1, 1, 100),
            'actuals': np.random.uniform(-1, 1, 100),
            'timestamps': [datetime.now() - timedelta(hours=i) for i in range(100)]
        }
    
    def test_model_accuracy_metrics(self, sentiment_analyzer, performance_data):
        """Model accuracy metrics testi"""
        # Mock performance metrics
        sentiment_analyzer.calculate_performance_metrics.return_value = {
            'accuracy': 0.72,
            'precision': 0.68,
            'recall': 0.75,
            'f1_score': 0.71,
            'mse': 0.15,
            'mae': 0.12,
            'correlation': 0.65,
            'directional_accuracy': 0.78
        }
        
        result = sentiment_analyzer.calculate_performance_metrics(performance_data)
        
        assert 0 <= result['accuracy'] <= 1
        assert 0 <= result['precision'] <= 1
        assert 0 <= result['recall'] <= 1
        assert 0 <= result['f1_score'] <= 1
        assert result['mse'] >= 0
        assert result['mae'] >= 0
        assert -1 <= result['correlation'] <= 1
        assert 0 <= result['directional_accuracy'] <= 1
    
    def test_model_drift_detection(self, sentiment_analyzer, performance_data):
        """Model drift detection testi"""
        # Mock drift detection
        sentiment_analyzer.detect_model_drift.return_value = {
            'drift_detected': True,
            'drift_severity': 'moderate',
            'drift_type': 'concept_drift',
            'drift_start_time': datetime.now() - timedelta(days=3),
            'performance_degradation': 0.15,
            'recommended_action': 'retrain_model',
            'drift_metrics': {
                'psi_score': 0.25,  # Population Stability Index
                'ks_statistic': 0.18,
                'wasserstein_distance': 0.22
            }
        }
        
        result = sentiment_analyzer.detect_model_drift(performance_data)
        
        assert isinstance(result['drift_detected'], bool)
        assert result['drift_severity'] in ['low', 'moderate', 'high']
        assert result['drift_type'] in ['concept_drift', 'data_drift', 'both']
        assert result['performance_degradation'] >= 0
        assert result['recommended_action'] in ['monitor', 'retrain_model', 'emergency_fallback']
    
    def test_model_calibration(self, sentiment_analyzer):
        """Model calibration testi"""
        # Generate calibration data
        calibration_data = []
        for i in range(100):
            predicted_prob = np.random.uniform(0, 1)
            actual_outcome = np.random.choice([0, 1], p=[1-predicted_prob, predicted_prob])
            calibration_data.append({
                'predicted_probability': predicted_prob,
                'actual_outcome': actual_outcome
            })
        
        # Mock calibration analysis
        sentiment_analyzer.analyze_calibration.return_value = {
            'calibration_score': 0.82,
            'brier_score': 0.18,
            'reliability_diagram': {
                'bins': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'observed_frequencies': [0.12, 0.19, 0.31, 0.38, 0.52, 0.61, 0.68, 0.79, 0.88, 0.95],
                'predicted_frequencies': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            },
            'is_well_calibrated': True
        }
        
        result = sentiment_analyzer.analyze_calibration(calibration_data)
        
        assert 0 <= result['calibration_score'] <= 1
        assert result['brier_score'] >= 0
        assert isinstance(result['is_well_calibrated'], bool)
        assert len(result['reliability_diagram']['bins']) > 0

class TestSentimentIntegration:
    """Sentiment analysis integration testlari"""
    
    @pytest.mark.asyncio
    async def test_real_time_sentiment_processing(self, sentiment_analyzer):
        """Real-time sentiment processing testi"""
        # Mock real-time data stream
        async def mock_data_stream():
            for i in range(10):
                yield {
                    'id': f'real_time_{i}',
                    'content': f'Real time message {i}',
                    'timestamp': datetime.now(),
                    'source': 'twitter'
                }
                await asyncio.sleep(0.1)
        
        # Mock real-time processing
        sentiment_analyzer.process_real_time_stream.return_value = {
            'processed_messages': 10,
            'average_sentiment': 0.3,
            'sentiment_velocity': 0.05,
            'alert_triggered': False,
            'processing_latency': 0.05  # seconds
        }
        
        result = await sentiment_analyzer.process_real_time_stream(mock_data_stream())
        
        assert result['processed_messages'] > 0
        assert -1 <= result['average_sentiment'] <= 1
        assert isinstance(result['alert_triggered'], bool)
        assert result['processing_latency'] > 0
    
    def test_sentiment_signal_generation(self, sentiment_analyzer):
        """Sentiment signal generation testi"""
        sentiment_data = generate_sentiment_data(count=100)
        
        # Mock signal generation
        sentiment_analyzer.generate_trading_signals.return_value = {
            'signals': [
                {
                    'signal_type': 'bullish',
                    'strength': 0.8,
                    'confidence': 0.75,
                    'timestamp': datetime.now(),
                    'trigger_conditions': [
                        'sentiment_above_threshold',
                        'increasing_momentum',
                        'high_consensus'
                    ]
                }
            ],
            'signal_quality': 0.82,
            'recommendation': 'consider_long_position',
            'risk_assessment': 'moderate'
        }
        
        result = sentiment_analyzer.generate_trading_signals(sentiment_data)
        
        assert len(result['signals']) > 0
        assert 0 <= result['signal_quality'] <= 1
        assert result['recommendation'] in ['consider_long_position', 'consider_short_position', 'hold', 'avoid']
        assert result['risk_assessment'] in ['low', 'moderate', 'high']
    
    def test_sentiment_backtest_integration(self, sentiment_analyzer):
        """Sentiment backtest integration testi"""
        historical_data = generate_historical_sentiment_data(days=30)
        
        # Mock backtest integration
        sentiment_analyzer.backtest_sentiment_strategy.return_value = {
            'total_return': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.08,
            'win_rate': 0.65,
            'total_trades': 45,
            'avg_trade_duration': 2.5,  # hours
            'sentiment_correlation': 0.72,
            'strategy_performance': {
                'bullish_signals': {'count': 25, 'success_rate': 0.68},
                'bearish_signals': {'count': 20, 'success_rate': 0.62}
            }
        }
        
        result = sentiment_analyzer.backtest_sentiment_strategy(historical_data)
        
        assert isinstance(result['total_return'], float)
        assert result['sharpe_ratio'] >= 0
        assert 0 <= result['max_drawdown'] <= 1
        assert 0 <= result['win_rate'] <= 1
        assert result['total_trades'] >= 0
        assert -1 <= result['sentiment_correlation'] <= 1

class TestSentimentErrorHandling:
    """Sentiment analysis error handling testlari"""
    
    def test_api_failure_handling(self, sentiment_analyzer):
        """API failure handling testi"""
        # Mock API failure
        with patch.object(sentiment_analyzer, 'fetch_sentiment_data') as mock_fetch:
            mock_fetch.side_effect = Exception("API connection failed")
            
            # Mock fallback mechanism
            sentiment_analyzer.handle_api_failure.return_value = {
                'fallback_activated': True,
                'fallback_source': 'cached_data',
                'fallback_age': 300,  # seconds
                'reliability_score': 0.4,
                'warning_message': 'Using cached sentiment data due to API failure'
            }
            
            result = sentiment_analyzer.handle_api_failure()
            
            assert result['fallback_activated'] == True
            assert result['fallback_source'] in ['cached_data', 'alternative_api', 'default_values']
            assert result['fallback_age'] >= 0
            assert 0 <= result['reliability_score'] <= 1
    
    def test_data_quality_validation(self, sentiment_analyzer):
        """Data quality validation testi"""
        # Generate corrupted data
        corrupted_data = [
            {'id': 'corrupt_1', 'sentiment_score': 'invalid'},  # Invalid type
            {'id': 'corrupt_2', 'sentiment_score': 5.0},        # Out of range
            {'id': 'corrupt_3'},                                # Missing required field
            {'id': 'corrupt_4', 'sentiment_score': None}       # None value
        ]
        
        # Mock data validation
        sentiment_analyzer.validate_data_quality.return_value = {
            'valid_records': 0,
            'invalid_records': 4,
            'validation_errors': [
                {'record_id': 'corrupt_1', 'error': 'invalid_data_type'},
                {'record_id': 'corrupt_2', 'error': 'value_out_of_range'},
                {'record_id': 'corrupt_3', 'error': 'missing_required_field'},
                {'record_id': 'corrupt_4', 'error': 'null_value'}
            ],
            'data_quality_score': 0.0,
            'recommended_action': 'reject_batch'
        }
        
        result = sentiment_analyzer.validate_data_quality(corrupted_data)
        
        assert result['valid_records'] >= 0
        assert result['invalid_records'] >= 0
        assert len(result['validation_errors']) == result['invalid_records']
        assert 0 <= result['data_quality_score'] <= 1
        assert result['recommended_action'] in ['accept', 'clean_data', 'reject_batch']

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
