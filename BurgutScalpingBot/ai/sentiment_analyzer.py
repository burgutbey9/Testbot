"""
AI Sentiment Analyzer - Kengaytirilgan sentiment tahlili
Turli manbalardan sentiment ma'lumotlarini yig'ib, ishonchli tahlil qiladi
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import deque
import re

from utils.rate_limiter import RateLimiter
from utils.advanced_logger import TradingLogger

class SentimentSource(Enum):
    TWITTER = "twitter"
    REDDIT = "reddit"
    NEWS = "news"
    ORDERBOOK = "orderbook"
    SOCIAL_VOLUME = "social_volume"
    WHALE_ALERTS = "whale_alerts"

@dataclass
class SentimentData:
    source: SentimentSource
    score: float  # -1 to 1
    confidence: float  # 0 to 1
    timestamp: datetime
    volume: int
    keywords: List[str]
    raw_data: Dict

@dataclass
class AggregatedSentiment:
    overall_score: float
    confidence: float
    reliability: float
    sources_count: int
    timestamp: datetime
    breakdown: Dict[str, float]
    trend: str  # 'bullish', 'bearish', 'neutral'

class SentimentAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = TradingLogger()
        self.rate_limiter = RateLimiter(max_calls=100, window=60)
        
        # Sentiment history
        self.sentiment_history = deque(maxlen=1000)
        self.source_weights = {
            SentimentSource.ORDERBOOK: 0.3,
            SentimentSource.WHALE_ALERTS: 0.25,
            SentimentSource.TWITTER: 0.2,
            SentimentSource.NEWS: 0.15,
            SentimentSource.REDDIT: 0.1,
            SentimentSource.SOCIAL_VOLUME: 0.05
        }
        
        # Performance tracking
        self.source_performance = {}
        self.last_analysis_time = datetime.now()
        
        # Keywords for crypto sentiment
        self.bullish_keywords = [
            'moon', 'bull', 'pump', 'hodl', 'buy', 'rally', 'breakout',
            'support', 'strong', 'bullish', 'up', 'rise', 'rocket'
        ]
        
        self.bearish_keywords = [
            'crash', 'dump', 'bear', 'sell', 'drop', 'fall', 'bearish',
            'resistance', 'weak', 'down', 'decline', 'correction'
        ]
        
        self.initialize_sources()
    
    def initialize_sources(self):
        """Sentiment manbalarini ishga tushirish"""
        try:
            # Initialize source performance tracking
            for source in SentimentSource:
                self.source_performance[source] = {
                    'accuracy': 0.5,
                    'last_update': datetime.now(),
                    'total_predictions': 0,
                    'correct_predictions': 0
                }
            
            self.logger.log_info("Sentiment analyzer initialized successfully")
            
        except Exception as e:
            self.logger.log_error(f"Failed to initialize sentiment analyzer: {e}")
            raise
    
    async def analyze_sentiment(self, symbol: str = "BTC") -> AggregatedSentiment:
        """Asosiy sentiment tahlili"""
        try:
            # Rate limiting check
            can_proceed, message = self.rate_limiter.check_rate_limit("sentiment_analysis")
            if not can_proceed:
                self.logger.log_warning(f"Rate limit exceeded for sentiment analysis: {message}")
                return self._get_cached_sentiment()
            
            # Parallel sentiment collection
            sentiment_tasks = []
            for source in SentimentSource:
                if self._is_source_active(source):
                    task = asyncio.create_task(self._collect_sentiment_from_source(source, symbol))
                    sentiment_tasks.append(task)
            
            # Gather all sentiment data
            sentiment_results = await asyncio.gather(*sentiment_tasks, return_exceptions=True)
            
            # Filter successful results
            valid_sentiments = []
            for result in sentiment_results:
                if isinstance(result, SentimentData):
                    valid_sentiments.append(result)
                elif isinstance(result, Exception):
                    self.logger.log_error(f"Sentiment collection failed: {result}")
            
            if not valid_sentiments:
                self.logger.log_warning("No valid sentiment data collected")
                return self._get_neutral_sentiment()
            
            # Aggregate sentiments
            aggregated = self._aggregate_sentiments(valid_sentiments)
            
            # Store in history
            self.sentiment_history.append(aggregated)
            
            # Update performance metrics
            self._update_performance_metrics(aggregated)
            
            self.logger.log_info(f"Sentiment analysis completed: {aggregated.overall_score:.3f}")
            return aggregated
            
        except Exception as e:
            self.logger.log_error(f"Sentiment analysis failed: {e}")
            return self._get_neutral_sentiment()
    
    async def _collect_sentiment_from_source(self, source: SentimentSource, symbol: str) -> SentimentData:
        """Har bir manbadan sentiment yig'ish"""
        try:
            if source == SentimentSource.ORDERBOOK:
                return await self._analyze_orderbook_sentiment(symbol)
            elif source == SentimentSource.WHALE_ALERTS:
                return await self._analyze_whale_sentiment(symbol)
            elif source == SentimentSource.TWITTER:
                return await self._analyze_twitter_sentiment(symbol)
            elif source == SentimentSource.NEWS:
                return await self._analyze_news_sentiment(symbol)
            elif source == SentimentSource.REDDIT:
                return await self._analyze_reddit_sentiment(symbol)
            elif source == SentimentSource.SOCIAL_VOLUME:
                return await self._analyze_social_volume_sentiment(symbol)
            else:
                raise ValueError(f"Unknown sentiment source: {source}")
                
        except Exception as e:
            self.logger.log_error(f"Failed to collect sentiment from {source}: {e}")
            raise
    
    async def _analyze_orderbook_sentiment(self, symbol: str) -> SentimentData:
        """Orderbook asosida sentiment tahlili"""
        try:
            # Simulated orderbook analysis
            # Real implementation would connect to exchange API
            
            # Calculate bid/ask ratio, spread, depth
            bid_ask_ratio = np.random.uniform(0.4, 0.6)  # Would be real data
            spread = np.random.uniform(0.001, 0.005)
            depth_imbalance = np.random.uniform(-0.3, 0.3)
            
            # Convert to sentiment score
            sentiment_score = 0.0
            if bid_ask_ratio > 0.55:
                sentiment_score += 0.3
            elif bid_ask_ratio < 0.45:
                sentiment_score -= 0.3
            
            if spread < 0.002:
                sentiment_score += 0.2  # Tight spread = good liquidity
            
            sentiment_score += depth_imbalance
            
            # Normalize to [-1, 1]
            sentiment_score = max(-1, min(1, sentiment_score))
            
            return SentimentData(
                source=SentimentSource.ORDERBOOK,
                score=sentiment_score,
                confidence=0.8,
                timestamp=datetime.now(),
                volume=1000,
                keywords=["orderbook", "depth", "spread"],
                raw_data={
                    "bid_ask_ratio": bid_ask_ratio,
                    "spread": spread,
                    "depth_imbalance": depth_imbalance
                }
            )
            
        except Exception as e:
            self.logger.log_error(f"Orderbook sentiment analysis failed: {e}")
            raise
    
    async def _analyze_whale_sentiment(self, symbol: str) -> SentimentData:
        """Whale alerts asosida sentiment"""
        try:
            # Simulated whale activity analysis
            # Real implementation would connect to whale alert APIs
            
            large_transactions = np.random.randint(0, 5)
            avg_transaction_size = np.random.uniform(100, 1000)  # BTC
            net_flow = np.random.uniform(-500, 500)
            
            # Convert to sentiment
            sentiment_score = 0.0
            if net_flow > 100:
                sentiment_score = 0.6  # Large inflow = bullish
            elif net_flow < -100:
                sentiment_score = -0.6  # Large outflow = bearish
            else:
                sentiment_score = net_flow / 200  # Neutral to slight bias
            
            confidence = min(0.9, large_transactions * 0.2 + 0.1)
            
            return SentimentData(
                source=SentimentSource.WHALE_ALERTS,
                score=sentiment_score,
                confidence=confidence,
                timestamp=datetime.now(),
                volume=large_transactions,
                keywords=["whale", "large_transaction", "flow"],
                raw_data={
                    "large_transactions": large_transactions,
                    "avg_size": avg_transaction_size,
                    "net_flow": net_flow
                }
            )
            
        except Exception as e:
            self.logger.log_error(f"Whale sentiment analysis failed: {e}")
            raise
    
    async def _analyze_twitter_sentiment(self, symbol: str) -> SentimentData:
        """Twitter sentiment tahlili"""
        try:
            # Simulated Twitter sentiment
            # Real implementation would use Twitter API or sentiment APIs
            
            tweet_count = np.random.randint(100, 1000)
            positive_tweets = np.random.randint(0, tweet_count)
            negative_tweets = np.random.randint(0, tweet_count - positive_tweets)
            neutral_tweets = tweet_count - positive_tweets - negative_tweets
            
            # Calculate sentiment score
            if tweet_count > 0:
                sentiment_score = (positive_tweets - negative_tweets) / tweet_count
            else:
                sentiment_score = 0.0
            
            # Adjust based on volume
            volume_boost = min(0.2, tweet_count / 5000)
            sentiment_score = sentiment_score * (1 + volume_boost)
            
            confidence = min(0.7, tweet_count / 1000)
            
            return SentimentData(
                source=SentimentSource.TWITTER,
                score=sentiment_score,
                confidence=confidence,
                timestamp=datetime.now(),
                volume=tweet_count,
                keywords=["twitter", "social", "mentions"],
                raw_data={
                    "positive_tweets": positive_tweets,
                    "negative_tweets": negative_tweets,
                    "neutral_tweets": neutral_tweets,
                    "total_tweets": tweet_count
                }
            )
            
        except Exception as e:
            self.logger.log_error(f"Twitter sentiment analysis failed: {e}")
            raise
    
    async def _analyze_news_sentiment(self, symbol: str) -> SentimentData:
        """News sentiment tahlili"""
        try:
            # Simulated news sentiment
            # Real implementation would use news APIs or NLP
            
            news_count = np.random.randint(5, 50)
            bullish_news = np.random.randint(0, news_count)
            bearish_news = np.random.randint(0, news_count - bullish_news)
            
            if news_count > 0:
                sentiment_score = (bullish_news - bearish_news) / news_count
            else:
                sentiment_score = 0.0
            
            confidence = min(0.8, news_count / 20)
            
            return SentimentData(
                source=SentimentSource.NEWS,
                score=sentiment_score,
                confidence=confidence,
                timestamp=datetime.now(),
                volume=news_count,
                keywords=["news", "media", "headlines"],
                raw_data={
                    "bullish_news": bullish_news,
                    "bearish_news": bearish_news,
                    "total_news": news_count
                }
            )
            
        except Exception as e:
            self.logger.log_error(f"News sentiment analysis failed: {e}")
            raise
    
    async def _analyze_reddit_sentiment(self, symbol: str) -> SentimentData:
        """Reddit sentiment tahlili"""
        try:
            # Simulated Reddit sentiment
            posts_count = np.random.randint(10, 100)
            upvotes = np.random.randint(0, posts_count * 10)
            downvotes = np.random.randint(0, upvotes)
            
            if upvotes + downvotes > 0:
                sentiment_score = (upvotes - downvotes) / (upvotes + downvotes)
            else:
                sentiment_score = 0.0
            
            confidence = min(0.6, posts_count / 50)
            
            return SentimentData(
                source=SentimentSource.REDDIT,
                score=sentiment_score,
                confidence=confidence,
                timestamp=datetime.now(),
                volume=posts_count,
                keywords=["reddit", "community", "discussion"],
                raw_data={
                    "posts_count": posts_count,
                    "upvotes": upvotes,
                    "downvotes": downvotes
                }
            )
            
        except Exception as e:
            self.logger.log_error(f"Reddit sentiment analysis failed: {e}")
            raise
    
    async def _analyze_social_volume_sentiment(self, symbol: str) -> SentimentData:
        """Social volume sentiment tahlili"""
        try:
            # Simulated social volume analysis
            current_volume = np.random.randint(1000, 10000)
            avg_volume = np.random.randint(2000, 5000)
            
            volume_ratio = current_volume / avg_volume
            
            # High volume can indicate both bullish and bearish sentiment
            # We need additional context
            if volume_ratio > 2.0:
                sentiment_score = 0.3  # High interest, slightly bullish
            elif volume_ratio > 1.5:
                sentiment_score = 0.1
            elif volume_ratio < 0.5:
                sentiment_score = -0.2  # Low interest, slightly bearish
            else:
                sentiment_score = 0.0
            
            confidence = min(0.5, abs(volume_ratio - 1.0))
            
            return SentimentData(
                source=SentimentSource.SOCIAL_VOLUME,
                score=sentiment_score,
                confidence=confidence,
                timestamp=datetime.now(),
                volume=current_volume,
                keywords=["volume", "social", "activity"],
                raw_data={
                    "current_volume": current_volume,
                    "avg_volume": avg_volume,
                    "volume_ratio": volume_ratio
                }
            )
            
        except Exception as e:
            self.logger.log_error(f"Social volume sentiment analysis failed: {e}")
            raise
    
    def _aggregate_sentiments(self, sentiments: List[SentimentData]) -> AggregatedSentiment:
        """Sentimentlarni birlashtirish"""
        try:
            if not sentiments:
                return self._get_neutral_sentiment()
            
            # Weighted average calculation
            total_weighted_score = 0.0
            total_weight = 0.0
            total_confidence = 0.0
            breakdown = {}
            
            for sentiment in sentiments:
                # Get source weight
                source_weight = self.source_weights.get(sentiment.source, 0.1)
                
                # Adjust weight based on confidence and performance
                performance = self.source_performance.get(sentiment.source, {})
                performance_multiplier = performance.get('accuracy', 0.5)
                
                adjusted_weight = source_weight * sentiment.confidence * performance_multiplier
                
                # Add to totals
                total_weighted_score += sentiment.score * adjusted_weight
                total_weight += adjusted_weight
                total_confidence += sentiment.confidence
                
                # Store breakdown
                breakdown[sentiment.source.value] = sentiment.score
            
            # Calculate final scores
            if total_weight > 0:
                overall_score = total_weighted_score / total_weight
                avg_confidence = total_confidence / len(sentiments)
            else:
                overall_score = 0.0
                avg_confidence = 0.0
            
            # Calculate reliability based on source diversity and consistency
            reliability = self._calculate_reliability(sentiments)
            
            # Determine trend
            trend = self._determine_trend(overall_score, reliability)
            
            return AggregatedSentiment(
                overall_score=overall_score,
                confidence=avg_confidence,
                reliability=reliability,
                sources_count=len(sentiments),
                timestamp=datetime.now(),
                breakdown=breakdown,
                trend=trend
            )
            
        except Exception as e:
            self.logger.log_error(f"Sentiment aggregation failed: {e}")
            return self._get_neutral_sentiment()
    
    def _calculate_reliability(self, sentiments: List[SentimentData]) -> float:
        """Sentiment ishonchliligini hisoblash"""
        try:
            if len(sentiments) < 2:
                return 0.3  # Low reliability with few sources
            
            # Calculate variance in sentiment scores
            scores = [s.score for s in sentiments]
            variance = np.var(scores)
            
            # Lower variance = higher reliability
            consistency_score = max(0, 1 - variance)
            
            # Source diversity bonus
            diversity_score = min(1.0, len(sentiments) / len(SentimentSource))
            
            # Confidence weighted average
            avg_confidence = np.mean([s.confidence for s in sentiments])
            
            # Combined reliability
            reliability = (consistency_score * 0.4 + diversity_score * 0.3 + avg_confidence * 0.3)
            
            return min(1.0, max(0.0, reliability))
            
        except Exception as e:
            self.logger.log_error(f"Reliability calculation failed: {e}")
            return 0.5
    
    def _determine_trend(self, score: float, reliability: float) -> str:
        """Sentiment trend aniqlash"""
        try:
            # Adjust thresholds based on reliability
            bullish_threshold = 0.2 / max(0.1, reliability)
            bearish_threshold = -0.2 / max(0.1, reliability)
            
            if score > bullish_threshold:
                return "bullish"
            elif score < bearish_threshold:
                return "bearish"
            else:
                return "neutral"
                
        except Exception as e:
            self.logger.log_error(f"Trend determination failed: {e}")
            return "neutral"
    
    def _is_source_active(self, source: SentimentSource) -> bool:
        """Manba faolligini tekshirish"""
        try:
            # Check if source is enabled in config
            source_config = self.config.get('sentiment_sources', {})
            return source_config.get(source.value, True)
            
        except Exception as e:
            self.logger.log_error(f"Source activity check failed: {e}")
            return False
    
    def _get_cached_sentiment(self) -> AggregatedSentiment:
        """Oxirgi sentiment natijasini qaytarish"""
        if self.sentiment_history:
            return self.sentiment_history[-1]
        return self._get_neutral_sentiment()
    
    def _get_neutral_sentiment(self) -> AggregatedSentiment:
        """Neytral sentiment qaytarish"""
        return AggregatedSentiment(
            overall_score=0.0,
            confidence=0.1,
            reliability=0.1,
            sources_count=0,
            timestamp=datetime.now(),
            breakdown={},
            trend="neutral"
        )
    
    def _update_performance_metrics(self, sentiment: AggregatedSentiment):
        """Performance metrikalarini yangilash"""
        try:
            # This would be updated based on actual trading results
            # For now, we simulate performance tracking
            
            for source_name, score in sentiment.breakdown.items():
                try:
                    source = SentimentSource(source_name)
                    if source in self.source_performance:
                        perf = self.source_performance[source]
                        perf['last_update'] = datetime.now()
                        perf['total_predictions'] += 1
                        
                        # Simulate accuracy update (would be based on actual results)
                        perf['accuracy'] = min(0.95, perf['accuracy'] + 0.001)
                        
                except ValueError:
                    continue
                    
        except Exception as e:
            self.logger.log_error(f"Performance metrics update failed: {e}")
    
    def get_sentiment_trend(self, hours: int = 24) -> Dict:
        """Sentiment trend tahlili"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_sentiments = [
                s for s in self.sentiment_history 
                if s.timestamp > cutoff_time
            ]
            
            if not recent_sentiments:
                return {'trend': 'neutral', 'strength': 0.0, 'data_points': 0}
            
            scores = [s.overall_score for s in recent_sentiments]
            
            # Calculate trend
            if len(scores) > 1:
                trend_slope = np.polyfit(range(len(scores)), scores, 1)[0]
                trend_strength = abs(trend_slope)
                
                if trend_slope > 0.01:
                    trend = 'improving'
                elif trend_slope < -0.01:
                    trend = 'deteriorating'
                else:
                    trend = 'stable'
            else:
                trend = 'neutral'
                trend_strength = 0.0
            
            return {
                'trend': trend,
                'strength': trend_strength,
                'data_points': len(recent_sentiments),
                'avg_score': np.mean(scores),
                'latest_score': scores[-1] if scores else 0.0
            }
            
        except Exception as e:
            self.logger.log_error(f"Sentiment trend analysis failed: {e}")
            return {'trend': 'neutral', 'strength': 0.0, 'data_points': 0}
    
    def get_source_performance(self) -> Dict:
        """Manba performance statistikasi"""
        try:
            performance_summary = {}
            
            for source, perf in self.source_performance.items():
                performance_summary[source.value] = {
                    'accuracy': perf['accuracy'],
                    'total_predictions': perf['total_predictions'],
                    'last_update': perf['last_update'].isoformat(),
                    'weight': self.source_weights.get(source, 0.0)
                }
            
            return performance_summary
            
        except Exception as e:
            self.logger.log_error(f"Source performance retrieval failed: {e}")
            return {}
    
    def validate_sentiment_data(self, sentiment: SentimentData) -> bool:
        """Sentiment ma'lumotlarini tekshirish"""
        try:
            # Check score range
            if not -1.0 <= sentiment.score <= 1.0:
                return False
            
            # Check confidence range
            if not 0.0 <= sentiment.confidence <= 1.0:
                return False
            
            # Check timestamp
            if sentiment.timestamp > datetime.now():
                return False
            
            # Check volume
            if sentiment.volume < 0:
                return False
            
            return True
            
        except Exception as e:
            self.logger.log_error(f"Sentiment validation failed: {e}")
            return False
    
    async def health_check(self) -> Dict:
        """Sentiment analyzer health check"""
        try:
            health_status = {
                'status': 'healthy',
                'last_analysis': self.last_analysis_time.isoformat(),
                'sentiment_history_size': len(self.sentiment_history),
                'active_sources': sum(1 for s in SentimentSource if self._is_source_active(s)),
                'issues': []
            }
            
            # Check if analysis is too old
            if datetime.now() - self.last_analysis_time > timedelta(minutes=30):
                health_status['issues'].append("No recent sentiment analysis")
                health_status['status'] = 'warning'
            
            # Check source performance
            poor_sources = [
                s.value for s, perf in self.source_performance.items() 
                if perf['accuracy'] < 0.3
            ]
            
            if poor_sources:
                health_status['issues'].append(f"Poor performing sources: {poor_sources}")
                health_status['status'] = 'warning'
            
            return health_status
            
        except Exception as e:
            self.logger.log_error(f"Health check failed: {e}")
            return {'status': 'error', 'error': str(e)}
