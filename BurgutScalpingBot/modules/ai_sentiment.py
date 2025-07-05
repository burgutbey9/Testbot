import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import aiohttp
import json
import re
from textblob import TextBlob
import numpy as np
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

@dataclass
class NewsItem:
    """Yangilik elementi"""
    title: str
    content: str
    source: str
    timestamp: datetime
    url: str
    sentiment_score: float = 0.0
    relevance_score: float = 0.0
    keywords: List[str] = None

@dataclass
class SentimentMetrics:
    """Sentiment metrikalari"""
    overall_sentiment: float
    news_sentiment: float
    social_sentiment: float
    sentiment_trend: str
    confidence: float
    positive_count: int
    negative_count: int
    neutral_count: int
    key_topics: List[str]
    influential_sources: List[str]

class SentimentAnalyzer:
    """AI Sentiment Analyzer"""
    
    def __init__(self, config):
        self.config = config
        self.session = None
        self.hf_headers = {
            'Authorization': f'Bearer {config.api.hf_api_key}',
            'Content-Type': 'application/json'
        }
        
        # Sentiment model endpoints
        self.sentiment_models = {
            'general': 'https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest',
            'crypto': 'https://api-inference.huggingface.co/models/ElKulako/cryptobert',
            'financial': 'https://api-inference.huggingface.co/models/ProsusAI/finbert'
        }
        
        # Crypto-specific keywords
        self.crypto_keywords = [
            'bitcoin', 'ethereum', 'crypto', 'defi', 'nft', 'blockchain',
            'trading', 'pump', 'dump', 'moon', 'bear', 'bull', 'hodl',
            'scalping', 'arbitrage', 'yield', 'farming', 'staking'
        ]
        
        # Recent data storage
        self.recent_news = deque(maxlen=1000)
        self.sentiment_history = deque(maxlen=100)
        
        # News sources
        self.news_sources = {
            'newsapi': 'https://newsapi.org/v2/everything',
            'reddit': 'https://www.reddit.com/r/CryptoCurrency/hot.json',
            'twitter': None  # Twitter API v2 integration kerak
        }
        
    async def initialize(self):
        """Komponentni boshlash"""
        try:
            logger.info("🔄 AI Sentiment Analyzer boshlanyapti...")
            
            # HTTP session yaratish
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.api.timeout_seconds)
            )
            
            # HuggingFace modellarini test qilish
            await self.test_hf_models()
            
            # News API ni test qilish
            await self.test_news_apis()
            
            logger.info("✅ AI Sentiment Analyzer muvaffaqiyatli boshlandi")
            
        except Exception as e:
            logger.error(f"❌ AI Sentiment Analyzer boshlashda xato: {str(e)}")
            raise
    
    async def test_hf_models(self):
        """HuggingFace modellarini test qilish"""
        test_text = "Bitcoin price is going up today"
        
        for model_name, model_url in self.sentiment_models.items():
            try:
                result = await self.query_hf_model(model_url, test_text)
                if result:
                    logger.info(f"✅ {model_name} modeli ishlaydi")
                else:
                    logger.warning(f"⚠️ {model_name} modeli javob bermadi")
            except Exception as e:
                logger.error(f"❌ {model_name} modeli xatosi: {str(e)}")
    
    async def test_news_apis(self):
        """News API larini test qilish"""
        try:
            # NewsAPI test
            news_data = await self.fetch_news_data()
            if news_data:
                logger.info("✅ NewsAPI ishlaydi")
            else:
                logger.warning("⚠️ NewsAPI javob bermadi")
                
            # Reddit test
            reddit_data = await self.fetch_reddit_data()
            if reddit_data:
                logger.info("✅ Reddit API ishlaydi")
            else:
                logger.warning("⚠️ Reddit API javob bermadi")
                
        except Exception as e:
            logger.error(f"❌ News API test xatosi: {str(e)}")
    
    async def analyze(self) -> Dict[str, Any]:
        """Asosiy sentiment tahlil funksiyasi"""
        try:
            # Yangi news ma'lumotlarini olish
            news_items = await self.fetch_all_news()
            
            # Sentiment tahlili
            analyzed_news = await self.analyze_news_sentiment(news_items)
            
            # Metrikalari hisoblash
            metrics = await self.calculate_sentiment_metrics(analyzed_news)
            
            # Trendni aniqlash
            trend_analysis = await self.analyze_sentiment_trend()
            
            # Signallarni generatsiya qilish
            signals = await self.generate_sentiment_signals(metrics, trend_analysis)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'news_count': len(analyzed_news),
                'metrics': asdict(metrics),
                'trend_analysis': trend_analysis,
                'signals': signals,
                'health': 'OK'
            }
            
        except Exception as e:
            logger.error(f"❌ Sentiment tahlilida xato: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'health': 'ERROR'
            }
    
    async def fetch_all_news(self) -> List[NewsItem]:
        """Barcha manbalardan yangilik olish"""
        all_news = []
        
        # NewsAPI dan olish
        try:
            news_data = await self.fetch_news_data()
            if news_data:
                all_news.extend(news_data)
        except Exception as e:
            logger.error(f"❌ NewsAPI xatosi: {str(e)}")
        
        # Reddit dan olish
        try:
            reddit_data = await self.fetch_reddit_data()
            if reddit_data:
                all_news.extend(reddit_data)
        except Exception as e:
            logger.error(f"❌ Reddit API xatosi: {str(e)}")
        
        # Takroriy yangiliklarni filtrlash
        unique_news = self.filter_duplicate_news(all_news)
        
        # Recent news ni yangilash
        self.recent_news.extend(unique_news)
        
        return unique_news
    
    async def fetch_news_data(self) -> List[NewsItem]:
        """NewsAPI dan ma'lumot olish"""
        try:
            url = self.news_sources['newsapi']
            params = {
                'q': 'cryptocurrency OR bitcoin OR ethereum OR DeFi OR blockchain',
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 50,
                'apiKey': self.config.api.newsapi_key
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = data.get('articles', [])
                    
                    news_items = []
                    for article in articles:
                        if self.is_crypto_relevant(article.get('title', '') + ' ' + article.get('description', '')):
                            news_item = NewsItem(
                                title=article.get('title', ''),
                                content=article.get('description', ''),
                                source=article.get('source', {}).get('name', 'Unknown'),
                                timestamp=datetime.fromisoformat(article.get('publishedAt', '').replace('Z', '+00:00')),
                                url=article.get('url', ''),
                                keywords=self.extract_keywords(article.get('title', '') + ' ' + article.get('description', ''))
                            )
                            news_items.append(news_item)
                    
                    return news_items
                else:
                    logger.error(f"❌ NewsAPI HTTP xatosi: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"❌ NewsAPI ma'lumot olishda xato: {str(e)}")
            return []
    
    async def fetch_reddit_data(self) -> List[NewsItem]:
        """Reddit dan ma'lumot olish"""
        try:
            url = self.news_sources['reddit']
            headers = {'User-Agent': 'BurgutScalpingBot/1.0'}
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    posts = data.get('data', {}).get('children', [])
                    
                    news_items = []
                    for post_data in posts:
                        post = post_data.get('data', {})
                        
                        if self.is_crypto_relevant(post.get('title', '') + ' ' + post.get('selftext', '')):
                            news_item = NewsItem(
                                title=post.get('title', ''),
                                content=post.get('selftext', ''),
                                source='Reddit',
                                timestamp=datetime.fromtimestamp(post.get('created_utc', 0)),
                                url=f"https://reddit.com{post.get('permalink', '')}",
                                keywords=self.extract_keywords(post.get('title', '') + ' ' + post.get('selftext', ''))
                            )
                            news_items.append(news_item)
                    
                    return news_items
                else:
                    logger.error(f"❌ Reddit API HTTP xatosi: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"❌ Reddit ma'lumot olishda xato: {str(e)}")
            return []
    
    def is_crypto_relevant(self, text: str) -> bool:
        """Matn crypto bilan bog'liq ekanligini tekshirish"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.crypto_keywords)
    
    def extract_keywords(self, text: str) -> List[str]:
        """Matndan kalit so'zlarni ajratish"""
        found_keywords = []
        text_lower = text.lower()
        
        for keyword in self.crypto_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def filter_duplicate_news(self, news_items: List[NewsItem]) -> List[NewsItem]:
        """Takroriy yangiliklarni filtrlash"""
        seen_titles = set()
        unique_news = []
        
        for item in news_items:
            title_key = item.title.lower().strip()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_news.append(item)
        
        return unique_news
    
    async def analyze_news_sentiment(self, news_items: List[NewsItem]) -> List[NewsItem]:
        """Yangiliklar sentiment tahlili"""
        analyzed_items = []
        
        for item in news_items:
            try:
                # Matn tayyorlash
                text = f"{item.title} {item.content}"
                
                # HuggingFace modeli bilan tahlil
                hf_sentiment = await self.analyze_with_hf(text)
                
                # TextBlob bilan tahlil (fallback)
                tb_sentiment = self.analyze_with_textblob(text)
                
                # Sentiment ball hisoblash
                if hf_sentiment:
                    item.sentiment_score = hf_sentiment['score']
                else:
                    item.sentiment_score = tb_sentiment
                
                # Relevance score hisoblash
                item.relevance_score = self.calculate_relevance_score(text)
                
                analyzed_items.append(item)
                
            except Exception as e:
                logger.error(f"❌ Sentiment tahlilida xato: {str(e)}")
                analyzed_items.append(item)  # Xato bo'lsa ham qo'shamiz
        
        return analyzed_items
    
    async def analyze_with_hf(self, text: str) -> Optional[Dict]:
        """HuggingFace modeli bilan tahlil"""
        try:
            # Eng yaxshi modelni tanlash
            model_url = self.sentiment_models['crypto']
            
            result = await self.query_hf_model(model_url, text)
            
            if result and len(result) > 0:
                # Eng yuqori skorli natijani olish
                best_result = max(result, key=lambda x: x.get('score', 0))
                
                # Sentiment scoreini normalize qilish (-1 dan 1 gacha)
                label = best_result.get('label', '').upper()
                score = best_result.get('score', 0)
                
                if 'POSITIVE' in label or 'BULLISH' in label:
                    return {'score': score, 'label': 'POSITIVE'}
                elif 'NEGATIVE' in label or 'BEARISH' in label:
                    return {'score': -score, 'label': 'NEGATIVE'}
                else:
                    return {'score': 0, 'label': 'NEUTRAL'}
            
            return None
            
        except Exception as e:
            logger.error(f"❌ HuggingFace tahlilida xato: {str(e)}")
            return None
    
    async def query_hf_model(self, model_url: str, text: str) -> Optional[List[Dict]]:
        """HuggingFace modelidan so'rov"""
        try:
            payload = {'inputs': text}
            
            async with self.session.post(
                model_url, 
                headers=self.hf_headers, 
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result
                else:
                    logger.error(f"❌ HuggingFace HTTP xatosi: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ HuggingFace so'rov xatosi: {str(e)}")
            return None
    
    def analyze_with_textblob(self, text: str) -> float:
        """TextBlob bilan sentiment tahlili"""
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity  # -1 dan 1 gacha
        except Exception as e:
            logger.error(f"❌ TextBlob tahlilida xato: {str(e)}")
            return 0.0
    
    def calculate_relevance_score(self, text: str) -> float:
        """Matnning crypto bilan bog'liqlik scorini hisoblash"""
        try:
            text_lower = text.lower()
            keyword_count = sum(1 for keyword in self.crypto_keywords if keyword in text_lower)
            
            # Normalize qilish
            max_possible_score = min(len(self.crypto_keywords), 10)
            relevance_score = min(1.0, keyword_count / max_possible_score)
            
            return relevance_score
            
        except Exception as e:
            logger.error(f"❌ Relevance score hisoblashda xato: {str(e)}")
            return 0.0
    
    async def calculate_sentiment_metrics(self, news_items: List[NewsItem]) -> SentimentMetrics:
        """Sentiment metrikalari hisoblash"""
        if not news_items:
            return SentimentMetrics(
                overall_sentiment=0.0,
                news_sentiment=0.0,
                social_sentiment=0.0,
                sentiment_trend='NEUTRAL',
                confidence=0.0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                key_topics=[],
                influential_sources=[]
            )
        
        # Sentiment scores
        sentiment_scores = [item.sentiment_score for item in news_items]
        overall_sentiment = np.mean(sentiment_scores)
        
        # Source bo'yicha ajratish
        news_sources = ['Reuters', 'CoinDesk', 'CoinTelegraph', 'CryptoNews']
        social_sources = ['Reddit', 'Twitter']
        
        news_sentiments = [
            item.sentiment_score for item in news_items 
            if item.source in news_sources
        ]
        social_sentiments = [
            item.sentiment_score for item in news_items 
            if item.source in social_sources
        ]
        
        news_sentiment = np.mean(news_sentiments) if news_sentiments else 0.0
        social_sentiment = np.mean(social_sentiments) if social_sentiments else 0.0
        
        # Sentiment counts
        positive_count = len([s for s in sentiment_scores if s > 0.1])
        negative_count = len([s for s in sentiment_scores if s < -0.1])
        neutral_count = len(sentiment_scores) - positive_count - negative_count
        
        # Sentiment trend
        if overall_sentiment > 0.2:
            sentiment_trend = 'BULLISH'
        elif overall_sentiment < -0.2:
            sentiment_trend = 'BEARISH'
        else:
            sentiment_trend = 'NEUTRAL'
        
        # Confidence (relevance va sentiment strength asosida)
        relevance_scores = [item.relevance_score for item in news_items]
        confidence = np.mean(relevance_scores) * min(1.0, abs(overall_sentiment) * 2)
        
        # Key topics
        all_keywords = []
        for item in news_items:
            all_keywords.extend(item.keywords or [])
        
        keyword_counts = defaultdict(int)
        for keyword in all_keywords:
            keyword_counts[keyword] += 1
        
        key_topics = [
            keyword for keyword, count in 
            sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        
        # Influential sources
        source_counts = defaultdict(int)
        for item in news_items:
            source_counts[item.source] += 1
        
        influential_sources = [
            source for source, count in 
            sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
        
        return SentimentMetrics(
            overall_sentiment=overall_sentiment,
            news_sentiment=news_sentiment,
            social_sentiment=social_sentiment,
            sentiment_trend=sentiment_trend,
            confidence=confidence,
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            key_topics=key_topics,
            influential_sources=influential_sources
        )
    
    async def analyze_sentiment_trend(self) -> Dict[str, Any]:
        """Sentiment trend tahlili"""
        try:
            # Oxirgi 10 ta sentiment o'rtachasini olish
            if len(self.sentiment_history) < 3:
                return {
                    'direction': 'UNKNOWN',
                    'strength': 0.0,
                    'confidence': 0.0
                }
            
            recent_sentiments = list(self.sentiment_history)[-10:]
            
            # Trend yo'nalishini aniqlash
            if len(recent_sentiments) >= 3:
                first_half = recent_sentiments[:len(recent_sentiments)//2]
                second_half = recent_sentiments[len(recent_sentiments)//2:]
                
                first_avg = np.mean(first_half)
                second_avg = np.mean(second_half)
                
                trend_change = second_avg - first_avg
                
                if trend_change > 0.1:
                    direction = 'IMPROVING'
                elif trend_change < -0.1:
                    direction = 'DETERIORATING'
                else:
                    direction = 'STABLE'
                
                return {
                    'direction': direction,
                    'strength': abs(trend_change),
                    'confidence': min(1.0, len(recent_sentiments) / 10)
                }
            
            return {
                'direction': 'STABLE',
                'strength': 0.0,
                'confidence': 0.5
            }
            
        except Exception as e:
            logger.error(f"❌ Trend tahlilida xato: {str(e)}")
            return {
                'direction': 'UNKNOWN',
                'strength': 0.0,
                'confidence': 0.0
            }
    
    async def generate_sentiment_signals(self, metrics: SentimentMetrics, trend: Dict) -> Dict[str, Any]:
        """Sentiment signallarini generatsiya qilish"""
        signals = {
            'action': 'HOLD',
            'strength': 0.5,
            'confidence': metrics.confidence,
            'reasons': []
        }
        
        try:
            # Asosiy sentiment tahlili
            if metrics.overall_sentiment > 0.3 and metrics.confidence > 0.6:
                signals['action'] = 'BUY'
                signals['strength'] = min(1.0, metrics.overall_sentiment * 2)
                signals['reasons'].append(f'Ijobiy sentiment: {metrics.overall_sentiment:.2f}')
            
            elif metrics.overall_sentiment < -0.3 and metrics.confidence > 0.6:
                signals['action'] = 'SELL'
                signals['strength'] = min(1.0, abs(metrics.overall_sentiment) * 2)
                signals['reasons'].append(f'Salbiy sentiment: {metrics.overall_sentiment:.2f}')
            
            # Trend tahlili
            if trend['direction'] == 'IMPROVING' and trend['strength'] > 0.2:
                if signals['action'] == 'HOLD':
                    signals['action'] = 'BUY'
                signals['strength'] = min(1.0, signals['strength'] + trend['strength'])
                signals['reasons'].append(f'Yaxshilanayotgan trend: {trend["strength"]:.2f}')
            
            elif trend['direction'] == 'DETERIORATING' and trend['strength'] > 0.2:
                if signals['action'] == 'HOLD':
                    signals['action'] = 'SELL'
                signals['strength'] = min(1.0, signals['strength'] + trend['strength'])
                signals['reasons'].append(f'Yomonlashayotgan trend: {trend["strength"]:.2f}')
            
            # Social vs News sentiment farqi
            sentiment_diff = abs(metrics.social_sentiment - metrics.news_sentiment)
            if sentiment_diff > 0.4:
                signals['strength'] = max(0.3, signals['strength'] - 0.2)
                signals['reasons'].append(f'Social va news sentiment farqi: {sentiment_diff:.2f}')
            
            # Sentiment history yangilash
            self.sentiment_history.append(metrics.overall_sentiment)
            
        except Exception as e:
            logger.error(f"❌ Sentiment signal generatsiyasida xato: {str(e)}")
        
        return signals
    
    async def health_check(self):
        """Sog'lik tekshiruvi"""
        try:
            # Session tekshirish
            if not self.session or self.session.closed:
                raise Exception("HTTP session yopilgan")
            
            # HuggingFace API tekshirish
            test_result = await self.query_hf_model(
                self.sentiment_models['general'], 
                "test message"
            )
            
            if not test_result:
                raise Exception("HuggingFace API javob bermayapti")
            
            logger.info("✅ AI Sentiment Analyzer sog'lom")
            
        except Exception as e:
            logger.error(f"❌ AI Sentiment Analyzer sog'lik tekshiruvida xato: {str(e)}")
            raise
    
    async def shutdown(self):
        """Komponentni to'xtatish"""
        logger.info("🛑 AI Sentiment Analyzer to'xtatilmoqda...")
        
        if self.session and not self.session.closed:
            await self.session.close()
        
        logger.info("✅ AI Sentiment Analyzer to'xtatildi")
