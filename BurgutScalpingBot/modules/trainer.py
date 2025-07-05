import asyncio
import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
import warnings
warnings.filterwarnings('ignore')

from .utils import BurgutLogger, DataManager, PerformanceMonitor
from .backtest import BacktestEngine, BacktestResult
from config import *

@dataclass
class TrainingResult:
    """Training natijasi"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    cross_val_score: float
    training_time: float
    feature_importance: Dict[str, float]
    model_path: str
    created_at: datetime

class FeatureEngineer:
    """Feature engineering sinfi"""
    
    def __init__(self):
        self.logger = BurgutLogger().logger
        self.scaler = StandardScaler()
        
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Texnik ko'rsatkichlarni yaratish"""
        try:
            # Moving averages
            df['sma_5'] = df['close'].rolling(window=5).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Stochastic
            low_14 = df['low'].rolling(window=14).min()
            high_14 = df['high'].rolling(window=14).max()
            df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
            
            # Volume features
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['price_volume'] = df['close'] * df['volume']
            
            # Volatility
            df['volatility'] = df['close'].rolling(window=20).std()
            df['atr'] = self.calculate_atr(df)
            
            # Price features
            df['price_change'] = df['close'].pct_change()
            df['price_change_5'] = df['close'].pct_change(periods=5)
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']
            
            # Trend features
            df['trend_5'] = np.where(df['close'] > df['sma_5'], 1, -1)
            df['trend_20'] = np.where(df['close'] > df['sma_20'], 1, -1)
            df['trend_strength'] = abs(df['close'] - df['sma_20']) / df['sma_20']
            
            self.logger.info("Texnik ko'rsatkichlar yaratildi")
            return df
            
        except Exception as e:
            self.logger.error(f"Feature engineering da xatolik: {e}")
            raise
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range hisoblash"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        return true_range.rolling(window=period).mean()
    
    def create_sentiment_features(self, df: pd.DataFrame, sentiment_data: Dict[str, Any]) -> pd.DataFrame:
        """Sentiment featurelari yaratish"""
        try:
            # Sentiment score
            df['sentiment_score'] = sentiment_data.get('sentiment_score', 0.0)
            df['sentiment_strength'] = sentiment_data.get('sentiment_strength', 0.0)
            
            # News volume
            df['news_volume'] = sentiment_data.get('news_volume', 0)
            df['social_volume'] = sentiment_data.get('social_volume', 0)
            
            # Fear & Greed index
            df['fear_greed'] = sentiment_data.get('fear_greed', 50)
            
            # Reddit/Twitter mentions
            df['reddit_mentions'] = sentiment_data.get('reddit_mentions', 0)
            df['twitter_mentions'] = sentiment_data.get('twitter_mentions', 0)
            
            self.logger.info("Sentiment features yaratildi")
            return df
            
        except Exception as e:
            self.logger.error(f"Sentiment features yaratishda xatolik: {e}")
            return df
    
    def create_orderflow_features(self, df: pd.DataFrame, orderflow_data: Dict[str, Any]) -> pd.DataFrame:
        """Order flow featurelari yaratish"""
        try:
            # Buy/Sell pressure
            df['buy_pressure'] = orderflow_data.get('buy_pressure', 0.5)
            df['sell_pressure'] = orderflow_data.get('sell_pressure', 0.5)
            df['pressure_ratio'] = df['buy_pressure'] / (df['sell_pressure'] + 1e-6)
            
            # Large transactions
            df['large_buys'] = orderflow_data.get('large_buys', 0)
            df['large_sells'] = orderflow_data.get('large_sells', 0)
            df['whale_activity'] = df['large_buys'] - df['large_sells']
            
            # Liquidity metrics
            df['bid_ask_spread'] = orderflow_data.get('bid_ask_spread', 0.0)
            df['order_book_depth'] = orderflow_data.get('order_book_depth', 0.0)
            df['liquidity_ratio'] = orderflow_data.get('liquidity_ratio', 1.0)
            
            # MEV metrics
            df['mev_activity'] = orderflow_data.get('mev_activity', 0.0)
            df['sandwich_attacks'] = orderflow_data.get('sandwich_attacks', 0)
            
            self.logger.info("Order flow features yaratildi")
            return df
            
        except Exception as e:
            self.logger.error(f"Order flow features yaratishda xatolik: {e}")
            return df
    
    def create_target_variable(self, df: pd.DataFrame, prediction_horizon: int = 5) -> pd.Series:
        """Target variable yaratish"""
        try:
            # Kelgusidagi narx o'zgarishi
            future_price = df['close'].shift(-prediction_horizon)
            current_price = df['close']
            
            # Profit threshold (1% dan ko'p o'zgarish)
            profit_threshold = 0.01
            
            # Target: 1 (BUY), 0 (HOLD), -1 (SELL)
            price_change = (future_price - current_price) / current_price
            
            target = np.where(price_change > profit_threshold, 1,
                             np.where(price_change < -profit_threshold, -1, 0))
            
            return pd.Series(target, index=df.index)
            
        except Exception as e:
            self.logger.error(f"Target variable yaratishda xatolik: {e}")
            return pd.Series(np.zeros(len(df)), index=df.index)
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Barcha featurelarni tayyorlash"""
        try:
            # Feature columns
            feature_columns = [
                'sma_5', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
                'rsi', 'macd', 'macd_signal', 'macd_histogram',
                'bb_width', 'bb_position', 'stoch_k', 'stoch_d',
                'volume_ratio', 'volatility', 'atr',
                'price_change', 'price_change_5', 'high_low_ratio',
                'close_open_ratio', 'trend_5', 'trend_20', 'trend_strength',
                'sentiment_score', 'sentiment_strength', 'news_volume',
                'fear_greed', 'buy_pressure', 'sell_pressure',
                'whale_activity', 'liquidity_ratio'
            ]
            
            # Mavjud columnlarni olish
            available_columns = [col for col in feature_columns if col in df.columns]
            
            if not available_columns:
                raise ValueError("Hech qanday feature mavjud emas")
            
            # Features va target
            X = df[available_columns].copy()
            y = self.create_target_variable(df)
            
            # NaN qiymatlarni to'ldirish
            X = X.fillna(method='ffill').fillna(0)
            
            # Sonsiz qiymatlarni olib tashlash
            X = X.replace([np.inf, -np.inf], 0)
            
            # Scaling
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            
            self.logger.info(f"Features tayyorlandi: {X_scaled.shape}")
            return X_scaled, y
            
        except Exception as e:
            self.logger.error(f"Features tayyorlashda xatolik: {e}")
            raise

class ModelTrainer:
    """AI Model trainer sinfi"""
    
    def __init__(self):
        self.logger = BurgutLogger().logger
        self.data_manager = DataManager()
        self.performance_monitor = PerformanceMonitor()
        self.feature_engineer = FeatureEngineer()
        
        # Model saqlash yo'li
        self.models_dir = Path("data/ai_models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Modellar
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        }
        
        self.trained_models = {}
        self.training_results = {}
        
    def load_training_data(self, data_path: str) -> pd.DataFrame:
        """Training ma'lumotlarni yuklash"""
        try:
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.json'):
                data = self.data_manager.load_json(data_path)
                df = pd.DataFrame(data)
            else:
                raise ValueError(f"Noto'g'ri fayl formati: {data_path}")
            
            # Timestamp columnini datetime ga aylantirish
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            self.logger.info(f"Training data yuklandi: {len(df)} qator")
            return df
            
        except Exception as e:
            self.logger.error(f"Training data yuklashda xatolik: {e}")
            raise
    
    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> TrainingResult:
        """Model training"""
        try:
            start_time = datetime.now()
            
            model = self.models[model_name]
            
            # Model training
            model.fit(X_train, y_train)
            
            # Cross validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # Test predictions
            X_train_pred = model.predict(X_train)
            
            # Metrics
            accuracy = accuracy_score(y_train, X_train_pred)
            report = classification_report(y_train, X_train_pred, output_dict=True)
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(X_train.columns, model.feature_importances_))
                # Top 10 muhim featurelar
                feature_importance = dict(sorted(feature_importance.items(), 
                                               key=lambda x: x[1], reverse=True)[:10])
            else:
                feature_importance = {}
            
            # Model saqlash
            model_path = self.models_dir / f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            joblib.dump(model, model_path)
            
            # Scaler ham saqlash
            scaler_path = self.models_dir / f"scaler_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            joblib.dump(self.feature_engineer.scaler, scaler_path)
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Training result
            result = TrainingResult(
                model_name=model_name,
                accuracy=accuracy,
                precision=report['weighted avg']['precision'],
                recall=report['weighted avg']['recall'],
                f1_score=report['weighted avg']['f1-score'],
                cross_val_score=cv_scores.mean(),
                training_time=training_time,
                feature_importance=feature_importance,
                model_path=str(model_path),
                created_at=datetime.now()
            )
            
            # Modelni saqlash
            self.trained_models[model_name] = model
            self.training_results[model_name] = result
            
            self.logger.info(f"Model training yakunlandi: {model_name}, accuracy: {accuracy:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Model training da xatolik: {e}")
            raise
    
    def evaluate_model(self, model_name: str, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Model evaluation"""
        try:
            model = self.trained_models[model_name]
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            evaluation_result = {
                'accuracy': accuracy,
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score'],
                'support': report['weighted avg']['support']
            }
            
            self.logger.info(f"Model evaluation: {model_name}, test accuracy: {accuracy:.4f}")
            return evaluation_result
            
        except Exception as e:
            self.logger.error(f"Model evaluation da xatolik: {e}")
            return {}
    
    def select_best_model(self) -> str:
        """Eng yaxshi modelni tanlash"""
        try:
            if not self.training_results:
                raise ValueError("Training results yo'q")
            
            # F1-score bo'yicha eng yaxshi model
            best_model = max(self.training_results.items(), 
                           key=lambda x: x[1].f1_score)
            
            best_model_name = best_model[0]
            best_result = best_model[1]
            
            self.logger.info(f"Eng yaxshi model: {best_model_name}, F1-score: {best_result.f1_score:.4f}")
            return best_model_name
            
        except Exception as e:
            self.logger.error(f"Best model tanlashda xatolik: {e}")
            return list(self.training_results.keys())[0] if self.training_results else 'random_forest'
    
    def save_training_results(self, filename: str = None):
        """Training natijalarini saqlash"""
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"training_results_{timestamp}.json"
            
            # Results lug'atga aylantirish
            results_dict = {}
            for model_name, result in self.training_results.items():
                results_dict[model_name] = {
                    'model_name': result.model_name,
                    'accuracy': result.accuracy,
                    'precision': result.precision,
                    'recall': result.recall,
                    'f1_score': result.f1_score,
                    'cross_val_score': result.cross_val_score,
                    'training_time': result.training_time,
                    'feature_importance': result.feature_importance,
                    'model_path': result.model_path,
                    'created_at': result.created_at.isoformat()
                }
            
            # Saqlash
            results_path = self.models_dir / filename
            with open(results_path, 'w') as f:
                json.dump(results_dict, f, indent=2)
            
            self.logger.info(f"Training results saqlandi: {results_path}")
            
        except Exception as e:
            self.logger.error(f"Training results saqlashda xatolik: {e}")
    
    def load_model(self, model_path: str):
        """Saqlangan modelni yuklash"""
        try:
            model = joblib.load(model_path)
            self.logger.info(f"Model yuklandi: {model_path}")
            return model
            
        except Exception as e:
            self.logger.error(f"Model yuklashda xatolik: {e}")
            raise
    
    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """Bashorat qilish"""
        try:
            model = self.trained_models[model_name]
            predictions = model.predict(X)
            return predictions
            
        except Exception as e:
            self.logger.error(f"Prediction da xatolik: {e}")
            return np.zeros(len(X))
    
    def get_feature_importance(self, model_name: str) -> Dict[str, float]:
        """Feature importance olish"""
        try:
            if model_name in self.training_results:
                return self.training_results[model_name].feature_importance
            return {}
            
        except Exception as e:
            self.logger.error(f"Feature importance olishda xatolik: {e}")
            return {}

class StrategyOptimizer:
    """Strategiya optimizatsiya sinfi"""
    
    def __init__(self):
        self.logger = BurgutLogger().logger
        self.backtest_engine = None
        
    def optimize_parameters(self, strategy_class, data_source: str, 
                          param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Strategiya parametrlarini optimizatsiya qilish"""
        try:
            best_params = None
            best_score = float('-inf')
            
            # Grid search
            from itertools import product
            
            # Parametr kombinatsiyalarini yaratish
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            
            for param_combination in product(*param_values):
                params = dict(zip(param_names, param_combination))
                
                # Strategiya yaratish
                strategy = strategy_class(**params)
                
                # Backtest
                self.backtest_engine = BacktestEngine(strategy)
                
                start_date = datetime.now() - timedelta(days=30)
                end_date = datetime.now()
                
                result = self.backtest_engine.run_backtest(data_source, start_date, end_date)
                
                # Sharpe ratio bo'yicha baholash
                score = result.sharpe_ratio
                
                if score > best_score:
                    best_score = score
                    best_params = params
                
                self.logger.info(f"Parametrlar: {params}, Score: {score:.4f}")
            
            self.logger.info(f"Eng yaxshi parametrlar: {best_params}, Score: {best_score:.4f}")
            return best_params
            
        except Exception as e:
            self.logger.error(f"Parametr optimizatsiyasida xatolik: {e}")
            return {}

async def train_model():
    """Asosiy training funksiyasi"""
    try:
        logger = BurgutLogger().logger
        trainer = ModelTrainer()
        
        # Training data yuklash
        data_path = "data/historical_data.csv"
        
        # Agar fayl mavjud bo'lmasa, demo data yaratish
        if not Path(data_path).exists():
            logger.warning("Historical data topilmadi, demo data yaratilmoqda...")
            demo_data = create_demo_data()
            demo_data.to_csv(data_path)
        
        df = trainer.load_training_data(data_path)
        
        # Feature engineering
        df = trainer.feature_engineer.create_technical_features(df)
        
        # Demo sentiment va orderflow data
        sentiment_data = {
            'sentiment_score': 0.6,
            'sentiment_strength': 0.8,
            'news_volume': 100,
            'fear_greed': 60
        }
        
        orderflow_data = {
            'buy_pressure': 0.55,
            'sell_pressure': 0.45,
            'large_buys': 10,
            'large_sells': 5,
            'liquidity_ratio': 1.2
        }
        
        df = trainer.feature_engineer.create_sentiment_features(df, sentiment_data)
        df = trainer.feature_engineer.create_orderflow_features(df, orderflow_data)
        
        # Features va target tayyorlash
        X, y = trainer.feature_engineer.prepare_features(df)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Modellarni training qilish
        training_results = {}
        
        for model_name in trainer.models.keys():
            logger.info(f"Training boshlandi: {model_name}")
            result = trainer.train_model(model_name, X_train, y_train)
            training_results[model_name] = result
            
            # Evaluation
            eval_result = trainer.evaluate_model(model_name, X_test, y_test)
            logger.info(f"Evaluation: {model_name}, {eval_result}")
        
        # Eng yaxshi modelni tanlash
        best_model_name = trainer.select_best_model()
        
        # Natijalarni saqlash
        trainer.save_training_results()
        
        logger.info(f"Training yakunlandi. Eng yaxshi model: {best_model_name}")
        
        # Har 24 soatda bir marta training
        await asyncio.sleep(24 * 60 * 60)  # 24 soat
        
    except Exception as e:
        logger.error(f"Training da xatolik: {e}")
        # 1 soat kutib qayta urinish
        await asyncio.sleep(60 * 60)

def create_demo_data() -> pd.DataFrame:
    """Demo data yaratish"""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='1H')
    
    # Tasodifiy narx ma'lumotlari
    np.random.seed(42)
    price = 100.0
    data = []
    
    for date in dates:
        # Random walk
        change = np.random.normal(0, 0.02)
        price = price * (1 + change)
        
        # OHLCV
        open_price = price
        high_price = price * (1 + abs(np.random.normal(0, 0.01)))
        low_price = price * (1 - abs(np.random.normal(0, 0.01)))
        close_price = price * (1 + np.random.normal(0, 0.005))
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
        
        price = close_price
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    asyncio.run(train_model())
