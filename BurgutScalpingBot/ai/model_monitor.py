"""
AI Model Performance Monitoring va Drift Detection
Bu modul AI modellarning performance'ini kuzatib boradi va model drift'ni aniqlaydi
"""
import time
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import deque
import logging

@dataclass
class ModelPrediction:
    """Model prediction ma'lumotlari"""
    timestamp: datetime
    prediction: float
    confidence: float
    actual_result: Optional[float] = None
    market_condition: str = "normal"

@dataclass
class ModelMetrics:
    """Model metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_confidence: float

class ModelMonitor:
    """AI Model Performance Monitor"""
    
    def __init__(self, window_size: int = 100, drift_threshold: float = 0.15):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.predictions = deque(maxlen=window_size)
        self.baseline_metrics = None
        self.current_metrics = None
        self.drift_detected = False
        self.last_evaluation = datetime.now()
        self.performance_history = []
        self.logger = logging.getLogger(__name__)
        
    def add_prediction(self, prediction: ModelPrediction):
        """Yangi prediction qo'shish"""
        self.predictions.append(prediction)
        
        # Har 10 prediction'da metrics yangilanadi
        if len(self.predictions) % 10 == 0:
            self._update_metrics()
            
    def _update_metrics(self):
        """Metrics'ni yangilash"""
        if len(self.predictions) < 20:  # Minimum ma'lumot kerak
            return
            
        # Faqat actual_result mavjud bo'lgan prediction'lar
        completed_predictions = [p for p in self.predictions if p.actual_result is not None]
        
        if len(completed_predictions) < 10:
            return
            
        try:
            self.current_metrics = self._calculate_metrics(completed_predictions)
            
            # Baseline o'rnatish (birinchi marta)
            if self.baseline_metrics is None:
                self.baseline_metrics = self.current_metrics
                self.logger.info("Baseline metrics established")
                return
                
            # Drift detection
            self._check_drift()
            
            # Performance history saqlash
            self.performance_history.append({
                'timestamp': datetime.now(),
                'metrics': self.current_metrics,
                'drift_detected': self.drift_detected
            })
            
            # Eski history'ni tozalash (oxirgi 1000 ta)
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
                
        except Exception as e:
            self.logger.error(f"Metrics calculation error: {e}")
            
    def _calculate_metrics(self, predictions: List[ModelPrediction]) -> ModelMetrics:
        """Metrics hisoblash"""
        # Prediction va actual results
        y_pred = [p.prediction for p in predictions]
        y_actual = [p.actual_result for p in predictions]
        confidences = [p.confidence for p in predictions]
        
        # Binary classification uchun threshold
        threshold = 0.5
        y_pred_binary = [1 if p > threshold else 0 for p in y_pred]
        y_actual_binary = [1 if a > 0 else 0 for a in y_actual]  # Profit/Loss
        
        # Accuracy
        correct = sum(1 for p, a in zip(y_pred_binary, y_actual_binary) if p == a)
        accuracy = correct / len(y_pred_binary)
        
        # Precision, Recall, F1
        tp = sum(1 for p, a in zip(y_pred_binary, y_actual_binary) if p == 1 and a == 1)
        fp = sum(1 for p, a in zip(y_pred_binary, y_actual_binary) if p == 1 and a == 0)
        fn = sum(1 for p, a in zip(y_pred_binary, y_actual_binary) if p == 0 and a == 1)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Financial metrics
        returns = np.array(y_actual)
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        # Max drawdown
        cumulative_returns = np.cumsum(returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Win rate
        win_rate = sum(1 for r in returns if r > 0) / len(returns)
        
        # Average confidence
        avg_confidence = np.mean(confidences)
        
        return ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=abs(max_drawdown),
            win_rate=win_rate,
            avg_confidence=avg_confidence
        )
        
    def _check_drift(self):
        """Model drift tekshirish"""
        if self.baseline_metrics is None or self.current_metrics is None:
            return
            
        # Accuracy degradation
        accuracy_drop = self.baseline_metrics.accuracy - self.current_metrics.accuracy
        
        # Sharpe ratio degradation
        sharpe_drop = self.baseline_metrics.sharpe_ratio - self.current_metrics.sharpe_ratio
        
        # Confidence degradation
        confidence_drop = self.baseline_metrics.avg_confidence - self.current_metrics.avg_confidence
        
        # Drift criteria
        drift_signals = [
            accuracy_drop > self.drift_threshold,
            sharpe_drop > 0.5,  # Sharpe ratio 0.5 dan ko'p pasaysa
            confidence_drop > 0.2,  # Confidence 20% dan ko'p pasaysa
            self.current_metrics.max_drawdown > 0.1  # 10% dan ko'p drawdown
        ]
        
        if any(drift_signals):
            if not self.drift_detected:
                self.drift_detected = True
                self.logger.warning(f"Model drift detected! Accuracy drop: {accuracy_drop:.3f}, "
                                  f"Sharpe drop: {sharpe_drop:.3f}, Confidence drop: {confidence_drop:.3f}")
        else:
            self.drift_detected = False
            
    def get_model_health(self) -> Dict:
        """Model health status"""
        if self.current_metrics is None:
            return {
                'status': 'insufficient_data',
                'message': 'Yetarli ma\'lumot yo\'q',
                'recommendations': ['Ko\'proq ma\'lumot to\'plash kerak']
            }
            
        health_score = self._calculate_health_score()
        
        status = 'healthy'
        if health_score < 0.3:
            status = 'critical'
        elif health_score < 0.6:
            status = 'warning'
            
        recommendations = self._generate_recommendations()
        
        return {
            'status': status,
            'health_score': health_score,
            'drift_detected': self.drift_detected,
            'metrics': self.current_metrics,
            'recommendations': recommendations,
            'last_update': self.last_evaluation
        }
        
    def _calculate_health_score(self) -> float:
        """Model health score (0-1)"""
        if self.current_metrics is None:
            return 0.0
            
        # Weighted health score
        weights = {
            'accuracy': 0.25,
            'sharpe_ratio': 0.25,
            'win_rate': 0.20,
            'max_drawdown': 0.15,
            'avg_confidence': 0.15
        }
        
        # Normalize metrics to 0-1 scale
        normalized_accuracy = min(self.current_metrics.accuracy / 0.8, 1.0)  # 80% accuracy = 1.0
        normalized_sharpe = min(abs(self.current_metrics.sharpe_ratio) / 2.0, 1.0)  # Sharpe 2.0 = 1.0
        normalized_win_rate = min(self.current_metrics.win_rate / 0.6, 1.0)  # 60% win rate = 1.0
        normalized_drawdown = max(0, 1.0 - self.current_metrics.max_drawdown / 0.2)  # 20% drawdown = 0.0
        normalized_confidence = self.current_metrics.avg_confidence
        
        health_score = (
            weights['accuracy'] * normalized_accuracy +
            weights['sharpe_ratio'] * normalized_sharpe +
            weights['win_rate'] * normalized_win_rate +
            weights['max_drawdown'] * normalized_drawdown +
            weights['avg_confidence'] * normalized_confidence
        )
        
        # Drift penalty
        if self.drift_detected:
            health_score *= 0.7  # 30% penalty for drift
            
        return min(health_score, 1.0)
        
    def _generate_recommendations(self) -> List[str]:
        """Tavsiyalar generatsiya qilish"""
        recommendations = []
        
        if self.current_metrics is None:
            return ['Ma\'lumot yetarli emas']
            
        # Accuracy past
        if self.current_metrics.accuracy < 0.55:
            recommendations.append('Model accuracy past - yangi ma\'lumotlar bilan qayta o\'rgatish kerak')
            
        # Sharpe ratio past
        if abs(self.current_metrics.sharpe_ratio) < 0.5:
            recommendations.append('Risk-adjusted return past - risk management parametrlarini qayta sozlash kerak')
            
        # Win rate past
        if self.current_metrics.win_rate < 0.4:
            recommendations.append('Yutish foizi past - strategiya parametrlarini qayta sozlash kerak')
            
        # Max drawdown yuqori
        if self.current_metrics.max_drawdown > 0.08:
            recommendations.append('Maksimal yo\'qotish yuqori - position sizing'ni kamaytirish kerak')
            
        # Confidence past
        if self.current_metrics.avg_confidence < 0.6:
            recommendations.append('Model confidence past - yangi feature'lar qo\'shish kerak')
            
        # Drift detected
        if self.drift_detected:
            recommendations.append('Model drift aniqlandi - backup strategiyaga o\'tish tavsiya etiladi')
            
        if not recommendations:
            recommendations.append('Model yaxshi holatda - monitoring davom etsin')
            
        return recommendations
        
    def reset_baseline(self):
        """Baseline metrics'ni qayta o'rnatish"""
        self.baseline_metrics = self.current_metrics
        self.drift_detected = False
        self.logger.info("Baseline metrics reset")
        
    def get_performance_report(self) -> Dict:
        """Performance hisobot"""
        if not self.performance_history:
            return {'message': 'Performance history mavjud emas'}
            
        # Oxirgi 24 soat
        last_24h = [
            p for p in self.performance_history 
            if p['timestamp'] > datetime.now() - timedelta(hours=24)
        ]
        
        # Oxirgi 7 kun
        last_7d = [
            p for p in self.performance_history 
            if p['timestamp'] > datetime.now() - timedelta(days=7)
        ]
        
        return {
            'current_status': self.get_model_health(),
            'last_24h_performance': self._analyze_period(last_24h),
            'last_7d_performance': self._analyze_period(last_7d),
            'drift_incidents': len([p for p in self.performance_history if p['drift_detected']]),
            'total_predictions': len(self.predictions),
            'monitoring_since': self.performance_history[0]['timestamp'] if self.performance_history else None
        }
        
    def _analyze_period(self, period_data: List[Dict]) -> Dict:
        """Muayyan davr tahlili"""
        if not period_data:
            return {'message': 'Ma\'lumot mavjud emas'}
            
        accuracies = [p['metrics'].accuracy for p in period_data]
        sharpe_ratios = [p['metrics'].sharpe_ratio for p in period_data]
        
        return {
            'avg_accuracy': np.mean(accuracies),
            'avg_sharpe': np.mean(sharpe_ratios),
            'accuracy_trend': 'improving' if accuracies[-1] > accuracies[0] else 'declining',
            'drift_detected': any(p['drift_detected'] for p in period_data),
            'data_points': len(period_data)
        }
        
    def export_metrics(self, filepath: str):
        """Metrics'ni faylga eksport qilish"""
        export_data = {
            'baseline_metrics': self.baseline_metrics.__dict__ if self.baseline_metrics else None,
            'current_metrics': self.current_metrics.__dict__ if self.current_metrics else None,
            'performance_history': [
                {
                    'timestamp': p['timestamp'].isoformat(),
                    'metrics': p['metrics'].__dict__,
                    'drift_detected': p['drift_detected']
                }
                for p in self.performance_history
            ],
            'configuration': {
                'window_size': self.window_size,
                'drift_threshold': self.drift_threshold
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
            
        self.logger.info(f"Metrics exported to {filepath}")


# Singleton instance
model_monitor = ModelMonitor()


def get_model_monitor() -> ModelMonitor:
    """Global model monitor instance"""
    return model_monitor
