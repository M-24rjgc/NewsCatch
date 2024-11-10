import numpy as np
from scipy.optimize import curve_fit
from datetime import datetime, timedelta
import logging
from collections import defaultdict
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger(__name__)

class TopicLifecycle:
    """话题生命周期分析"""
    
    def __init__(self):
        self.topic_data = defaultdict(list)
        self.lifecycle_params = {}
        self.model_metrics = defaultdict(list)
        
        # 模型配置
        self.models = {
            'logistic': self._fit_logistic,
            'gaussian': self._fit_gaussian,
            'polynomial': self._fit_polynomial
        }
        
        # 预测评估指标
        self.prediction_metrics = defaultdict(list)
        
    def _logistic_function(self, t, L, k, t0):
        """Logistic增长函数"""
        return L / (1 + np.exp(-k * (t - t0)))
    
    def add_topic_data(self, topic_id, timestamp, heat_value):
        """添加话题热度数据"""
        self.topic_data[topic_id].append({
            'timestamp': timestamp,
            'heat': heat_value
        })
    
    def _fit_logistic(self, data: List[Dict]) -> Tuple[Dict, float]:
        """Logistic模型拟合"""
        times = np.array([(d['timestamp'] - data[0]['timestamp']).total_seconds() / 3600 
                         for d in data])
        heats = np.array([d['heat'] for d in data])
        
        try:
            popt, _ = curve_fit(
                self._logistic_function,
                times,
                heats,
                p0=[max(heats), 0.1, np.mean(times)],
                maxfev=1000
            )
            
            L, k, t0 = popt
            predicted = self._logistic_function(times, L, k, t0)
            mse = mean_squared_error(heats, predicted)
            
            return {'L': L, 'k': k, 't0': t0}, mse
            
        except Exception as e:
            self.logger.error(f"Logistic拟合失败: {e}")
            return None, float('inf')
    
    def _fit_gaussian(self, data: List[Dict]) -> Tuple[Dict, float]:
        """高斯模型拟合"""
        times = np.array([(d['timestamp'] - data[0]['timestamp']).total_seconds() / 3600 
                         for d in data])
        heats = np.array([d['heat'] for d in data])
        
        try:
            popt, _ = curve_fit(
                self._gaussian_function,
                times,
                heats,
                p0=[max(heats), np.mean(times), np.std(times)],
                maxfev=1000
            )
            
            A, mu, sigma = popt
            predicted = self._gaussian_function(times, A, mu, sigma)
            mse = mean_squared_error(heats, predicted)
            
            return {'A': A, 'mu': mu, 'sigma': sigma}, mse
            
        except Exception as e:
            self.logger.error(f"高斯拟合失败: {e}")
            return None, float('inf')
    
    def analyze_lifecycle(self, topic_id: int) -> Optional[Dict]:
        """分析话题生命周期"""
        try:
            data = self.topic_data[topic_id]
            if len(data) < 5:
                return None
            
            best_model = None
            best_score = float('inf')
            
            # 尝试所有模型
            for model_name, fit_func in self.models.items():
                try:
                    params, score = fit_func(data)
                    if params and score < best_score:
                        best_score = score
                        best_model = (model_name, params)
                except Exception as e:
                    self.logger.warning(f"{model_name}模型拟合失败: {e}")
            
            if best_model:
                model_name, params = best_model
                result = {
                    'model': model_name,
                    'params': params,
                    'score': best_score,
                    'timestamp': datetime.now()
                }
                
                # 更新生命周期参数
                self.lifecycle_params[topic_id] = result
                
                # 记录评估指标
                self.model_metrics[topic_id].append({
                    'timestamp': datetime.now(),
                    'model': model_name,
                    'score': best_score,
                    'data_points': len(data)
                })
                
                return result
                
        except Exception as e:
            self.logger.error(f"生命周期分析失败: {e}")
            return None
    
    def predict_trend(self, topic_id: int, hours_ahead: int = 24) -> Optional[Dict]:
        """预测话题趋势"""
        try:
            params = self.lifecycle_params.get(topic_id)
            if not params:
                return None
            
            # Bootstrap预测
            predictions = []
            for _ in range(100):
                sample_data = self._bootstrap_sample(self.topic_data[topic_id])
                pred = self._predict_single(sample_data, params, hours_ahead)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            
            result = {
                'prediction': np.median(predictions, axis=0),
                'lower_bound': np.percentile(predictions, 2.5, axis=0),
                'upper_bound': np.percentile(predictions, 97.5, axis=0),
                'confidence': 0.95,
                'timestamp': datetime.now()
            }
            
            # 记录预测指标
            self.prediction_metrics[topic_id].append({
                'timestamp': datetime.now(),
                'hours_ahead': hours_ahead,
                'prediction_range': result['upper_bound'] - result['lower_bound']
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"趋势预测失败: {e}")
            return None
    
    def get_lifecycle_stats(self, topic_id: int) -> Dict:
        """获取生命周期统计信息"""
        stats = {
            'total_data_points': len(self.topic_data[topic_id]),
            'model_history': self.model_metrics[topic_id][-5:],
            'prediction_accuracy': self._calculate_prediction_accuracy(topic_id),
            'current_stage': self._determine_lifecycle_stage(topic_id)
        }
        return stats
    
    def is_burst_event(self, topic_id, threshold_growth=0.5, threshold_heat=100):
        """判断是否为突发事件"""
        try:
            if topic_id not in self.lifecycle_params:
                self.analyze_lifecycle(topic_id)
            
            params = self.lifecycle_params.get(topic_id)
            if not params:
                return False
            
            # 判断条件：
            # 1. 增长率超过阈值
            # 2. 最大热度超过阈值
            # 3. 处于增长阶段
            is_burst = (
                params['growth_rate'] > threshold_growth and
                params['max_heat'] > threshold_heat and
                params['stage'] == 'growth'
            )
            
            return is_burst
            
        except Exception as e:
            logger.error(f"突发事件判断失败: {e}")
            return False 