import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import scipy.sparse as sp
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class TopicCluster:
    """话题聚类系统"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=self._load_stopwords(),
            ngram_range=(1, 2)  # 添加二元语法支持
        )
        
        # 增量学习支持
        self.previous_vectors = None
        self.previous_labels = None
        self.topic_centroids: Dict[int, np.ndarray] = {}
        
        # 聚类历史
        self.cluster_history: List[Dict] = []
        
        # 聚类评估指标
        self.clustering_metrics: Dict[str, List[float]] = {
            'silhouette_scores': [],
            'inertia_scores': [],
            'cluster_sizes': []
        }
        
        # 优化的DBSCAN参数
        self.eps = 0.3
        self.min_samples = 2
        self.metric = 'cosine'
        
    def _load_stopwords(self):
        """加载停用词"""
        try:
            with open('data/stopwords.txt', 'r', encoding='utf-8') as f:
                return set([line.strip() for line in f])
        except Exception as e:
            logger.error(f"加载停用词失败: {e}")
            return set()
    
    def preprocess_texts(self, texts):
        """文本预处理"""
        processed_texts = []
        for text in texts:
            # 分词
            words = jieba.cut(text)
            # 去除停用词
            words = [w for w in words if w not in self._load_stopwords()]
            processed_texts.append(' '.join(words))
        return processed_texts
    
    def optimize_parameters(self, vectors: sp.csr_matrix) -> Tuple[float, int]:
        """优化聚类参数"""
        best_score = -1
        best_params = (self.eps, self.min_samples)
        
        eps_range = [0.2, 0.3, 0.4]
        min_samples_range = [2, 3, 4]
        
        for eps in eps_range:
            for min_samples in min_samples_range:
                clustering = DBSCAN(
                    eps=eps,
                    min_samples=min_samples,
                    metric=self.metric
                ).fit(vectors)
                
                if len(set(clustering.labels_)) > 1:  # 至少有一个聚类
                    score = silhouette_score(vectors, clustering.labels_)
                    if score > best_score:
                        best_score = score
                        best_params = (eps, min_samples)
        
        return best_params
    
    def cluster_topics(self, texts: List[str], timestamps: List[datetime]) -> Dict:
        """执行话题聚类"""
        try:
            # 文本向量化
            current_vectors = self.vectorizer.fit_transform(texts)
            
            # 标准化向量
            current_vectors = normalize(current_vectors)
            
            # 增量聚类
            if self.previous_vectors is not None:
                combined_vectors = sp.vstack([self.previous_vectors, current_vectors])
            else:
                combined_vectors = current_vectors
            
            # 优化参数
            eps, min_samples = self.optimize_parameters(combined_vectors)
            
            # 执行聚类
            clustering = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric=self.metric,
                n_jobs=-1
            ).fit(combined_vectors)
            
            # 更新评估指标
            if len(set(clustering.labels_)) > 1:
                self.clustering_metrics['silhouette_scores'].append(
                    silhouette_score(combined_vectors, clustering.labels_)
                )
                self.clustering_metrics['cluster_sizes'].append(
                    len(set(clustering.labels_)) - 1  # 不计入噪声
                )
            
            # 更新话题中心点
            self._update_centroids(combined_vectors, clustering.labels_)
            
            # 保存当前结果
            self.previous_vectors = current_vectors
            self.previous_labels = clustering.labels_[-len(texts):]
            
            # 格式化结果
            clusters = self._format_clusters(texts, timestamps, self.previous_labels)
            
            # 记录聚类历史
            self.cluster_history.append({
                'timestamp': datetime.now(),
                'num_clusters': len(clusters),
                'total_texts': len(texts),
                'parameters': {'eps': eps, 'min_samples': min_samples}
            })
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"话题聚类失败: {e}")
            return {}
    
    def _update_centroids(self, vectors: sp.csr_matrix, labels: np.ndarray) -> None:
        """更新话题中心点"""
        for label in set(labels):
            if label != -1:
                mask = labels == label
                centroid = vectors[mask].mean(axis=0)
                self.topic_centroids[label] = centroid.toarray().flatten()
    
    def get_similar_topics(self, text: str, threshold: float = 0.5) -> List[int]:
        """查找相似话题"""
        vector = self.vectorizer.transform([text])
        similar_topics = []
        
        for label, centroid in self.topic_centroids.items():
            similarity = cosine_similarity(vector, centroid.reshape(1, -1))[0][0]
            if similarity >= threshold:
                similar_topics.append((label, similarity))
        
        return sorted(similar_topics, key=lambda x: x[1], reverse=True)
    
    def get_clustering_stats(self) -> Dict:
        """获取聚类统计信息"""
        return {
            'total_clusters': len(self.topic_centroids),
            'avg_silhouette_score': np.mean(self.clustering_metrics['silhouette_scores']),
            'avg_cluster_size': np.mean(self.clustering_metrics['cluster_sizes']),
            'history': self.cluster_history[-10:]  # 最近10次聚类记录
        }
    
    def get_hot_topics(self, min_size=3):
        """获取热门话题"""
        hot_topics = []
        for label, cluster in self.cluster_history.items():
            if len(cluster) >= min_size:
                recent_activity = cluster[-min_size:]
                growth_rate = (recent_activity[-1]['size'] - recent_activity[0]['size']) / min_size
                hot_topics.append({
                    'label': label,
                    'size': recent_activity[-1]['size'],
                    'growth_rate': growth_rate,
                    'avg_similarity': np.mean([item['avg_similarity'] for item in recent_activity])
                })
        
        return sorted(hot_topics, key=lambda x: x['growth_rate'], reverse=True) 