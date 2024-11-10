from collections import Counter, deque
from datetime import datetime, timedelta
from .kleinberg_detector import KleinbergDetector

class BurstDetector:
    def __init__(self, window_size=1800, threshold=1.5):
        """
        初始化突发检测器
        :param window_size: 时间窗口大小（秒）
        :param threshold: 突发阈值
        """
        self.window_size = window_size
        self.threshold = threshold
        self.word_counts = Counter()
        self.time_windows = deque()
        self.min_word_count = 2
        self.kleinberg = KleinbergDetector()
        
    def detect_burst(self, words):
        """
        检测文本是否包含突发事件
        :param words: 分词后的文本
        :return: 是否检测到突发事件
        """
        current_time = datetime.now()
        current_counts = Counter(words)
        
        # 更新Kleinberg检测器
        self.kleinberg.add_document(words, current_time)
        
        # 检测突发词
        burst_words = []
        for word, count in current_counts.items():
            if count < self.min_word_count:
                continue
            
            # 使用Kleinberg模型检测突发
            bursts = self.kleinberg.detect_bursts(word, self.window_size)
            if bursts:
                burst_words.append(word)
            else:
                # 使用简单阈值作为备选方法
                historical_avg = self._get_historical_average(word)
                if count > historical_avg * self.threshold:
                    burst_words.append(word)
        
        # 更新历史数据
        self._update_time_window(current_time)
        self.time_windows.append((current_time, current_counts))
        self.word_counts.update(current_counts)
        
        return len(burst_words) >= 2
    
    def _update_time_window(self, current_time):
        """更新时间窗口，移除过期数据"""
        window_start = current_time - timedelta(seconds=self.window_size)
        
        # 移除过期的时间窗口
        while self.time_windows and self.time_windows[0][0] < window_start:
            _, old_counts = self.time_windows.popleft()
            # 更新总词频
            for word, count in old_counts.items():
                self.word_counts[word] -= count
                if self.word_counts[word] <= 0:
                    del self.word_counts[word]
    
    def _get_historical_average(self, word):
        """获取词的历史平均词频"""
        if not self.time_windows:
            return 0
        
        total_count = self.word_counts[word]
        window_count = len(self.time_windows)
        return total_count / window_count if window_count > 0 else 0