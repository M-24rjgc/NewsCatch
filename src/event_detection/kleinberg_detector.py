from collections import defaultdict
import numpy as np
from datetime import datetime, timedelta

class KleinbergDetector:
    def __init__(self, s=2, gamma=1):
        """
        初始化Kleinberg突发检测器
        :param s: 突发状态之间的倍数关系
        :param gamma: 状态转移代价参数
        """
        self.s = s
        self.gamma = gamma
        self.time_windows = []
        self.word_streams = defaultdict(list)
        
    def add_document(self, words, timestamp):
        """添加新文档"""
        for word in words:
            self.word_streams[word].append((timestamp, 1))
    
    def detect_bursts(self, word, window_size=3600):
        """
        检测特定词的突发区间
        :param word: 要检测的词
        :param window_size: 时间窗口大小（秒）
        :return: 突发区间列表 [(start_time, end_time, burst_level)]
        """
        if word not in self.word_streams:
            return []
            
        # 获取词的时间序列
        stream = self.word_streams[word]
        if not stream:
            return []
            
        # 计算时间窗口内的词频
        windows = self._compute_windows(stream, window_size)
        if not windows:
            return []
            
        # 使用Viterbi算法找出最可能的状态序列
        states = self._viterbi(windows)
        
        # 识别突发区间
        bursts = self._find_burst_intervals(states, windows)
        
        return bursts
    
    def _compute_windows(self, stream, window_size):
        """计算时间窗口内的词频"""
        windows = []
        current_window = []
        window_start = stream[0][0]
        
        for timestamp, count in stream:
            if (timestamp - window_start).total_seconds() <= window_size:
                current_window.append(count)
            else:
                if current_window:
                    windows.append(sum(current_window))
                current_window = [count]
                window_start = timestamp
                
        if current_window:
            windows.append(sum(current_window))
            
        return windows
    
    def _viterbi(self, windows):
        """使用Viterbi算法计算最可能的状态序列"""
        n = len(windows)
        k = int(np.log(max(windows)) / np.log(self.s)) + 1
        
        # 动态规划表
        dp = np.zeros((k, n))
        backtrack = np.zeros((k, n), dtype=int)
        
        # 初始化
        for q in range(k):
            dp[q, 0] = self._cost(windows[0], q)
            
        # 填充动态规划表
        for t in range(1, n):
            for q in range(k):
                costs = [dp[r, t-1] + self.gamma * abs(q-r) + 
                        self._cost(windows[t], q) for r in range(k)]
                dp[q, t] = min(costs)
                backtrack[q, t] = np.argmin(costs)
        
        # 回溯找出最优路径
        states = np.zeros(n, dtype=int)
        states[-1] = np.argmin(dp[:, -1])
        for t in range(n-2, -1, -1):
            states[t] = backtrack[states[t+1], t+1]
            
        return states
    
    def _cost(self, x, q):
        """计算发射概率的对数"""
        expected = self.s ** q
        return x * np.log(expected) - expected
    
    def _find_burst_intervals(self, states, windows):
        """识别突发区间"""
        bursts = []
        current_burst = None
        
        for i, state in enumerate(states):
            if state > 0:  # 突发状态
                if current_burst is None:
                    current_burst = [i, i, state]
                elif state > current_burst[2]:  # 更高级别的突发
                    current_burst[2] = state
            else:  # 非突发状态
                if current_burst is not None:
                    current_burst[1] = i - 1
                    bursts.append(tuple(current_burst))
                    current_burst = None
                    
        if current_burst is not None:
            current_burst[1] = len(states) - 1
            bursts.append(tuple(current_burst))
            
        return bursts 