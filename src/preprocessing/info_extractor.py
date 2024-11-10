import re
from datetime import datetime

class InfoExtractor:
    def __init__(self):
        # 地点提取模式
        self.location_patterns = [
            r'在([^，。；、]{2,10})(发生|举行|开展)',
            r'([^，。；、]{2,10})(市|省|县|区).*?发生',
        ]
        
        # 影响提取模式
        self.impact_patterns = [
            r'造成([^，。；、]+?)(损失|伤亡|影响)',
            r'导致([^，。；、]+)',
        ]
        
        # 措施提取模式
        self.measure_patterns = [
            r'已(采取|开展|进行)([^，。；、]+?(措施|行动|工作))',
            r'正在([^，。；、]+?(处理|解决|应对))',
        ]
    
    def extract_info(self, title, content):
        """提取新闻信息"""
        info = {
            'time': datetime.now().strftime('%Y年%m月%d日%H时%M分'),
            'location': self._extract_location(title, content),
            'impact': self._extract_impact(content),
            'measures': self._extract_measures(content)
        }
        return info
    
    def _extract_location(self, title, content):
        text = title + content
        for pattern in self.location_patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[0][0] if isinstance(matches[0], tuple) else matches[0]
        return '相关地区'
    
    def _extract_impact(self, content):
        for pattern in self.impact_patterns:
            matches = re.findall(pattern, content)
            if matches:
                return matches[0][0] if isinstance(matches[0], tuple) else matches[0]
        return '具体影响正在评估中'
    
    def _extract_measures(self, content):
        for pattern in self.measure_patterns:
            matches = re.findall(pattern, content)
            if matches:
                return matches[0][0] if isinstance(matches[0], tuple) else matches[0]
        return '正在采取相应措施' 