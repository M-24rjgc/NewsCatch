from abc import ABC, abstractmethod
from typing import Dict, Any, List

class FeedbackCollector(ABC):
    """反馈收集器基类"""
    
    @abstractmethod
    def collect(self) -> List[Dict[str, Any]]:
        """收集反馈的抽象方法"""
        pass
    
    @abstractmethod
    def analyze(self, feedback: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析反馈的抽象方法"""
        pass 