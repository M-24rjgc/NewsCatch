from abc import ABC, abstractmethod
from typing import Dict, Any

class PublisherBase(ABC):
    """发布器基类，定义基本接口"""
    
    @abstractmethod
    def publish(self, content: str, metadata: Dict[str, Any]) -> bool:
        """发布内容的抽象方法"""
        pass
    
    @abstractmethod
    def check_status(self) -> bool:
        """检查发布状态的抽象方法"""
        pass 