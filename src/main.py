import os
import logging
from datetime import datetime
from pathlib import Path
from event_detection.crawler import NewsSpider
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from clustering.topic_cluster import TopicCluster
from lifecycle.topic_lifecycle import TopicLifecycle

# 设置日志
def setup_logging():
    """配置日志系统"""
    log_dir = Path(__file__).parent.parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f'crawler_{datetime.now().strftime("%Y%m%d")}.log'
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('news_system')

logger = setup_logging()

class NewsPipeline:
    """新闻处理管道"""
    
    def __init__(self):
        """初始化处理管道"""
        self.setup_directories()
        logger.info("所有组件初始化成功")
        self.topic_cluster = TopicCluster()
        self.lifecycle_analyzer = TopicLifecycle()
    
    def setup_directories(self):
        """创建必要的目录结构"""
        project_root = Path(__file__).parent.parent
        dirs = [
            'generated_news',
            'data',
            'logs',
            'generated_news/archive'
        ]
        
        for dir_name in dirs:
            dir_path = project_root / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"确保目录存在: {dir_path}")
    
    def process_item(self, item, spider):
        """处理新闻项"""
        try:
            # 生成文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'news_{timestamp}.txt'
            
            # 构建保存路径
            save_path = Path(__file__).parent.parent / 'generated_news' / filename
            
            # 格式化新闻内容
            content = (
                "=" * 80 + "\n"
                f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"来源：{item['url']}\n"
                f"来源网站：{item['source']}\n"
                "=" * 80 + "\n\n"
                f"标题：{item['title']}\n\n"
                f"内容：\n{item['content']}\n\n"
                "=" * 80 + "\n"
            )
            
            # 保存文件
            save_path.write_text(content, encoding='utf-8')
            logger.info(f"新闻已保存到: {save_path}")
            
            # 话题聚类
            clusters = self.topic_cluster.cluster_topics(
                [item['content']], 
                [datetime.now()]
            )
            
            # 如果属于某个已知话题
            if clusters:
                for label, cluster_items in clusters.items():
                    # 更新生命周期数据
                    self.lifecycle_analyzer.add_topic_data(
                        label,
                        datetime.now(),
                        len(cluster_items)
                    )
                    
                    # 分析生命周期
                    lifecycle = self.lifecycle_analyzer.analyze_lifecycle(label)
                    
                    # 判断是否为突发事件
                    if self.lifecycle_analyzer.is_burst_event(label):
                        logger.info(f"检测到突发事件: topic_id={label}")
                        # 处理突发事件...
            
            return item
            
        except Exception as e:
            logger.error(f"处理新闻失败: {e}")
            return item

def setup_project():
    """初始化项目环境"""
    try:
        # 创建必要的目录
        project_root = Path(__file__).parent.parent
        dirs = [
            'generated_news',
            'data',
            'logs',
            'generated_news/archive'
        ]
        
        for dir_name in dirs:
            dir_path = project_root / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"确保目录存在: {dir_path}")
            
        return True
        
    except Exception as e:
        logger.error(f"项目初始化失败: {e}")
        return False

def main():
    """主程序入口"""
    try:
        logger.info("开始初始化系统...")
        
        if not setup_project():
            logger.error("项目初始化失败")
            return
        
        # 获取设置
        settings = get_project_settings()
        settings.update({
            'LOG_LEVEL': 'DEBUG',
            'LOG_FILE': 'crawler.log',
            'ITEM_PIPELINES': {
                'main.NewsPipeline': 300  # 使用相对路径
            },
            'CONCURRENT_REQUESTS': 1,
            'DOWNLOAD_DELAY': 3,
            'COOKIES_ENABLED': True,
            'REQUEST_FINGERPRINTER_IMPLEMENTATION': '2.7'
        })
        
        # 创建爬虫进程
        logger.info("创建爬虫进程...")
        process = CrawlerProcess(settings)
        
        # 添加爬虫
        logger.info("添加新闻爬虫...")
        process.crawl(NewsSpider)
        
        # 启动爬虫
        logger.info("启动爬虫进程...")
        process.start()
        
        logger.info("爬虫任务完成")
        
    except Exception as e:
        logger.error(f"系统运行出错: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 