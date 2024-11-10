import os
import logging
from datetime import datetime
from event_detection.crawler import NewsSpider
from preprocessing.text_processor import TextPreprocessor
from event_detection.burst_detector import BurstDetector
from classification.event_classifier import EventClassifier
from generation.news_generator import NewsGenerator
from preprocessing.info_extractor import InfoExtractor
from quality.content_checker import ContentChecker

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('test.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('test_system')

def test_crawler():
    logger.info("开始测试爬虫模块...")
    try:
        spider = NewsSpider()
        url = "https://www.xinhuanet.com/politics/"
        spider.driver.get(url)
        content = spider.driver.page_source
        logger.info(f"成功获取页面内容，长度: {len(content)}")
        return content
    except Exception as e:
        logger.error(f"爬虫测试失败: {e}")
        return None

def test_preprocessing(content):
    logger.info("开始测试预处理模块...")
    try:
        processor = TextPreprocessor()
        clean_text = processor.clean_content(content)
        words = processor.segment_text(clean_text)
        logger.info(f"清理后文本长度: {len(clean_text)}")
        logger.info(f"分词结果示例: {words[:10]}")
        return clean_text, words
    except Exception as e:
        logger.error(f"预处理测试失败: {e}")
        return None, None

def test_burst_detection(words):
    logger.info("开始测试突发检测模块...")
    try:
        detector = BurstDetector()
        burst_events = detector.detect_burst(words)
        logger.info(f"检测到的突发事件: {burst_events}")
        return burst_events
    except Exception as e:
        logger.error(f"突发检测测试失败: {e}")
        return None

def test_classification(text):
    logger.info("开始测试分类模块...")
    try:
        classifier = EventClassifier()
        event_type = classifier.classify_event(text)
        logger.info(f"事件类型: {event_type}")
        return event_type
    except Exception as e:
        logger.error(f"分类测试失败: {e}")
        return None

def test_info_extraction(text):
    logger.info("开始测试信息提取模块...")
    try:
        extractor = InfoExtractor()
        info = extractor.extract_info(text)
        logger.info(f"提取的信息: {info}")
        return info
    except Exception as e:
        logger.error(f"信息提取测试失败: {e}")
        return None

def test_news_generation(event_type, event_info):
    logger.info("开始测试新闻生成模块...")
    try:
        generator = NewsGenerator()
        news = generator.generate_news(event_type, event_info)
        logger.info(f"生成的新闻:\n{news}")
        return news
    except Exception as e:
        logger.error(f"新闻生成测试失败: {e}")
        return None

def test_quality_check(news):
    logger.info("开始测试质量检查模块...")
    try:
        checker = ContentChecker()
        passed, issues = checker.check_content(news)
        logger.info(f"质量检查结果: {'通过' if passed else '未通过'}")
        if not passed:
            logger.warning(f"质量问题: {issues}")
        return passed, issues
    except Exception as e:
        logger.error(f"质量检查测试失败: {e}")
        return False, [str(e)]

def run_all_tests():
    logger.info("开始运行所有测试...")
    
    # 1. 爬虫测试
    content = test_crawler()
    if not content:
        logger.error("爬虫测试失败，终止测试")
        return
    
    # 2. 预处理测试
    clean_text, words = test_preprocessing(content)
    if not clean_text or not words:
        logger.error("预处理测试失败，终止测试")
        return
    
    # 3. 突发检测测试
    burst_events = test_burst_detection(words)
    if burst_events is None:
        logger.error("突发检测测试失败，终止测试")
        return
    
    # 4. 分类测试
    event_type = test_classification(clean_text)
    if not event_type:
        logger.error("分类测试失败，终止测试")
        return
    
    # 5. 信息提取测试
    event_info = test_info_extraction(clean_text)
    if not event_info:
        logger.error("信息提取测试失败，终止测试")
        return
    
    # 6. 新闻生成测试
    news = test_news_generation(event_type, event_info)
    if not news:
        logger.error("新闻生成测试失败，终止测试")
        return
    
    # 7. 质量检查测试
    passed, issues = test_quality_check(news)
    
    logger.info("所有测试完成")

if __name__ == "__main__":
    run_all_tests() 