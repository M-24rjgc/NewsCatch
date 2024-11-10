import os
from pathlib import Path

# Scrapy settings
BOT_NAME = 'news_crawler'

# 项目路径设置
PROJECT_ROOT = str(Path(__file__).parent.parent)
SPIDER_MODULES = ['src.event_detection']
NEWSPIDER_MODULE = 'src.event_detection'

# 爬虫设置
ROBOTSTXT_OBEY = False
CONCURRENT_REQUESTS = 1  # 降低并发数
DOWNLOAD_DELAY = 5  # 增加下载延迟
COOKIES_ENABLED = True

# 超时设置
DOWNLOAD_TIMEOUT = 60

# 重试设置
RETRY_ENABLED = True
RETRY_TIMES = 5
RETRY_HTTP_CODES = [500, 502, 503, 504, 408, 429, 403, 404]

# 输出设置
LOG_LEVEL = 'DEBUG'
LOG_FORMAT = '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
LOG_FILE = 'crawler.log'

# Item Pipeline
ITEM_PIPELINES = {
    'main.NewsPipeline': 300,
}

# 下载设置
DOWNLOAD_TIMEOUT = 180
DOWNLOAD_DELAY = 2
RANDOMIZE_DOWNLOAD_DELAY = True

# 添加以下设置
DOWNLOADER_MIDDLEWARES = {
    'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
    'scrapy.downloadermiddlewares.retry.RetryMiddleware': 90,
    'scrapy.downloadermiddlewares.httpcompression.HttpCompressionMiddleware': 810,
}

# 增加超时时间
DOWNLOAD_TIMEOUT = 180

# 增加重试次数
RETRY_TIMES = 5
RETRY_HTTP_CODES = [500, 502, 503, 504, 408, 429, 403, 404]

# 禁用robots.txt
ROBOTSTXT_OBEY = False

# 降低并发
CONCURRENT_REQUESTS = 1
CONCURRENT_REQUESTS_PER_DOMAIN = 1

# 增加下载延迟
DOWNLOAD_DELAY = 5
RANDOMIZE_DOWNLOAD_DELAY = True

# SSL/HTTPS设置
DOWNLOADER_CLIENT_TLS_METHOD = 'TLSv1.2'
DOWNLOADER_CLIENT_TLS_VERBOSE_LOGGING = True

# 禁用SSL验证
DOWNLOADER_HTTPCLIENTFACTORY = 'scrapy.core.downloader.webclient.ScrapyHTTPClientFactory'
DOWNLOADER_CLIENTCONTEXTFACTORY = 'scrapy.core.downloader.contextfactory.BrowserLikeContextFactory'

# 请求头设置
DEFAULT_REQUEST_HEADERS = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}