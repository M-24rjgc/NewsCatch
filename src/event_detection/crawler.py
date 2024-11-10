from scrapy import Spider, Request
from datetime import datetime, timedelta
import logging
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.edge.service import Service as EdgeService
    from webdriver_manager.microsoft import EdgeChromiumDriverManager
except ImportError as e:
    raise ImportError(f"请安装必要的依赖: {e}")
import time
from typing import Optional, Dict, List, Generator, Any, Union
from scrapy.http import Request, Response
import base64
import winreg
from selenium.common.exceptions import TimeoutException
import re
import jieba
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import os
from collections import deque, defaultdict
import random
from typing import Optional, Dict, List, Set
from urllib.parse import urlparse
import aiohttp
import asyncio

class NewsSpider(Spider):
    name: str = 'news_spider'
    
    def __init__(self, *args, **kwargs) -> None:
        super(NewsSpider, self).__init__(*args, **kwargs)
        self.start_urls = [
            'https://www.xinhuanet.com/politics/',  # 使用更具体的新闻分类页面
            'https://news.163.com/domestic/',
            'https://china.chinadaily.com.cn/'
        ]
        
        try:
            edge_options = webdriver.EdgeOptions()
            
            # 基本设置
            edge_options.add_argument('--headless')
            edge_options.add_argument('--disable-gpu')
            edge_options.add_argument('--no-sandbox')
            
            # 内容处理设置
            edge_options.add_argument('--autoplay-policy=no-user-gesture-required')
            edge_options.add_argument('--disable-features=IsolateOrigins,site-per-process')
            
            # 安全设置
            edge_options.add_argument('--ignore-certificate-errors')
            edge_options.add_argument('--allow-running-insecure-content')
            edge_options.add_argument('--disable-web-security')
            edge_options.add_argument('--allow-insecure-localhost')
            
            # 性能设置
            edge_options.add_argument('--disable-dev-shm-usage')
            edge_options.add_argument('--disable-extensions')
            edge_options.add_argument('--disable-notifications')
            edge_options.add_argument('--disable-popup-blocking')
            
            # 启用JavaScript但控制其行为
            edge_options.add_experimental_option("prefs", {
                "profile.default_content_setting_values": {
                    "javascript": 1,  # 1 允许，2 阻止
                    "images": 2,  # 禁用图片加载
                    "notifications": 2,
                    "auto_select_certificate": 2,
                    "mixed_script": 1,  # 允许混合内容
                    "media_stream": 2,
                    "plugins": 2
                }
            })
            
            # 使用指定路径的Edge驱动
            service = EdgeService(executable_path="D:\\NewsCatch\\edgedriver_win64\\msedgedriver.exe")
            self.driver = webdriver.Edge(
                service=service,
                options=edge_options
            )
            
            # 设置超时和等待
            self.driver.set_page_load_timeout(20)
            self.driver.implicitly_wait(10)
            
            # 初始化分词器
            jieba.initialize()
            
            # 初始化TF-IDF向量化器
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words=self.load_stopwords()
            )
            
            self.logger.info("Edge WebDriver 初始化成功")
            
        except Exception as e:
            self.logger.error(f"WebDriver 初始化失败: {e}")
            raise
        
        # 网站特定的解析规则
        self.parse_rules = {
            'xinhuanet': {
                'list_items': '//div[contains(@class, "news-item")] | //ul[contains(@class, "dataList")]/li',
                'title': './/h3/a/text() | .//div[@class="title"]/a/text()',
                'link': './/h3/a/@href | .//div[@class="title"]/a/@href',
                'content': '//div[@id="detail"]//p/text() | //div[@class="article"]//p/text()',
                'time': '//div[contains(@class, "time")]/text() | //span[contains(@class, "date")]/text()'
            },
            'news.163': {
                'list_items': '//div[contains(@class, "news-item")] | //div[contains(@class, "ndi_main")]',
                'title': './/h3/a/text() | .//div[@class="news_title"]/a/text()',
                'link': './/a/@href',
                'content': '//div[@class="post_body"]//p/text() | //div[@class="post_content"]//p/text()',
                'time': '//div[@class="post_time_source"]/text() | //div[@class="pub_time"]/text()'
            },
            'chinadaily': {
                'list_items': '//div[contains(@class, "item-none")] | //ul[@class="newsList"]/li',
                'title': './/h2/a/text() | .//h3/a/text()',
                'link': './/h2/a/@href | .//h3/a/@href',
                'content': '//div[@id="Content"]//p/text() | //div[@class="article"]//p/text()',
                'time': '//div[@class="info"]/span/text() | //span[@class="time"]/text()'
            }
        }
        
        # 突发事件检测相关初始化
        self.time_window = timedelta(hours=24)  # 时间窗口
        self.word_frequencies = defaultdict(list)  # 词频统计
        self.burst_threshold = 2.0  # 突发阈值
        
        # 话题聚类相关初始化
        self.min_cluster_size = 3  # 最小聚类大小
        self.similarity_threshold = 0.6  # 相似度阈值
        
        # 新闻模板
        self.news_templates = {
            'natural_disaster': {
                'title': '【{type}快讯】{location}发生{event}',
                'body': '''
                    {time}，{location}发生{event}。
                    初步统计：{casualties}。
                    目前，{response}。
                    {additional_info}
                '''
            },
            'accident': {
                'title': '【突发】{location}发生{event}事故',
                'body': '''
                    {time}，{location}发生{event}事故。
                    事故造成{casualties}。
                    事故原因正在调查中。
                    {response}
                    {additional_info}
                '''
            },
            # 可以添加更多模板...
        }
        
        # 创建新闻存储目录
        self.news_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'generated_news')
        if not os.path.exists(self.news_dir):
            os.makedirs(self.news_dir)
        
        # 代理和Cookie池
        self.proxy_pool = self._init_proxy_pool()
        self.cookie_pool = self._init_cookie_pool()
        
        # 访问控制
        self.rate_limiter = defaultdict(lambda: {
            'requests': 0,
            'last_time': time.time(),
            'limit': 2  # 默认限制
        })
        
        # 自定义限制
        self.domain_limits = {
            'xinhuanet.com': {'limit': 3, 'delay': 5},
            'news.163.com': {'limit': 2, 'delay': 8},
            'chinadaily.com.cn': {'limit': 2, 'delay': 6}
        }
        
        # URL去重集合
        self.visited_urls: Set[str] = set()
        
        # 错误追踪
        self.retry_counts = defaultdict(int)
        self.max_retries = 3
        self.error_urls = defaultdict(list)
        
        # 异步会话
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def _init_session(self):
        """初始化异步HTTP会话"""
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def close(self):
        """关闭资源"""
        if self.session:
            await self.session.close()
        if hasattr(self, 'driver'):
            self.driver.quit()
    
    def _init_proxy_pool(self) -> deque:
        """初始化代理池"""
        try:
            # 实际项目中应从代理API获取
            proxies = [
                'http://proxy1.example.com:8080',
                'http://proxy2.example.com:8080'
            ]
            return deque(proxies)
        except Exception as e:
            self.logger.error(f"初始化代理池失败: {e}")
            return deque()
    
    def _init_cookie_pool(self) -> List[Dict[str, str]]:
        """初始化Cookie池"""
        return [
            {'sessionid': 'xxx', 'token': 'yyy'},
            {'sessionid': 'aaa', 'token': 'bbb'}
        ]
    
    def _rotate_proxy(self) -> Optional[str]:
        """轮换代理"""
        if self.proxy_pool:
            current = self.proxy_pool.popleft()
            self.proxy_pool.append(current)
            return current
        return None
    
    def _get_random_cookie(self) -> Dict[str, str]:
        """随机获取Cookie"""
        return random.choice(self.cookie_pool) if self.cookie_pool else {}
    
    def _check_rate_limit(self, url: str) -> float:
        """检查并计算需要等待的时间"""
        domain = urlparse(url).netloc
        now = time.time()
        rate_info = self.rate_limiter[domain]
        domain_config = self.domain_limits.get(domain, {'limit': 2, 'delay': 5})
        
        if now - rate_info['last_time'] > 60:
            rate_info['requests'] = 0
            rate_info['last_time'] = now
            return domain_config['delay']
            
        if rate_info['requests'] >= domain_config['limit']:
            wait_time = 60 - (now - rate_info['last_time']) + domain_config['delay']
            rate_info['requests'] = 0
            rate_info['last_time'] = now + wait_time
            return wait_time
            
        rate_info['requests'] += 1
        return domain_config['delay']
    
    async def fetch_url(self, url: str) -> Optional[str]:
        """异步获取URL内容"""
        if url in self.visited_urls:
            return None
            
        wait_time = self._check_rate_limit(url)
        await asyncio.sleep(wait_time)
        
        try:
            proxy = self._rotate_proxy()
            cookies = self._get_random_cookie()
            
            async with self.session.get(
                url,
                proxy=proxy,
                cookies=cookies,
                timeout=30
            ) as response:
                if response.status == 200:
                    self.visited_urls.add(url)
                    return await response.text()
                else:
                    self.error_urls[response.status].append(url)
                    return None
                    
        except Exception as e:
            self.logger.error(f"获取URL失败 {url}: {e}")
            self.retry_counts[url] += 1
            if self.retry_counts[url] < self.max_retries:
                await asyncio.sleep(2 ** self.retry_counts[url])  # 指数退避
                return await self.fetch_url(url)
            return None
    
    async def process_url(self, url: str) -> Dict:
        """处理单个URL"""
        content = await self.fetch_url(url)
        if not content:
            return {}
            
        try:
            # 使用现有的解析规则处理内容
            domain = urlparse(url).netloc
            rules = self.parse_rules.get(domain)
            if not rules:
                return {}
                
            # 解析内容
            result = self._parse_content(content, rules)
            result['url'] = url
            result['timestamp'] = datetime.now()
            
            return result
            
        except Exception as e:
            self.logger.error(f"处理URL失败 {url}: {e}")
            return {}
    
    async def crawl(self, start_urls: List[str]):
        """异步爬取入口"""
        await self._init_session()
        try:
            tasks = [self.process_url(url) for url in start_urls]
            results = await asyncio.gather(*tasks)
            return [r for r in results if r]
        finally:
            await self.close()
    
    def closed(self, reason: str) -> None:
        """爬虫关闭时关闭浏览器"""
        if hasattr(self, 'driver'):
            self.driver.quit()
    
    def start_requests(self) -> Generator[Request, None, None]:
        """使用Selenium获取页面内容"""
        for url in self.start_urls:
            try:
                # 增加重试次数
                for attempt in range(3):
                    try:
                        # 设置请求超时
                        self.driver.set_page_load_timeout(30)
                        
                        # 尝试访问页面
                        self.driver.get(url)
                        
                        # 等待页面加载
                        WebDriverWait(self.driver, 20).until(
                            EC.presence_of_element_located((By.TAG_NAME, "body"))
                        )
                        
                        # 处理可能的证书错误页面
                        if "证书" in self.driver.title or "SSL" in self.driver.title:
                            self.driver.execute_script(
                                'return document.getElementById("proceed-button").click()'
                            )
                            time.sleep(2)
                        
                        # 获取页面源码
                        page_source = self.driver.page_source
                        self.logger.debug(f"Page source length: {len(page_source)}")
                        
                        # 验证页面内容
                        if len(page_source) > 1000:
                            yield Request(
                                url=url,
                                callback=self.parse,
                                dont_filter=True,
                                meta={'page_source': page_source}
                            )
                            break
                        else:
                            raise Exception("Page content too short")
                        
                    except Exception as e:
                        self.logger.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                        if attempt == 2:
                            self.logger.error(f"Failed to fetch {url} after 3 attempts")
                            
            except Exception as e:
                self.logger.error(f"Error fetching {url}: {str(e)}")
    
    def parse(self, response: Response) -> Generator[Dict[str, Any], None, None]:
        """解析新闻列表页面
        
        Args:
            response: 响应对象
        
        Yields:
            Generator[Dict[str, Any], None, None]: 解析后的新闻数据
        """
        domain = self._get_domain(response.url)
        rules = self.parse_rules.get(domain)
        
        if not rules:
            self.logger.warning(f"No parse rules found for domain: {domain}")
            return
        
        try:
            # 使用Selenium查找新闻列表
            self.logger.info(f"Searching for news items on {response.url}")
            news_items = self.driver.find_elements(By.XPATH, rules['list_items'])
            self.logger.info(f"Found {len(news_items)} news items on {response.url}")
            
            for item in news_items:
                try:
                    # 提取链接和标题
                    title_element = item.find_element(By.XPATH, rules['title'])
                    title = title_element.text.strip()
                    link = title_element.get_attribute('href')
                    
                    self.logger.debug(f"Found news item: {title} ({link})")
                    
                    if link and title:
                        # 访问具体新闻页面
                        self.logger.info(f"Accessing news page: {link}")
                        self.driver.get(link)
                        
                        # 等待内容加载
                        WebDriverWait(self.driver, 10).until(
                            EC.presence_of_element_located((By.XPATH, rules['content']))
                        )
                        
                        # 提取内容
                        content_elements = self.driver.find_elements(By.XPATH, rules['content'])
                        content = '\n'.join([elem.text.strip() for elem in content_elements if elem.text.strip()])
                        
                        # 提取时间
                        try:
                            time_elem = WebDriverWait(self.driver, 5).until(
                                EC.presence_of_element_located((By.XPATH, rules['time']))
                            )
                            pub_time = time_elem.text.strip()
                        except:
                            pub_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            self.logger.warning(f"Failed to extract time for {link}, using current time")
                        
                        if content:
                            self.logger.info(f"Successfully extracted content from {link}")
                            yield {
                                'title': title,
                                'content': content,
                                'publish_time': pub_time,
                                'source_url': link,
                                'source_site': domain
                            }
                        else:
                            self.logger.warning(f"No content extracted from {link}")
                            
                except Exception as e:
                    self.logger.error(f"Error processing news item: {str(e)}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error parsing page {response.url}: {str(e)}")
    
    def _get_domain(self, url: str) -> Optional[str]:
        """从URL中提取域名"""
        if 'xinhuanet.com' in url:
            return 'xinhuanet'
        elif 'news.163.com' in url:
            return 'news.163'
        elif 'chinadaily.com.cn' in url:
            return 'chinadaily'
        return None

    def clean_content(self, content):
        """改进的内容清洗"""
        # 移除HTML标签
        content = re.sub(r'<[^>]+>', '', content)
        
        # 移除广告内容
        ad_patterns = [
            r'广告',
            r'推广',
            r'sponsored',
            r'advertisement',
            r'相关推荐',
            r'热门推荐',
            r'更多精彩',
            r'点击查看'
        ]
        for pattern in ad_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        # 移除多余空白字符
        content = re.sub(r'\s+', ' ', content).strip()
        
        # 移除特殊字符
        content = re.sub(r'[^\w\s\u4e00-\u9fff]', '', content)
        
        return content

    def extract_entities(self, text):
        """实体识别"""
        entities = {
            'persons': [],
            'locations': [],
            'organizations': [],
            'time': []
        }
        
        # 使用jieba进行词性标注
        words = pseg.cut(text)
        
        for word, flag in words:
            if flag == 'nr':  # 人名
                entities['persons'].append(word)
            elif flag == 'ns':  # 地名
                entities['locations'].append(word)
            elif flag == 'nt':  # 机构名
                entities['organizations'].append(word)
            elif flag == 't':  # 时间词
                entities['time'].append(word)
        
        # 去重
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities

    def cluster_topics(self, texts):
        """话题聚类"""
        try:
            # 将文本转换为TF-IDF向量
            X = self.vectorizer.fit_transform(texts)
            
            # 使用DBSCAN进行聚类
            clustering = DBSCAN(
                eps=0.3,  # 邻域半径
                min_samples=2,  # 最小样本数
                metric='cosine'  # 使用余弦相似度
            ).fit(X.toarray())
            
            # 获取聚类结果
            labels = clustering.labels_
            
            # 整理聚类结果
            clusters = {}
            for idx, label in enumerate(labels):
                if label == -1:  # 噪声点
                    continue
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(texts[idx])
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"话题聚类失败: {e}")
            return {}

    def load_stopwords(self):
        """加载停用词"""
        try:
            with open('data/stopwords.txt', 'r', encoding='utf-8') as f:
                return set([line.strip() for line in f])
        except Exception as e:
            self.logger.warning(f"加载停用词失败: {e}")
            return set()

    def detect_burst_event(self, text, timestamp):
        """突发事件检测"""
        words = jieba.cut(text)
        current_time = datetime.fromtimestamp(timestamp)
        burst_words = []
        
        for word in words:
            # 更新词频统计
            self.word_frequencies[word].append((current_time, 1))
            
            # 清理过期数据
            self.word_frequencies[word] = [
                (t, f) for t, f in self.word_frequencies[word]
                if current_time - t <= self.time_window
            ]
            
            # 计算突发程度
            if len(self.word_frequencies[word]) > 1:
                recent_freq = sum(f for _, f in self.word_frequencies[word][-10:])
                historical_freq = sum(f for _, f in self.word_frequencies[word][:-10])
                if historical_freq > 0:
                    burst_ratio = recent_freq / historical_freq
                    if burst_ratio > self.burst_threshold:
                        burst_words.append((word, burst_ratio))
        
        return sorted(burst_words, key=lambda x: x[1], reverse=True)

    def optimize_clustering(self, texts):
        """优化的话题聚类"""
        # 文本向量化
        vectors = self.vectorizer.fit_transform(texts)
        
        # 计算相似度矩阵
        similarity_matrix = cosine_similarity(vectors)
        
        # 优化的DBSCAN聚类
        clustering = DBSCAN(
            eps=1 - self.similarity_threshold,
            min_samples=self.min_cluster_size,
            metric='precomputed'
        ).fit(1 - similarity_matrix)
        
        # 整理聚类结果
        clusters = defaultdict(list)
        for idx, label in enumerate(clustering.labels_):
            if label != -1:  # 排除噪声点
                clusters[label].append({
                    'text': texts[idx],
                    'similarity_score': np.mean(similarity_matrix[idx][clustering.labels_ == label])
                })
        
        return dict(clusters)

    def generate_news(self, event_type, event_info):
        """新闻生成"""
        try:
            template = self.news_templates.get(event_type)
            if not template:
                self.logger.warning(f"未找到事件类型 {event_type} 的模板")
                return None
            
            # 生成标题
            title = template['title'].format(**event_info)
            
            # 生成正文
            body = template['body'].format(**event_info)
            
            # 清理格式
            body = re.sub(r'\s+', ' ', body).strip()
            
            return {
                'title': title,
                'content': body,
                'type': event_type,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            self.logger.error(f"新闻生成失败: {e}")
            return None

    def check_quality(self, news):
        """质量检查"""
        issues = []
        
        # 检查必要字段
        required_fields = ['title', 'content', 'type', 'timestamp']
        for field in required_fields:
            if not news.get(field):
                issues.append(f"缺少必要字段: {field}")
        
        # 检查内容长度
        if len(news['content']) < 100:
            issues.append("新闻内容过短")
        elif len(news['content']) > 5000:
            issues.append("新闻内容过长")
        
        # 检查敏感词
        sensitive_words = self.load_sensitive_words()
        for word in sensitive_words:
            if word in news['content']:
                issues.append(f"包含敏感词: {word}")
        
        # 检查格式规范
        if not news['title'].endswith(('。', '！', '？', '…')):
            issues.append("标题格式不规范")
        
        # 检查时间合法性
        try:
            datetime.strptime(news['timestamp'], '%Y-%m-%d %H:%M:%S')
        except ValueError:
            issues.append("时间格式不正确")
        
        return len(issues) == 0, issues

    def load_sensitive_words(self):
        """加载敏感词库"""
        try:
            with open('data/sensitive_words.txt', 'r', encoding='utf-8') as f:
                return set(line.strip() for line in f)
        except Exception as e:
            self.logger.warning(f"加载敏感词失败: {e}")
            return set()

    def save_news(self, news):
        """保存生成的新闻"""
        try:
            # 生成文件名：类型_时间戳.txt
            filename = f"{news['type']}_{news['timestamp'].replace(':', '-').replace(' ', '_')}.txt"
            filepath = os.path.join(self.news_dir, filename)
            
            # 格式化新闻内容
            content = (
                "=" * 80 + "\n"
                f"生成时间：{news['timestamp']}\n"
                f"事件类型：{news['type']}\n"
                "=" * 80 + "\n\n"
                f"{news['title']}\n\n"
                f"{news['content']}\n\n"
                "=" * 80 + "\n"
            )
            
            # 保存到文件
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.logger.info(f"新闻已保存到: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"保存新闻失败: {e}")
            return None