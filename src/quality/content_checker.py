import re
from typing import Dict, List, Tuple
import jieba
from collections import Counter

class ContentChecker:
    def __init__(self):
        # 扩展质量检查规则
        self.rules = {
            'length': {
                'min': 100,
                'max': 3000,
                'paragraph_min': 20,  # 每段最少字数
                'title_max': 30      # 标题最大字数
            },
            'required_fields': [
                'time', 'location', 'event_type', 'impact', 'measures'
            ],
            'forbidden_words': {
                '谣言', '未证实', '据说', '疑似', '可能', '不明', '未知',
                '据传', '网传', '或许', '大概', '也许', '可能会'
            },
            'required_elements': {
                'title': True,       # 必须有标题
                'time': True,        # 必须有时间
                'source': True,      # 必须有来源
                'paragraphs': 2      # 至少2个段落
            },
            'style_rules': {
                'max_sentence_length': 100,  # 单句最大长度
                'max_paragraph_length': 500, # 段落最大长度
                'min_content_density': 0.6   # 最小内容密度（实词/总词数）
            }
        }
        
    def check_content(self, news_content: str, news_info: Dict) -> Tuple[bool, List[str]]:
        """检查新闻内容质量"""
        issues = []
        
        # 基本检查
        basic_issues = self._check_basic(news_content, news_info)
        issues.extend(basic_issues)
        
        # 格式检查
        format_issues = self._check_format(news_content)
        issues.extend(format_issues)
        
        # 风格检查
        style_issues = self._check_style(news_content)
        issues.extend(style_issues)
        
        # 内容密度检查
        density_issues = self._check_content_density(news_content)
        issues.extend(density_issues)
        
        return len(issues) == 0, issues
    
    def _check_basic(self, content: str, info: Dict) -> List[str]:
        """基本检查"""
        issues = []
        
        # 长度检查
        length = len(content)
        if length < self.rules['length']['min']:
            issues.append(f"新闻内容过短（当前{length}字符，最少需要{self.rules['length']['min']}字符）")
        elif length > self.rules['length']['max']:
            issues.append(f"新闻内容过长（当前{length}字符，最多允许{self.rules['length']['max']}字符）")
        
        # 必需字段检查
        for field in self.rules['required_fields']:
            if field not in info or not info[field]:
                issues.append(f"缺少必需信息：{field}")
        
        # 禁用词检查
        for word in self.rules['forbidden_words']:
            if word in content:
                issues.append(f"包含不当词汇：{word}")
        
        return issues
    
    def _check_format(self, content: str) -> List[str]:
        """格式检查"""
        issues = []
        
        # 标题检查
        if not re.match(r'^【.*】', content):
            issues.append("缺少规范的新闻标题")
        else:
            title = re.match(r'^【(.*)】', content).group(1)
            if len(title) > self.rules['length']['title_max']:
                issues.append(f"标题过长（当前{len(title)}字符，最多允许{self.rules['length']['title_max']}字符）")
        
        # 段落检查
        paragraphs = content.split('\n')
        if len(paragraphs) < self.rules['required_elements']['paragraphs']:
            issues.append(f"段落数量不足（当前{len(paragraphs)}段，至少需要{self.rules['required_elements']['paragraphs']}段）")
        
        # 标点符号检查
        if re.search(r'[。，、；：？！]{2,}', content):
            issues.append("存在重复标点符号")
        
        return issues
    
    def _check_style(self, content: str) -> List[str]:
        """风格检查"""
        issues = []
        
        # 句子长度检查
        sentences = re.split(r'[。！？]', content)
        for sentence in sentences:
            if len(sentence.strip()) > self.rules['style_rules']['max_sentence_length']:
                issues.append(f"存在过长句子：{sentence[:30]}...")
        
        # 段落长度检查
        paragraphs = content.split('\n')
        for paragraph in paragraphs:
            if len(paragraph.strip()) > self.rules['style_rules']['max_paragraph_length']:
                issues.append(f"存在过长段落：{paragraph[:30]}...")
        
        return issues
    
    def _check_content_density(self, content: str) -> List[str]:
        """内容密度检查"""
        issues = []
        
        # 分词
        words = list(jieba.cut(content))
        # 统计虚词（可以扩展）
        function_words = {'的', '了', '着', '和', '与', '及', '或', '而', '但'}
        content_words = [w for w in words if w not in function_words and len(w.strip()) > 0]
        
        # 计算内容密度
        density = len(content_words) / len(words) if words else 0
        if density < self.rules['style_rules']['min_content_density']:
            issues.append(f"内容密度过低（当前{density:.2f}，最低要求{self.rules['style_rules']['min_content_density']}）")
        
        return issues
    
    def add_rule(self, rule_type: str, rule_content: any) -> None:
        """添加新规则"""
        if rule_type in self.rules:
            if isinstance(self.rules[rule_type], dict):
                self.rules[rule_type].update(rule_content)
            elif isinstance(self.rules[rule_type], (list, set)):
                if isinstance(self.rules[rule_type], list):
                    self.rules[rule_type].extend(rule_content)
                else:
                    self.rules[rule_type].update(rule_content)
        else:
            self.rules[rule_type] = rule_content