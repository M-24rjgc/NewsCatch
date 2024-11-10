import jieba
import re
import os

class TextPreprocessor:
    def __init__(self):
        # 使用自定义的中文停用词
        self.stop_words = {
            '的', '了', '和', '是', '就', '都', '而', '及', '与', '着',
            '或', '一个', '没有', '我们', '你们', '他们', '它们', '这个',
            '那个', '这些', '那些', '这样', '那样', '之', '的话', '说',
            '在', '有', '这', '那', '要', '会', '对', '能', '去', '过',
            '好', '来', '让', '被', '但', '又', '等', '已', '于', '向'
        }
        
        # 尝试加载额外的停用词
        additional_words = self.load_stopwords()
        if additional_words:
            self.stop_words.update(additional_words)
    
    def clean_text(self, text):
        if not text:
            return ""
        # 去除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        # 去除特殊字符
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def segment_text(self, text):
        if not text:
            return []
        # 分词
        words = jieba.cut(text)
        # 去停用词
        words = [w for w in words if w not in self.stop_words]
        return words
    
    def load_stopwords(self):
        """从文件加载额外的停用词"""
        try:
            stopwords_file = os.path.join(os.path.dirname(__file__), '../data/stopwords.txt')
            if os.path.exists(stopwords_file):
                with open(stopwords_file, 'r', encoding='utf-8') as f:
                    return {line.strip() for line in f if line.strip() and not line.startswith('#')}
        except Exception as e:
            print(f"加载停用词文件时出错: {e}")
        return set()