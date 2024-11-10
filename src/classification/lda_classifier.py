from gensim import corpora, models
import numpy as np
import jieba
import logging
from collections import defaultdict

class LDAClassifier:
    def __init__(self, num_topics=5, passes=20):
        """
        初始化LDA分类器
        :param num_topics: 主题数量
        :param passes: 训练迭代次数
        """
        self.num_topics = num_topics
        self.passes = passes
        self.dictionary = None
        self.lda_model = None
        self.topic_names = {
            0: 'natural_disaster',
            1: 'accident',
            2: 'public_health',
            3: 'social_security',
            4: 'other'
        }
        
    def train(self, documents):
        """
        训练LDA模型
        :param documents: 文档列表，每个文档是分词后的列表
        """
        # 创建词典
        self.dictionary = corpora.Dictionary(documents)
        
        # 创建语料库
        corpus = [self.dictionary.doc2bow(doc) for doc in documents]
        
        # 训练LDA模型
        self.lda_model = models.LdaModel(
            corpus=corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            passes=self.passes,
            alpha='auto',
            random_state=42
        )
        
    def classify(self, text):
        """
        对文本进行分类
        :param text: 待分类的文本
        :return: 事件类型
        """
        if not self.lda_model or not self.dictionary:
            raise ValueError("模型未训练")
            
        # 分词
        words = list(jieba.cut(text))
        
        # 转换为BOW表示
        bow = self.dictionary.doc2bow(words)
        
        # 获取主题分布
        topic_dist = self.lda_model.get_document_topics(bow)
        
        # 选择概率最高的主题
        max_topic = max(topic_dist, key=lambda x: x[1])
        
        return self.topic_names[max_topic[0]]
    
    def update_model(self, new_documents):
        """
        更新模型
        :param new_documents: 新的文档列表
        """
        if not self.lda_model:
            self.train(new_documents)
            return
            
        # 更新词典
        self.dictionary.add_documents(new_documents)
        
        # 创建新的语料库
        new_corpus = [self.dictionary.doc2bow(doc) for doc in new_documents]
        
        # 更新模型
        self.lda_model.update(new_corpus)
    
    def get_topic_words(self, num_words=10):
        """
        获取每个主题的关键词
        :param num_words: 每个主题返回的关键词数量
        :return: 主题-关键词字典
        """
        if not self.lda_model:
            raise ValueError("模型未训练")
            
        topic_words = {}
        for topic_id in range(self.num_topics):
            words = self.lda_model.show_topic(topic_id, num_words)
            topic_words[self.topic_names[topic_id]] = [word for word, _ in words]
            
        return topic_words 