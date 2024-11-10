import jieba.analyse
from collections import Counter
from .lda_classifier import LDAClassifier

class EventClassifier:
    def __init__(self):
        # 关键词分类器
        self.event_keywords = {
            'natural_disaster': {'地震', '台风', '洪水', '暴雨', '干旱', '山火', '泥石流'},
            'accident': {'事故', '车祸', '爆炸', '坍塌', '起火', '泄露'},
            'public_health': {'疫情', '传染病', '食品安全', '污染'},
            'social_security': {'犯罪', '恐怖', '暴力', '骚乱', '纠纷'}
        }
        
        # LDA分类器
        self.lda_classifier = LDAClassifier()
        self.is_lda_trained = False
        self.training_documents = []
        
    def classify_event(self, text):
        """
        分类事件
        :param text: 待分类的文本
        :return: 事件类型
        """
        # 使用关键词方法
        keywords = jieba.analyse.extract_tags(text, topK=20)
        type_scores = {}
        for event_type, type_keywords in self.event_keywords.items():
            score = sum(1 for word in keywords if word in type_keywords)
            type_scores[event_type] = score
            
        # 如果LDA模型已训练，结合两种方法
        if self.is_lda_trained:
            try:
                lda_type = self.lda_classifier.classify(text)
                # 增加LDA分类结果的权重
                type_scores[lda_type] = type_scores.get(lda_type, 0) + 2
            except Exception as e:
                print(f"LDA分类出错: {e}")
        
        # 存储文档用于后续训练
        words = list(jieba.cut(text))
        self.training_documents.append(words)
        
        # 当收集到足够的文档时，训练或更新LDA模型
        if len(self.training_documents) >= 100:
            self._update_lda_model()
        
        # 返回得分最高的类型
        if not type_scores:
            return 'other'
        return max(type_scores.items(), key=lambda x: x[1])[0]
    
    def _update_lda_model(self):
        """更新LDA模型"""
        try:
            if not self.is_lda_trained:
                self.lda_classifier.train(self.training_documents)
                self.is_lda_trained = True
            else:
                self.lda_classifier.update_model(self.training_documents)
            
            # 清空训练文档
            self.training_documents = []
            
        except Exception as e:
            print(f"更新LDA模型失败: {e}")