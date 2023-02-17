"""
朴素贝叶斯分类模型（高斯分布、伯努利分布、多项式分布）
@author aldebran
@since 2020/8/15/ 22:16
"""
from collections import Iterator, defaultdict
import jieba
import re
import numpy
import math

# 过滤词语, TODO 完善停止词
stop_words_pattern = re.compile("[\\s+|,|.|\"|'|!|\?|:|;|,|。|“|‘|！|？|：|；]")


# 分割文章的方法，多种方法共用我觉得也是可行的
# args是n_gram中n的范围，n属于[args[0],args[1])
def split_text(text: str, method='n_gram', *args):
    sentences = list(filter(lambda it: it != '', re.split(stop_words_pattern, text)))
    result = []
    if method == 'n_gram':
        if len(args) == 0:
            args = [1, 2]  # 默认是1-gram
        for n in range(int(args[0]), int(args[1])):
            for sentence in sentences:
                if len(sentence) < n:
                    result.append(sentence)
                else:
                    for i in range(0, len(sentence) - n):
                        result.append(sentence[i:i + n])
        pass
    elif method == 'jieba':
        for sentence in sentences:
            for word in list(filter(lambda it: it != '', jieba.cut(sentence))):
                result.append(word)
    else:
        raise Exception(f'unsupported method: {method}')
    return result
    pass


# TODO TF-IDF等算法自动过滤干扰分类的常用词

# 词关联信息
class WordRef():
    def __init__(self, count: int, tf_idf: float = 0):
        self.count = count
        self.tf_idf = tf_idf


# 文章
class Article():

    def __init__(self, article_id, text: str, class_name: str = None):
        self.article_id = article_id
        self.word_count_map = defaultdict(lambda: WordRef(0, 0))
        self.words_count = 0
        self.class_name = class_name
        self.feature = None
        for word in split_text(text):
            self.word_count_map[word].count += 1
            self.words_count += 1
        # print(self.word_count_map)
        pass

    def get_feature(self, feature_words, regenerate=False):
        if regenerate or self.feature is None:
            self.feature = numpy.array(list(map(lambda it: self.word_count_map[it].count, feature_words)))
        return self.feature
        pass


# 文章组
class Articles():

    def __init__(self, filter_interval=50):
        self.id_article_map = dict()
        self.class_name_articles_map = defaultdict(lambda: [])
        self.feature_words = []
        self.feature_words_set = set()
        self.filter_interval = filter_interval
        pass

    def add_one(self, article: Article):
        self.id_article_map[article.article_id] = article
        self.class_name_articles_map[article.class_name].append(article)
        for word in article.word_count_map:
            if word not in self.feature_words_set:
                self.feature_words_set.add(word)
                self.feature_words.append(word)
        articles_count = len(self.id_article_map)
        if articles_count % self.filter_interval == 0:
            self.filter_disturb_items()
        pass

    def add_mul(self, articles: Iterator):
        for article in articles:
            self.add_one(article)

    # TODO 过滤干扰分类的常用词
    def filter_disturb_items(self):
        pass


# 朴素贝叶斯文本分类模型
class NaiveBayesTextClassification():
    def __init__(self,
                 method: str,  # 方法: Bernoulli、Gaussian、Multinomial
                 filter_interval=50,  # 过滤周期
                 ):
        if method not in ['Bernoulli', 'Gaussian']:
            raise Exception(f'unsupported method: {method}')
        self.class_name_probability = None
        self.method = method
        self.articles = Articles(filter_interval)
        self.data_after_train = None
        pass

    def fit(self, article: Article):
        self.articles.add_one(article)

    def fit_mul(self, articles: list):
        self.articles.add_mul(articles)

    def train(self):

        # 计算特征向量
        for article in self.articles.id_article_map.values():
            article.get_feature(self.articles.feature_words, True)

        # 计算先验概率
        self.class_name_probability = dict()
        class_count = len(self.articles.class_name_articles_map)
        all_articles_count = len(self.articles.id_article_map)
        for class_name in self.articles.class_name_articles_map:
            class_articles_count = len(self.articles.class_name_articles_map[class_name])
            self.class_name_probability[class_name] = (class_articles_count + 1) / \
                                                      (all_articles_count + 1 * class_count)  # 拉普拉斯平滑
        # print(self.class_name_probability)

        features_count = len(self.articles.feature_words)
        # 条件概率所需数据
        if self.method == 'Bernoulli':
            self.data_after_train = defaultdict(lambda: numpy.zeros(shape=(features_count,)))
            for class_name in self.articles.class_name_articles_map:
                for a in self.articles.class_name_articles_map[class_name]:
                    self.data_after_train[class_name] += a.feature
                self.data_after_train[class_name] = (self.data_after_train[class_name] + 1) / \
                                                    (numpy.sum(self.data_after_train[class_name])
                                                     + 1 * features_count)  # 拉普拉斯平滑
            pass
        elif self.method == 'Gaussian':
            self.data_after_train = defaultdict(
                lambda: {'σ': numpy.zeros(shape=(features_count,)),  # 标准差
                         'μ': numpy.zeros(shape=(features_count,))})  # 期望

            for class_name in self.articles.class_name_articles_map:
                articles = self.articles.class_name_articles_map[class_name]
                array = numpy.zeros(shape=(features_count, len(articles)))
                for i, a in enumerate(articles):
                    array[:, i] = a.feature.T
                max_o = 0
                for i in range(features_count):
                    u = numpy.average(array[i])  # 期望
                    o = numpy.std(array[i])  # 标准差
                    if o > max_o: max_o = o
                    self.data_after_train[class_name]['μ'][i] = u  # 期望
                    self.data_after_train[class_name]['σ'][i] = o  # 标准差
                for i in range(features_count):
                    if self.data_after_train[class_name]['σ'][i] == 0:
                        self.data_after_train[class_name]['σ'][i] = max_o * 2
                        pass

            # print(self.train_data)

            pass

        elif self.method == 'Multinomial':
            pass

    def predict(self, article: Article):
        feature = article.get_feature(self.articles.feature_words, True)
        result = defaultdict(lambda: 1.0)
        result.update(self.class_name_probability)

        if self.method == 'Bernoulli':
            for i in range(feature.shape[0]):
                x_i = 0 if feature[i] == 0 else 1
                for class_name in self.data_after_train:
                    p_1 = self.data_after_train[class_name][i]
                    result[class_name] *= x_i * p_1 + (1 - x_i) * (1 - p_1)
            # print(result)

        elif self.method == 'Gaussian':
            for i in range(feature.shape[0]):
                x_i = feature[i]
                for class_name in self.data_after_train:
                    u = self.data_after_train[class_name]['μ'][i]
                    o = self.data_after_train[class_name]['σ'][i]
                    result[class_name] *= math.exp(-math.pow(x_i - u, 2) / (2 * math.pow(o, 2))) / (
                            math.sqrt(2 * math.pi) * o)
            pass

        s = numpy.sum(numpy.array(list(result.values())))
        for class_name in result:
            result[class_name] = result[class_name] / s

        return result
        pass

    def predict_label(self, article: Article):
        result = self.predict(article)
        result_class_name = None
        result_class_p = -1
        for class_name in result:
            class_p = result[class_name]
            if class_p > result_class_p:
                result_class_p = class_p
                result_class_name = class_name
            pass
        return result_class_name
