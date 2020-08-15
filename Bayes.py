import jieba
import os
import re
from collections import *

'''
author: Aldebran
time: 2020/8/15/ 22:16
'''

ignore_list = [",", ".", "?", "【", "】", "!", "！", "？", "。", "#", "《", "》", "%", "'", '"', "“", "“", '的', '哪儿', '到底',
               '着', '网友', '了', '中', '大', '、', '，', '为', '前', '其', '-', '在', '是', '这是', '没有', '个', '一个', '：',
               '或', '有', '起来', '也', '和', '”', '要', '为了', '而', '甚至', '如何', '以', '内', '而', '被', '从', '没', '会', '然而',
               '上', '这', '表示', '日', '月', '众所周知', '与', '把', '我们', '但', '这些', '并', '都', '因为', '更', '同样']

ignore_list = set(ignore_list)
regex = re.compile('[^\u4e00-\u9fa5]')


def filter_words(words: list):
    rlt = []
    for word in words:
        word = re.sub(regex, '', word)
        word = word.strip()
        if len(word) == 0:
            continue
        if word in ignore_list:
            continue
        ignore = False
        for ig in ignore_list:
            if word.startswith(ig):
                ignore = True
        rlt.append(word)
    return rlt


def deal_sentence(sentence: str):
    sentence = re.sub(re.compile('[\r\n\t]'), ' ', sentence)
    return sentence


class Bayes():
    def __init__(self, class_p: dict):
        self.class_p = class_p
        pass

    def set_data(self, x_train: list, y_train: list):
        print('train\n')
        self.class_word_num = defaultdict(dict)
        self.class_word_num_sum = defaultdict(int)
        self.deal(x_train, y_train)

    def predict_one(self, x_predict, current_accuracy=6):
        x_predict = deal_sentence(x_predict)
        sp = list(jieba.cut(x_predict))
        sp = filter_words(sp)
        print('预测分割结果: %s' % sp)
        print('words len: %s' % len(sp))
        class_p_rlt = dict()
        total_p = 0.0
        not_belog = True
        for class_name in self.class_p:
            print('\n分类名: %s' % class_name)
            p1 = 1
            for word in sp:
                if word not in self.class_word_num[class_name] or self.class_word_num_sum[class_name] == 0:
                    p1 = p1 * 1 / pow(10, current_accuracy)
                    continue
                not_belog = False
                print('word: ' + word)
                print('出现次数: %s' % self.class_word_num[class_name][word])
                print('总次数: %s' % self.class_word_num_sum[class_name])
                print(self.class_word_num[class_name][word] / self.class_word_num_sum[class_name])
                p1 = p1 * self.class_word_num[class_name][word] / self.class_word_num_sum[class_name]
            print('final p1 :%s' % p1)
            if not_belog:
                p1 = 0
            class_p_rlt[class_name] = p1
            total_p = total_p + p1
        for class_name in self.class_p:
            class_p_rlt[class_name] = class_p_rlt[class_name] / total_p
        return class_p_rlt

    def predict(self, x_predict: str):
        max_accuracy = 6
        while True:
            print('current_accuracy :%s\n' % max_accuracy)
            try:
                return self.predict_one(x_predict, max_accuracy)
            except Exception as e:
                print(e)
                max_accuracy = max_accuracy - 1
                continue

    def deal(self, x_train: list, y_train: list):
        for x, y in zip(x_train, y_train):
            x = deal_sentence(x)
            sp = list(jieba.cut(x))
            sp = filter_words(sp)
            for word in sp:
                if word not in self.class_word_num[y]:
                    self.class_word_num[y][word] = 1
                else:
                    self.class_word_num[y][word] = self.class_word_num[y][word] + 1
                self.class_word_num_sum[y] = self.class_word_num_sum[y] + 1
        for class_name in self.class_word_num:
            print(class_name)
            print(self.class_word_num[class_name])


if __name__ == '__main__':
    x = []
    y = []
    base_dir = r'./bayes1'
    for sub_f in os.listdir(base_dir):
        tag = sub_f
        sub_f = os.path.join(base_dir, sub_f)
        for sub_f2 in os.listdir(sub_f):
            sub_f2 = os.path.join(sub_f, sub_f2)
            sub_f2 = open(sub_f2, encoding="utf-8").read()
            x.append(sub_f2)
            y.append(tag)

    print(x)
    print(len(x))
    print(y)
    print(len(y))
    c = {"金融": 0.333, "萌宠": 0.333, "汽车": 0.333}
    b = Bayes(c)
    b.set_data(x, y)
    x_predict = '''
    放入你想测试的新闻，示例：汽车
    '''
    print(b.predict(x_predict))
