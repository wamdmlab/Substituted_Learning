# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

def segmentWord(contents): 
    ret = [] 
    for sent in contents:
        word_list = list(jieba.cut(sent))
        combine_words = " ".join(word_list)
        ret.append(combine_words) 
    return ret


def readData(train ,test):
    print("训练样本 = %d" % len(train))
    print("测试样本 = %d" %len(test))
    content_train=segmentWord(train['content'])
    content_test=segmentWord(test['content'])

    vectorizer = TfidfVectorizer(analyzer='word',min_df=3,token_pattern=r"(?u)\b\w\w+\b")
    train_features = vectorizer.fit_transform(content_train)
    print("训练样本特征表长度为 %s"%str(train_features.shape))
    # print(vectorizer.get_feature_names()) #特征名展示
    test_features = vectorizer.transform(content_test)
    print("测试样本特征表长度为 %s"%str(test_features.shape))
    data=[train,test,train_features,test_features]
    return data
