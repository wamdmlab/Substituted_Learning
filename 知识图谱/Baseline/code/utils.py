# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

def prossess(contents): 
    ret_content=[]
    ret_label=[]
    for osent in tqdm(contents):
        sent=osent.strip().split('\t')
        if len(sent)==2:
            ret_content.append(sent[0])
            ret_label.append(int(sent[1]))
        else:
            ret_content.append(osent.strip())
    return ret_content,ret_label


def readData(train ,valid, test):
    print("训练样本 = %d" % len(train))
    print("验证样本 = %d" % len(valid))
    print("测试样本 = %d" %len(test))
    content_train,train_label=prossess(train)
    content_valid,valid_label=prossess(valid)
    content_test,test_label=prossess(test)
    

    vectorizer = TfidfVectorizer(analyzer='word',min_df=3,token_pattern=r"(?u)\b\w\w+\b")
    train_features = vectorizer.fit_transform(content_train)
    print("训练样本特征表长度为 %s"%str(train_features.shape))
    # print(vectorizer.get_feature_names()) #特征名展示
    valid_features = vectorizer.transform(content_valid)
    print("验证样本特征表长度为 %s"%str(valid_features.shape))
    test_features = vectorizer.transform(content_test)
    print("测试样本特征表长度为 %s"%str(test_features.shape))
    data=[train_features,train_label,valid_features,valid_label,test_features,test_label]
    return data
