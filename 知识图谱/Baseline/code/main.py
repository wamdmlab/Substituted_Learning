# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import sys
from sklearn.metrics import f1_score
import jieba
from model import model
from utils import readData
import sys
sys.dont_write_bytecode = True
if __name__=='__main__':
    print("--------------------------------------")
    print("获取数据与特征")
    with open('../data/train.tsv','r',encoding="utf_8") as f:
        d_train=f.readlines()# 训练数据集
    with open('../data/valid.tsv','r',encoding="utf_8") as f:
        d_valid=f.readlines()# 验证数据集
    with open('../data/test.tsv','r',encoding="utf_8") as f:
        d_test=f.readlines()# 测试数据集
    data=readData(d_train[:5000] ,d_valid[:1000], d_test[:1000])
    train_features,train_label,valid_features,valid_label,test_features,test_label=data
    print("--------------------------------------")
    mymodel=model(train_features,train_label,modelname="SVM")
    mymodel.train()
    preds=mymodel.predict(valid_features)
    ff=f1_score(valid_label, preds, average='macro')
    print('f1=%s'%ff)
    preds=mymodel.predict(test_features)
    with open('../data/predict.tsv','w',encoding='utf8') as f:
        for sent,pred in zip(d_test,preds):
            f.write('%s\t%d\n'%(sent.strip(),pred))
    print('预测结果保存至 data/predict.tsv')
