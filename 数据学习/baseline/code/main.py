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
    d_train=pd.read_csv('../data/train.csv',encoding="utf_8")# 训练数据集
    d_valid=pd.read_csv('../data/valid.csv',encoding="utf_8")# 验证数据集
    d_test=pd.read_csv('../data/test.csv',encoding="utf_8")# 测试数据集
    data=readData(d_train ,d_valid, d_test)
    train,valid,test,train_features,valid_features,test_features=data
    print("--------------------------------------")
    columns = d_train.columns.values.tolist() # 获取评论类别
    f1=0
    for col in tqdm(columns[1:]):
        print(col)
        train_label=train[col]
        valid_label=valid[col]
        mymodel=model(train_features,train_label,modelname="SVM")
        mymodel.train()
        preds=mymodel.predict(valid_features)
        ff=f1_score(valid_label, preds, average='macro')
        print('f1=%s'%ff)
        f1+=ff
        d_test[col]=mymodel.predict(test_features)
    print('平均 f1值=%s'%(f1/10))
    d_test.to_csv('../data/predict.csv')
    print('预测结果保存至 data/predict.csv')
