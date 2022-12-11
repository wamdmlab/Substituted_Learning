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
    d_train=pd.read_csv('../data/public_test.tsv',encoding="utf_8",sep='\t',names=['content','label'])# 训练数据集
    d_test=pd.read_csv('../data/noisy_data.tsv',encoding="utf_8",sep='\t',names=['content','label'])# 测试数据集
    data=readData(d_train, d_test)
    train,test,train_features,test_features=data
    print("--------------------------------------")
    train_label=train['label']
    mymodel=model(train_features,train_label,modelname="SVM")
    mymodel.train()
    preds=mymodel.predict(test_features)
    d_test['label']=mymodel.predict(test_features)
    with open('../data/refined_data.tsv','w',encoding='utf8') as f:
        for i,row in d_test.iterrows():
            f.write('%s\t%s\n'%(row['content'],row['label']))
        print('预测结果保存至 data/refined_data.tsv')
