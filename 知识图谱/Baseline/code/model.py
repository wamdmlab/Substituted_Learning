# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor  
import sys
from sklearn.metrics import f1_score
import jieba

class model():
    def __init__(self,train_features , train_label,modelname='SVM'):
        self.train_features=train_features
        self.train_label=train_label
        self.modelname=modelname
        self.mymodel=None
        if modelname=='SVM':
            self.mymodel=SVC(kernel= "rbf",verbose=1)#kernel：参数选择有rbf, linear, poly, Sigmoid, 默认的是"RBF";ß
        elif modelname=='randomforest':
            self.mymodel=RandomForestRegressor()
    def train(self):
        print('开始训练...')
        self.mymodel.fit(self.train_features,self.train_label)
    def predict(self,valid_features):
        preds=self.mymodel.predict(valid_features)
        preds=[np.round(lb) for lb in preds ]
        return preds
