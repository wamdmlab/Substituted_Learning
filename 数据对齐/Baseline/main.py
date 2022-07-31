import pandas as pd
import numpy as np
from pypinyin import lazy_pinyin

from model import getMostSimilar
from utils import readData,readData_toDict

from tqdm import tqdm
import time

def getDpaper(mappingtab,papers,authorId):
    if '-' not in authorId:
        authorId=eval(authorId)
    if authorId not in mappingtab:
        return []
    engPmap=mappingtab[authorId]
    tpp=[]
    for enp in engPmap:
        pid=enp['paperId']
        if pid in papers:
            tpp.append(papers[pid])
    return tpp
def get_ch_feature(cngPaper,keys):
    AuthorsPname=[]
    for ppp in cngPaper:
        cons=[]
        for pp in ppp:
            valu=pp[keys]
            if keys=='authors':
                valu=lazy_pinyin(valu)
            cons.append(str(valu))
        AuthorsPname.append(' '.join(cons))
    return AuthorsPname
def get_en_feature(engPaper,keys):
    AuthorsPname=[]
    ss=''
    for ppp in engPaper:
        valu=ppp[keys]
        ss+=' '+(str(valu)).lower()
    AuthorsPname.append(ss)
    return AuthorsPname
if __name__ == '__main__':
    print('读取数据...')
    testData=pd.read_csv('../data/UnAlignedData.csv',encoding='utf8')
    c_mappingtab=readData('../data/c_mappingtab.csv',['authorId','authorName','paperId'])
    d_mappingtab=readData('../data/d_mappingtab.csv',['authorId','authorName','paperId'])
    c_papers=readData_toDict('../data/c_papers.csv',['paperId','title','authors','abstract','keywords','year'])
    d_papers=readData_toDict('../data/d_papers.csv',['paperId','title','authors','dblp','journal','url','year','area','level','papertype'])
    print('数据大小:%d'%len(testData))
    print('开始对齐....')
    for index in tqdm(range(len(testData))):
        ena=eval(testData.loc[index,'en_author'])
        chas=eval(testData.loc[index,'same_name'])
        engPaper=getDpaper(d_mappingtab,d_papers,ena['authorId'])
        cngPaper=[getDpaper(c_mappingtab,c_papers,ch['authorId']) for ch in chas]
########################################################################################## 获取共同作者
        a_ch_corpus=get_ch_feature(cngPaper,'authors')
        a_en_corpus=get_en_feature(engPaper,'authors')
########################################################################################### 获取论文发表年份
        y_ch_corpus=get_ch_feature(cngPaper,'year')
        y_en_corpus=get_en_feature(engPaper,'year')

########################################################################################### 领域匹配
        ar_ch_corpus=get_ch_feature(cngPaper,'keywords')
        ar_en_corpus=get_en_feature(engPaper,'area')

        t_ch_corpus=get_ch_feature(cngPaper,'title')
        t_en_corpus=get_en_feature(engPaper,'title')

############################################################################################ 匹配
        a_ch_corpus=a_en_corpus+a_ch_corpus
        y_ch_corpus=y_en_corpus+y_ch_corpus
        ar_ch_corpus=ar_en_corpus+ar_ch_corpus
        t_ch_corpus=t_en_corpus+t_ch_corpus
        sindx=getMostSimilar([a_ch_corpus,y_ch_corpus,ar_ch_corpus,t_ch_corpus])
        testData.loc[index,'ch_author']=str(chas[sindx])
    testData.to_csv('../data/Aligned_Data.csv',index=0)
    print('完成,结果在 data/Aligned_Data.csv')