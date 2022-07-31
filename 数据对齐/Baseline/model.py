from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer ,TfidfVectorizer
import pandas as pd
import numpy as np





def getWeight(corpus):
    #构建自定义停用词表
    stpwrdpath = "stop.txt"
    stpwrd_dic = open(stpwrdpath, 'rb')
    stpwrd_content = stpwrd_dic.read()
    #将停用词表转换为list  
    stpwrdlst = stpwrd_content.splitlines()
    stpwrd_dic.close()
    vectorizer=TfidfVectorizer( sublinear_tf = True, max_df = 0.8)
    return vectorizer.fit_transform(corpus).toarray()



def getMostSimilar(corpus):
    ww=getWeight(corpus[0])
    for i in range(1,len(corpus)):
        ww=np.hstack((ww,getWeight(corpus[i])))
    sim=ww[0,:].dot(ww[1:,:].T)
    indx=np.argmax(sim)
    return indx