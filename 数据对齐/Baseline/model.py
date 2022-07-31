from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer ,TfidfVectorizer
import pandas as pd
import numpy as np

def getWeight(corpus):
    vectorizer=TfidfVectorizer()
    return vectorizer.fit_transform(corpus).toarray()



def getMostSimilar(corpus):
    ww=getWeight(corpus[0])
    for i in range(1,len(corpus)):
        ww=np.hstack((ww,getWeight(corpus[i])))
    sim=ww[0,:].dot(ww[1:,:].T)
    indx=np.argmax(sim)
    return indx