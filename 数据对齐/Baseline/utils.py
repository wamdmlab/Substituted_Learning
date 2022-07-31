import pandas as pd
import numpy as np
from tqdm import tqdm

def readData(path,cols):
    rdata={}
    ddd=pd.read_csv(path,encoding='utf8')
    ddd.columns=cols
    for index,row in tqdm(ddd.iterrows()):
        if row[cols[0]] not in rdata:
            rdata[row[cols[0]]]=[row]
        else:
            rdata[row[cols[0]]].append(row)
    return rdata

def readData_toDict(path,cols):
    rdata={}
    ddd=pd.read_csv(path,encoding='utf8')
    ddd.columns=cols
    for index,row in tqdm(ddd.iterrows()):
        if row[cols[0]] not in rdata:
            rdata[row[cols[0]]]=row
    return rdata