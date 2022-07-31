#-*- coding: UTF-8 -*- 

import matplotlib.pyplot as plt
import numpy as np
import copy
import os
step=8
alpha=0.06
fname="example1"


step_value=[]
step_time=[]
step_now=0
step_index=[]
def delect(time,value,index):
    global step
    global step_value
    global step_time
    global step_now
    global step_index
    if step_now < step:
        step_value.append(float(value)*-1)
        step_time.append(time)
        step_index.append(index)
        step_now=step_now+1
        
    else:
        print(step_value)
        print(step_time)
        stepValueGap=(float(step_value[len(step_value)-1])*-1)-(float(step_value[0])*-1)
        stepTimeGap=float(step_time[len(step_value)-1])-float(step_time[0])
        xiedu=stepValueGap/step
        print(xiedu)
        temp_value=copy.deepcopy(step_value)
        temp_index=copy.deepcopy(step_index)
        step_index.clear()
        step_value.clear()
        step_time.clear()
		
        step_now=0
        if abs(xiedu)>alpha:
            return temp_value,temp_index
        else:
            return [],[]
    return [],[]





f = open(fname)# 返回一个文件对象  
line = f.readline()             # 调用文件的 readline()方法  
points=[]
nums=[]
i=0
abvalues=[]
abindexs=[]

values_all=[]
index_all=[]
colors_all=[]
timeline=0
while line:  
    timeline=timeline+1
    #print line,                 # 在 Python 2中，后面跟 ',' 将忽略换行符  
    #print(line, end = '')       # 在 Python 3中使用
    line = f.readline()
    strlist=line.split(' ')
    if(len(strlist) == 3):
        nums.append(i)
        i=i+1
        points.append(float(strlist[1])*-1)
        #print(strlist[0],strlist[1])
        stepvalue,stepindex=delect(strlist[0],strlist[1],i)
        if(len(stepvalue) > 0):
            abvalues.extend(stepvalue)
            abindexs.extend(stepindex)
            stepvalue.clear()
            stepindex.clear()

    if timeline%30 == 0:    
        values_all.extend(points)
        index_all.extend(nums)
        for c in range(0,len(points)):
            colors_all.append(4)
        values_all.extend(abvalues)
        index_all.extend(abindexs)
        for c in range(0,len(abvalues)):
            colors_all.append(2)
	
        plt.scatter(index_all,values_all,c=colors_all)

        plt.pause(0.001)
        plt.cla()
		
        values_all.clear()
        index_all.clear()
        colors_all.clear()
			
    
os.system('pause')
f.close()