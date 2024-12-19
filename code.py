#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import necessary (basic) libraries
import pandas as pd
import numpy as np


# In[ ]:


# loading data
data=pd.read_csv("C:/Users/USER/Downloads/Statistical and AI Techniques in Data Mining/Project/heart.csv")
data


# In[ ]:


data.describe()


# In[ ]:


data.isnull().sum()


# No null values in dataframe.

# In[ ]:


# show all columns
data.columns


# Categorical Variables : sex, cp, fbs, restecg, exng, slp, caa, thall, output <n>
# Continuous Variables : age, trtbps, chol, thalachh, oldpeak 

# In[ ]:


# data having all deaths
data_death=data[data['output']==1]


# In[ ]:


# will classify all quantitaive columns according to quantiles of data with heart disease
data_death.describe(percentiles=(0.25,0.5,0.75))


# In[ ]:


# transform all continuos columns according to their quantiles in data portion with all had heart disease
## Contimuos
# age
for i in range(len(data)):
    if data['age'][i]<=44:
        data['age'][i]='age<=44'
    elif data['age'][i]>44 and data['age'][i]<=52:
        data['age'][i]='44<age<=52'
    elif data['age'][i]>52 and data['age'][i]<=59:
        data['age'][i]='52<age<=59'
    else:
        data['age'][i]='age>59'
        
        
        
# trtbps
for i in range(len(data)):
    if data['trtbps'][i]<120:
        data['trtbps'][i]='trtbps<=120'
    elif data['trtbps'][i]>120 and data['trtbps'][i]<130:
        data['trtbps'][i]='120<trtbps<=130'
    elif data['trtbps'][i]>130 and data['trtbps'][i]<140:
        data['trtbps'][i]='130<trtbps<=140'
    else:
        data['trtbps'][i]='trtbps>140'
        
        
# chol 208, 234, 267
for i in range(len(data)):
    if data['chol'][i]<208:
        data['chol'][i]='chol<=208'
    elif data['chol'][i]>208 and data['chol'][i]<234:
        data['chol'][i]='208<chol<=234'
    elif data['chol'][i]>234 and data['chol'][i]<267:
        data['chol'][i]='234<chol<=267'
    else:
        data['chol'][i]='chol>267'
        
        
# thalachh 149, 161, 172
for i in range(len(data)):
    if data['thalachh'][i]<149:
        data['thalachh'][i]='thalachh<=149'
    elif data['thalachh'][i]>149 and data['thalachh'][i]<161:
        data['thalachh'][i]='149<thalachh<=161'
    elif data['thalachh'][i]>161 and data['thalachh'][i]<172:
        data['thalachh'][i]='161<thalachh<=172'
    else:
        data['thalachh'][i]='thalachh>172'
        
    
# oldpeak 0, 0.2, 1
for i in range(len(data)):
    if data['oldpeak'][i]<0:
        data['oldpeak'][i]='oldpeak<=0'
    elif data['oldpeak'][i]>0 and data['oldpeak'][i]<0.2:
        data['oldpeak'][i]='0<oldpeak<=0.2'
    elif data['oldpeak'][i]>0.2 and data['oldpeak'][i]<1:
        data['oldpeak'][i]='0.2<oldpeak<=1'
    else:
        data['oldpeak'][i]='oldpeak>1'


# In[ ]:


# transform all categorical columns according to their categories in data portion with all had heart disease
# we will check unique values in feature first in each case
## Categorical


# In[ ]:


#sex
np.unique(data['sex'], return_counts=True)


# In[ ]:


for i in range(len(data)):
    if data['sex'][i]==1:
        data['sex'][i]='male'
    else:
        data['sex'][i]='female'


# In[ ]:


# cp
np.unique(data['cp'], return_counts=True)


# In[ ]:


for i in range(len(data)):
    if data['cp'][i]==1:
        data['cp'][i]='typical angina'
    elif data['cp'][i]==2:
        data['cp'][i]='atypical angina'
    elif data['cp'][i]==3:
        data['cp'][i]='non-anginal pain'
    else:
        data['cp'][i]='asymptomatic'


# In[ ]:


# fbs
np.unique(data['fbs'], return_counts=True)


# In[ ]:


for i in range(len(data)):
    if data['fbs'][i]==1:
        data['fbs'][i]='fbs>120 mg/dl'
    else:
        data['fbs'][i]='fbs <= 120 mg/dl'


# In[ ]:


# restecg
np.unique(data['restecg'], return_counts=True)


# In[ ]:


for i in range(len(data)):
    if data['restecg'][i]==0:
        data['restecg'][i]='normal'
    elif data['restecg'][i]==1:
        data['restecg'][i]='ST-T wave abnormality'
    else:
        data['restecg'][i]='definite left ventricular hypertrophy'


# In[ ]:


# exng
np.unique(data['exng'], return_counts=True)


# In[ ]:


for i in range(len(data)):
    if data['exng'][i]==1:
        data['exng'][i]='yes'
    else:
        data['exng'][i]='no'


# In[ ]:


# slp
np.unique(data['slp'], return_counts=True)


# In[ ]:


for i in range(len(data)):
    if data['slp'][i]==0:
        data['slp'][i]='downsloping'
    elif data['slp'][i]==1:
        data['slp'][i]='flat'
    else:
        data['slp'][i]='upsloping'


# In[ ]:


# caa
np.unique(data['caa'], return_counts=True)


# In[ ]:


for i in range(len(data)):
    if data['caa'][i]==0:
        data['caa'][i]='caa=0'
    elif data['caa'][i]==1:
        data['caa'][i]='caa=1'
    elif data['caa'][i]==2:
        data['caa'][i]='caa=2'
    elif data['caa'][i]==3:
        data['caa'][i]='caa=3'
    else:
        data['caa'][i]='caa=4'


# In[ ]:


# thall
np.unique(data['thall'], return_counts=True)


# In[ ]:


for i in range(len(data)):
    if data['thall'][i]==1:
        data['thall'][i]='no effect'
    elif data['thall'][i]==1:
        data['thall'][i]='fixed defect'
    elif data['thall'][i]==2:
        data['thall'][i]='normal'
    else:
        data['thall'][i]='reversable defect'


# In[ ]:


# following is showing all features used in our case
data.columns


# In[ ]:


# divide the dataset into two parts - one having heart disease to all members, another one is not having heart disease to anyone of members
data_1 = data[data['output']==1]
data_0 = data[data['output']==0]


# In[ ]:


# dropping 'output' column, as dividing data according to that
data_1.drop('output',axis=1,inplace=True)
data_0.drop('output',axis=1,inplace=True)


# In[ ]:


# dataset - all members having heart disease
data_1


# In[ ]:


# dataset - no member having heart disease
data_0


# In[ ]:


# to fix minsupcount and minconfidence, we will check unique values of each item for both datasets data_1 and data_0
## for data_1
print('For data_1:')
for i in range(len(data_1.columns)):
    print(data_1.columns[i],'---------------->', data_1[data_1.columns[i]].value_counts())


# For data_1, minsupcount = ....., minconfidence = .......

# In[ ]:


minsupcount_1 = 30
minconfidence_1 = 0.5


# In[ ]:


## for data_0
print('For data_0:')
for i in range(len(data_0.columns)):
    print(data_0.columns[i],'---------------->', data_0[data_0.columns[i]].value_counts())


# For data_0, minsupcount = ....., minconfidence = .......

# In[ ]:


minsupcount_0 =  30
minconfidence_0 = 0.5


# In[ ]:


# data in proper format of Market Basket Analysis
data_1_ARM=pd.get_dummies(data_1)
data_0_ARM=pd.get_dummies(data_0)


# In[ ]:


data_1_ARM


# # Apriori Algorithm

# ## Heart disease = yes (data_1)

# In[ ]:


# renaming columns for apriori
data_1_ARM.rename(columns={'age_44<age<=52':"A",'age_52<age<=59':"B",'age_age<=44':"C",'age_age>59':"D",'sex_female':"E",'sex_male':"F",'cp_asymptomatic':"G",'cp_atypical angina':"H",'cp_non-anginal pain':"I",'cp_typical angina':"J",'trtbps_120<trtbps<=130':"K",'trtbps_130<trtbps<=140':"L",'trtbps_trtbps<=120':"M",'trtbps_trtbps>140':"N",'chol_208<chol<=234':"O",'chol_234<chol<=267':"P",'chol_chol<=208':"Q",'chol_chol>267':"R",'fbs_fbs <= 120 mg/dl':"S",'fbs_fbs>120 mg/dl':"T",'restecg_ST-T wave abnormality':"U",'restecg_definite left ventricular hypertrophy':"V",'restecg_normal':"W",'thalachh_149<thalachh<=161':"X",'thalachh_161<thalachh<=172':"Y",'thalachh_thalachh<=149':"Z",'thalachh_thalachh>172':"a",'exng_no':"b",'exng_yes':"c",'oldpeak_0.2<oldpeak<=1':"d",'oldpeak_0<oldpeak<=0.2':"e",'oldpeak_oldpeak>1':"f",'slp_downsloping':"g",'slp_flat':"h",'slp_upsloping':"i",'caa_caa=0':"j",'caa_caa=1':"k",'caa_caa=2':"l",'caa_caa=3':"m",'caa_caa=4':"n",'thall_no effect':"o",'thall_normal':"p",'thall_reversable defect':"q"},inplace=True)


# In[ ]:


data_trns_1_ARM = dict()
items_1_ARM = []
for i in range(len(data_1_ARM)):
    s = []
    for j in range(len(data_1_ARM.columns)):
        if data_1_ARM[data_1_ARM.columns[j]][i]==1:
            s.append(data_1_ARM.columns[j])
            
    data_trns_1_ARM[i] = s
    items_1_ARM.append(s)


# In[ ]:


data_trns_1_ARM


# In[ ]:


items_1_ARM


# In[ ]:


elements_1 = list(data_1_ARM.columns)


# In[ ]:


data_dict_1_ARM = data_1_ARM.to_dict('list')
data_dict_1_ARM


# In[ ]:


minsupcount = minsupcount_1
itemset = []
for ele in elements_1:
    itemset.append(list(ele))
k = 1
while len(itemset)>=1:
    print()
    print("Pass_no:",k)
    print("itemset: ",itemset)
    print("    C",k,":")
    
    # getting frequent itemsets
    s=[]
    for i in range(len(itemset)):
        c=[]
        for m in range(len(data_dict_1_ARM)):
            b=[]
            for n in range(k):
                g = elements_1.index(itemset[i][n])
                b.append(data_dict_1_ARM[elements_1[g]][m])
            if all(b) is True:
                c.append(b)
        print(itemset[i],"  ",len(c))
        if len(c)<minsupcount:
            s.append(i)
            
    # frequent elements_1 for next level
    print("    Prunning step:")
    print("    L",k,":")
    new_itemset = []
    for i in range(len(itemset)):
        if i not in s:
            new_itemset.append(list(itemset[i]))
    number_of_freq_itemsets = len(new_itemset)
    print(new_itemset)
    itemset = []
    for i in range(len(new_itemset)):
        for j in range(i+1,len(new_itemset)):
            temp = []
            for m in range(k):
                if m != k-1:
                    if new_itemset[i][m] != new_itemset[j][m]:
                        break
                    else:
                        temp.extend(new_itemset[i][m])

            else:
                temp.extend([new_itemset[i][m],new_itemset[j][m]])
            itemset.append(temp)
    itemset = [ele for ele in itemset if ele!=[]]
    
    k += 1    


# In[ ]:


minsupcount_1


# In[ ]:





# ## Heart disease = no (data_0)

# In[ ]:


data_0_ARM.rename(columns={'age_44<age<=52':"A",'age_52<age<=59':"B",'age_age<=44':"C",'age_age>59':"D",'sex_female':"E",'sex_male':"F",'cp_asymptomatic':"G",'cp_atypical angina':"H",'cp_non-anginal pain':"I",'cp_typical angina':"J",'trtbps_120<trtbps<=130':"K",'trtbps_130<trtbps<=140':"L",'trtbps_trtbps<=120':"M",'trtbps_trtbps>140':"N",'chol_208<chol<=234':"O",'chol_234<chol<=267':"P",'chol_chol<=208':"Q",'chol_chol>267':"R",'fbs_fbs <= 120 mg/dl':"S",'fbs_fbs>120 mg/dl':"T",'restecg_ST-T wave abnormality':"U",'restecg_definite left ventricular hypertrophy':"V",'restecg_normal':"W",'thalachh_149<thalachh<=161':"X",'thalachh_161<thalachh<=172':"Y",'thalachh_thalachh<=149':"Z",'thalachh_thalachh>172':"a",'exng_no':"b",'exng_yes':"c",'oldpeak_0.2<oldpeak<=1':"d",'oldpeak_0<oldpeak<=0.2':"e",'oldpeak_oldpeak>1':"f",'slp_downsloping':"g",'slp_flat':"h",'slp_upsloping':"i",'caa_caa=0':"j",'caa_caa=1':"k",'caa_caa=2':"l",'caa_caa=3':"m",'caa_caa=4':"n",'thall_no effect':"o",'thall_normal':"p",'thall_reversable defect':"q"},inplace=True)


# In[ ]:


data_trns_0_ARM = dict()
items_0_ARM = []
for i in range(len(data_1_ARM),len(data)):
    s = []
    for j in range(len(data_0_ARM.columns)):
        if data_0_ARM[data_0_ARM.columns[j]][i]==1:
            s.append(data_0_ARM.columns[j])
            
    data_trns_0_ARM[i] = s
    items_0_ARM.append(s)


# In[ ]:


data_trns_0_ARM


# In[ ]:


items_0_ARM


# In[ ]:


elements_0 = list(data_0_ARM.columns)


# In[ ]:


data_dict_0_ARM = data_0_ARM.to_dict('list')
data_dict_0_ARM


# In[ ]:


minsupcount = minsupcount_0
itemset = []
for ele in elements_0:
    itemset.append(list(ele))
k = 1
while len(itemset)>=1:
    print()
    print("Pass_no:",k)
    print("itemset: ",itemset)
    print("    C",k,":")
    
    # getting frequent itemsets
    s=[]
    for i in range(len(itemset)):
        c=[]
        for m in range(len(data_dict_0_ARM)):
            b=[]
            for n in range(k):
                g = elements_0.index(itemset[i][n])
                b.append(data_dict_0_ARM[elements_0[g]][m])
            if all(b) is True:
                c.append(b)
        print(itemset[i],"  ",len(c))
        if len(c)<minsupcount:
            s.append(i)
            
    # frequent elements for next level
    print("    Prunning step:")
    print("    L",k,":")
    new_itemset = []
    for i in range(len(itemset)):
        if i not in s:
            new_itemset.append(list(itemset[i]))
    number_of_freq_itemsets = len(new_itemset)
    print(new_itemset)
    itemset = []
    for i in range(len(new_itemset)):
        for j in range(i+1,len(new_itemset)):
            temp = []
            for m in range(k):
                if m != k-1:
                    if new_itemset[i][m] != new_itemset[j][m]:
                        break
                    else:
                        temp.extend(new_itemset[i][m])

            else:
                temp.extend([new_itemset[i][m],new_itemset[j][m]])
            itemset.append(temp)
    itemset = [ele for ele in itemset if ele!=[]]
    
    k += 1    


# In[ ]:





# ## FP Growth

# In[ ]:


# using library
## for data_1
from fpgrowth_py import fpgrowth
freqItemSet, rules = fpgrowth(items_1_ARM, minSupRatio=0.65, minConf=0.5)
print(rules)


# In[ ]:


## for data_0
from fpgrowth_py import fpgrowth
freqItemSet, rules = fpgrowth(items_0_ARM, minSupRatio=0.65, minConf=0.5)
print(rules)


# In[ ]:





# ## Improvement Over FP Growth

# In[ ]:


# Add a vertex to the dictionary
def add_vertex(v):
  global graph
  global vertices_no
  if v in graph:
    print("Vertex ", v, " already exists.")
  else:
    vertices_no = vertices_no + 1
    graph[v] = []

# Add an edge between vertex v1 and v2 with edge weight e
def add_edge(v1, v2, e):
  global graph
  # Check if vertex v1 is a valid vertex
  if v1 not in graph:
    print("Vertex ", v1, " does not exist.")
  # Check if vertex v2 is a valid vertex
  elif v2 not in graph:
    print("Vertex ", v2, " does not exist.")
  else:
    # Since this code is not restricted to a directed or 
    # an undirected graph, an edge between v1 v2 does not
    # imply that an edge exists between v2 and v1
    temp = [v2, e]
    graph[v1].append(temp)

# Print the graph
def print_graph():
  global graph
  for vertex in graph:
    for edges in graph[vertex]:
      print(vertex, " -> ", edges[0], " edge weight: ", edges[1])


# ## data_1

# In[ ]:


itemset = list(data_1_ARM.columns)
itemlength_in_comb = 2
from itertools import combinations
temp = combinations(itemset, itemlength_in_comb)
k=1
minsupcount=minsupcount_1




while len(itemset)>=1:
    l=[]
    l1=[]
    print()
    print("Pass_no:",k)
    print("itemset: ",itemset)
    print("Here will be combination of",itemlength_in_comb, "combinations.")
    graph = {}
    # stores the number of vertices in the graph
    vertices_no = 0
    for i in range(len(itemset)):
        add_vertex(itemset[i])
     
    itemset=[]
    v=[]
    for i in list(temp):
        v.append(i)
        
    for i in v:
        c=0
        for j in range(len(data_1_ARM)):
            flag=0
            for h in range(k):
                if i[h] not in data_1[list(data_1_ARM.keys())[j]]:
                    flag=1
            if flag==0:
                c+=1
                    
        if c>=minsupcount:
            l.append([i,c])
            itemset.extend(list(i))
        
    print("Frequent itemsets with support count:")
    print(l)
    
    itemset=list(set(itemset))
    itemlength_in_comb += 1
    #print(itemlength_in_comb)
    temp = combinations(itemset, itemlength_in_comb)
    k+=1


# In[ ]:


data_1.keys()


# In[ ]:


data=


# ## data_0

# In[ ]:


list(data_0_ARM.keys())


# In[ ]:


itemset = list(data_0_ARM.keys())
itemlength_in_comb = 2
from itertools import combinations
temp = combinations(itemset, itemlength_in_comb)
k=1
minsupcount=minsupcount_0




while len(itemset)>=1:
    l=[]
    l1=[]
    print()
    print("Pass_no:",k)
    print("itemset: ",itemset)
    print("Here will be combination of",itemlength_in_comb, "combinations.")
    graph = {}
    # stores the number of vertices in the graph
    vertices_no = 0
    for i in range(len(itemset)):
        add_vertex(itemset[i])
     
    itemset=[]
    v=[]
    for i in list(temp):
        v.append(i)
        
    for i in v:
        c=0
        for j in range(len(data_0_ARM)):
            flag=0
            for h in range(k):
                if i[h] not in data_0_ARM[list(data_0_ARM.keys())[j]]:
                    flag=1
            if flag==0:
                c+=1
                    
        if c>=minsupcount:
            l.append([i,c])
            itemset.extend(list(i))
        
    print("Frequent itemsets with support count:")
    print(l)
    
    itemset=list(set(itemset))
    itemlength_in_comb += 1
    #print(itemlength_in_comb)
    temp = combinations(itemset, itemlength_in_comb)
    k+=1


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




