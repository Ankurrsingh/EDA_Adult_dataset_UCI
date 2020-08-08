#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("adult.csv")


# In[3]:


df.head(5)


# In[4]:


df.shape


# In[5]:


df.columns


# In[6]:


df.describe()


# In[7]:


df.info()


# In[8]:


from numpy import nan
df = df.replace('?',nan) #replacing ? with NaN
df.head(5)


# In[9]:


df.isnull()


# In[11]:


df.isnull().sum() #From here we can see that we have 2799, 2809, 857 NaN values in workclass, occupation, native-country columns


# In[12]:


df['native-country'].unique()  #Finding unique values in column native-column


# In[13]:


df['occupation'].unique()


# In[14]:


df['workclass'].unique()


# Filling The NaN value columns with mode(value most repeated ) or the first value[0] of the column 
# Its increase expressiveness of the data

# In[15]:


df['native-country'].fillna(df['native-country'].mode()[0],inplace = True)


# In[16]:


df['workclass'].fillna(df['workclass'][0],inplace = True)


# In[17]:


df['occupation'].fillna(df['occupation'].mode(),inplace = True)


# In[18]:


df.head(5)


# In[19]:


df['workclass'][0]


# In[20]:


df.isnull().sum() # Here we can see we have no Nan values left


# In[21]:


import seaborn as sns


# In[22]:


sns.pairplot(df) #Plot graph between each every pair of column possible


# Finding correlation among the columns 
# correlation basically tell how much one column is important in term of particular column

# In[23]:


df.corr() 


# Plotting graph which plot count of distinct values in the column

# In[24]:


plt.figure(figsize=(12,5))
a = sns.countplot(x='workclass',data=df)
plt.show()        


# In[25]:


plt.figure(figsize=(20,5))
sns.countplot(x='education',orient='h',data=df)
plt.show()


# In[26]:


plt.figure(figsize=(7,5))
ax = sns.countplot(x="income", data=df)
plt.show()


# box plot 
# It has following component : minimum, first quartile (Q1), median, third quartile (Q3), and “maximum”).]
# median (Q2/50th Percentile): the middle value of the dataset.
# first quartile (Q1/25th Percentile): the middle number between the smallest number (not the “minimum”) and the median of the dataset.
# third quartile (Q3/75th Percentile): the middle value between the median and the highest value (not the “maximum”) of the dataset.
# interquartile range (IQR): 25th to the 75th percentile.
# whiskers (shown in blue)
# outliers (shown as green circles)
# “maximum”: Q3 + 1.5*IQR
# “minimum”: Q1 -1.5*IQR

# In[27]:


fig = plt.figure(figsize=(5,5))
sns.boxplot(x='income',y='age',data=df).set_title('Box plot of INCOME and AGE')
plt.show


# In[28]:


plt.figure(figsize=(5,5))
sns.boxplot(x="income", y="capital-gain", data=df)
plt.show() #this show how values are concentrated near 0 to 20000


# In[29]:


plt.figure(figsize=(25,5))
sns.catplot(y="race", hue="income", kind="count",col="gender", data=df);


# In[30]:


sns.countplot(y="occupation", hue="income",data=df)


# In[31]:


sns.heatmap(df.corr()) #It basically Plot correation on graph we can infer range from scale at righmost side


# In[32]:


df.head(3)


# In[33]:


dff=df
df["workclass"].unique()


# Changing Categorical values Numerical values

# In[34]:


from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
le = preprocessing.LabelEncoder()


# Label Encoder: Encode target labels with value between 0 and n_classes-1.

# In[35]:


le = LabelEncoder()
df["marital-status"] = le.fit_transform(df["marital-status"])


# In[36]:


df.head(3)


# In[37]:


le = LabelEncoder()
df["gender"] = le.fit_transform(df["gender"])


# In[38]:


df["gender"].unique()


# In[39]:


df["workclass"].unique()


# In[40]:


df["workclass"] = le.fit_transform(df["workclass"])


# In[41]:


df["workclass"].unique()


# In[42]:


df["education"]=le.fit_transform(df["education"])


# In[43]:


df["relationship"]=le.fit_transform(df["relationship"])


# In[44]:


df["race"]=le.fit_transform(df["race"])


# In[45]:


df.head(10)


# In[46]:


df['occupation'].fillna(df['occupation'][0],inplace = True)


# In[47]:


df.head(10)


# In[48]:


df["occupation"]=le.fit_transform(df["occupation"])


# In[49]:


df["occupation"].unique()


# In[50]:


df["native-country"].unique()


# In[51]:


df["native-country"] = le.fit_transform(df["native-country"])


# In[52]:


df["native-country"].unique()


# In[53]:


df["income"] = le.fit_transform(df["income"])
df["income"].unique()


# In[54]:


df.head(10)


# In[55]:


df.corr()


# In[56]:


import sklearn
from sklearn.feature_selection import SelectKBest

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[57]:


predictors = ['marital-status', 'educational-num', 'relationship', 'age']  '''first taking only 4 colum having highest 
                                                                            correation to check the output'''
X = df[predictors] 
y = df["income"] 


# Pipelining :  Machine learning (ML) pipelines consist of several steps to train a model
#     The pipeline object is in the form of (key, value) pairs. 
#     Key is a string that has the name for a particular step and value is the name of the function or actual method

# In[58]:


algorithms = [ 
    #linear kernel
    [Pipeline([('scaler',StandardScaler()), 
               ('svc',LinearSVC(random_state=1))]), predictors],
    #rbf kernel
    [Pipeline([('scaler',StandardScaler()),
               ('svc',SVC(kernel="rbf", random_state=1))]), predictors],
    #polynomial kernel
    [Pipeline([('scaler',StandardScaler()),
               ('svc',SVC(kernel='poly', random_state=1))]), predictors],
    #sigmoidf kernel
    [Pipeline([('scaler',StandardScaler()),
               ('svc',SVC(kernel='sigmoid', random_state=1))]), predictors]
]


# In[59]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[60]:


'''Accuracy (ACC) and area-under-curve (AUC) are used as an evaluation of performance'''
alg_acc = {}
alg_auc = {}
for alg, predictors in algorithms:
    alg_acc[alg] = 0
    alg_auc[alg] = 0
i=0
for alg, predictors in algorithms:
    alg.fit(X_train, y_train)
    y_pred = alg.predict(X_test)
    acc_score = accuracy_score(y_test, y_pred) 
    print(acc_score)
    alg_acc[alg] += acc_score
    auc_score = roc_auc_score(y_test, y_pred) 
    print(auc_score)
    alg_auc[alg] += auc_score
for alg, predictors in algorithms:
    alg_acc[alg] /= 1
    alg_auc[alg] /= 1
    print ("## %s ACC=%f" % (alg, alg_acc[alg]))
    print ("## %s AUC=%f" % (alg, alg_auc[alg]))


# In[61]:


predictors1 = ['age','workclass','education','educational-num',
              'marital-status', 'occupation','relationship','race','gender',
              'capital-gain','capital-loss','hours-per-week', 'native-country']


# In[62]:


X = df[predictors1] 
y = df["income"] 


# In[63]:


algorithms = [ 
    #linear kernel
    [Pipeline([('scaler',StandardScaler()), 
               ('svc',LinearSVC(random_state=1))]), predictors1],
    #rbf kernel
    [Pipeline([('scaler',StandardScaler()),
               ('svc',SVC(kernel="rbf", random_state=1))]), predictors1],
    #polynomial kernel
    [Pipeline([('scaler',StandardScaler()),
               ('svc',SVC(kernel='poly', random_state=1))]), predictors1],
    #sigmoidf kernel
    [Pipeline([('scaler',StandardScaler()),
               ('svc',SVC(kernel='sigmoid', random_state=1))]), predictors1]
]


# In[64]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[70]:


'''taking all columns as input'''
'''Accuracy (ACC) and area-under-curve (AUC) are used as an evaluation of performance'''
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    alg_acc = {}
    alg_auc = {}
    for alg, predictors1 in algorithms:
        alg_acc[alg] = 0
        alg_auc[alg] = 0
    i=0
    for alg, predictors1 in algorithms:
        alg.fit(X_train, y_train)
        y_pred = alg.predict(X_test)
        acc_score = accuracy_score(y_test, y_pred) 
        print(acc_score)
        alg_acc[alg] += acc_score
        auc_score = roc_auc_score(y_test, y_pred) 
        print(auc_score)
        alg_auc[alg] += auc_score
    for alg, predictors in algorithms:
        alg_acc[alg] /= 1
        alg_auc[alg] /= 1
        print ("## %s ACC=%f" % (alg, alg_acc[alg]))
        print ("## %s AUC=%f" % (alg, alg_auc[alg]))


# In[ ]:




