#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

df= pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/edx/project/drinks.csv')
df.head()


# In[2]:


df.tail()


# PREGUNTA 1

# In[3]:


df.dtypes


# In[4]:


df.describe()


# PREGUNTA 2

# In[5]:


df.groupby("continent").beer_servings.sum()


# PREGUNTA 3

# In[6]:


df.groupby("continent").beer_servings.describe()


# PREGUNTA 4

# In[7]:


sns.boxplot (x = "continent" , y = "beer_servings" , data = df)


# PREGUNTA 5

# In[8]:


df[["beer_servings" , "wine_servings"]].corr()


# In[9]:


sns.regplot( x = "wine_servings" , y = "beer_servings" , data = df)


# PREGUNTA 6

# In[10]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm
X = df[['wine_servings']]
Y = df['total_litres_of_pure_alcohol']
lm.fit ( X , Y )
Yhat=lm.predict(X)
Yhat [0 : 5]


# In[11]:


print (lm.intercept_)


# In[12]:


print (lm.coef_)


# In[13]:


print('The R-square is: ', lm.score( X , Y ))


# PREGUNTA 7

# In[16]:


Y_data = df["total_litres_of_pure_alcohol"]
X_data = df.drop("total_litres_of_pure_alcohol" , axis = 1)

from sklearn.model_selection import train_test_split


X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=0)


print("number of test samples :", X_test.shape[0])
print("number of training samples:",X_train.shape[0])


# In[ ]:





# In[ ]:





# In[ ]:




