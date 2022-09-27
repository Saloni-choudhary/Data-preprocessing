#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries
# 

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]


# In[4]:


print(x)


# In[5]:


print(y)


# In[6]:


print(dataset.describe)


# ## Missing Values

# In[78]:


dataset.isnull().sum()


# In[79]:


dataset.fillna(dataset.mean())


# ## Encoding Categorical Data

# In[80]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))


# In[81]:


print(x)


# ## Encoding the Dependent Variable

# In[82]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# In[83]:


print(y)


# ## Splitting the dataset into training set and testing set

# In[84]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=1)


# In[85]:


print(x_train)


# In[86]:


print(x_test)


# In[87]:


print(y_train)


# In[88]:


print(y_test)

