#!/usr/bin/env python
# coding: utf-8

# In[1]:


input_data=[4,350,120,92,4,233,2,1,0,3]


# In[21]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')


# In[13]:


data = pd.read_csv("FastagFraud dataset after EDA.csv")
data


# In[14]:


# Sepratating & assigning features and target columns to x & y respectively

y = data['Fraud_indicator']
x = data.drop('Fraud_indicator',axis=1)


# In[15]:


x


# In[16]:


y


# In[17]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3,random_state=42)


# In[18]:


logreg = LogisticRegression()


# In[22]:


logreg.fit(x_train, y_train)


# In[28]:


y_pred_test = logreg.score(x_train, y_train) # training acc
test = logreg.score(x_test, y_test) # testing acc
print(f"Traning Result -: {train}")
print(f"Test Result -: {test}")


# In[25]:


print(x_test)


# In[26]:


print('Actual Values of Y Test are: \n',np.array(y_test))


# In[29]:


print('Predicted Values of Y Test are: \n', y_pred_test)


# In[30]:


input_data = [340,350,120,77,4,1981,2,1,0,3]

# changing the input_data list into numpy array
input_data_as_numpy_array = np.array(input_data)

# reshape the array into 1 row and all columns-type,  as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

print(input_data_reshaped)


# In[32]:


input_data = [340,350,120,77,4,1981,2,1,0,3]

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array into 1 row and all columns-type,  as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = logreg.predict(input_data_reshaped)
print(f"Fastag Fraud Detection for given data is : {prediction}")
if(prediction[0]==0):
    print('No Fastag Fraud')
else:
    print('Fastag Fraud')


# In[33]:


import pickle


# In[34]:


filename = 'trained_model.sav'
pickle.dump(logreg, open(filename, 'wb'))


# In[35]:


loaded_model = pickle.load(open('trained_model.sav', 'rb'))


# In[38]:


input_data = [340,350,120,77,4,1981,2,1,0,3]

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array into 1 row and all columns-type,  as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(f"Fastag Fraud Detection for given data is : {prediction}")
if(prediction[0]==0):
    print('No Fastag Fraud')
else:
    print('Fastag Fraud')


# In[ ]:




