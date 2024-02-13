#!/usr/bin/env python
# coding: utf-8

# # AIM: 
#    The aim of a Fastag fraud detection model using machine learning (ML) is to identify and prevent fraudulent activities related to the use of Fastags.and adapt to new patterns of fraudulent behavior, enhancing the accuracy of fraud detection over time.  

# These attributes capture key details about toll transactions, including vehicle information (type, dimensions, plate number), payment details (FastagID, transaction amount, amount paid), toll location and timing (TollBoothID, timestamp), lane usage (Lane_Type), vehicle speed, and a fraud indicator.
# 1. Transaction_ID: 
# A unique identifier assigned to each toll transaction, used for tracking and referencing purposes. (Like a receipt number for a toll payment
# 2. Timestamp:
# The exact date and time when the toll transaction occurred, crucial for chronological analysis and potential dispute resolution.
# 3. Vehicle_Type
# The type of vehicle involved in the transaction, such as car, truck, or motorcycle, often influencing toll rates and regulations.
# 4.FastagID
# A unique identifier linked to a prepaid toll payment account, enabling electronic toll collection without manual cash payments.
# 5. TollBoothID:
# A unique identifier assigned to each tollbooth, used for pinpointing locations and analyzing traffic patterns.
# 6.Lane_Type :
# The specific lane used for the transaction.
# 7.Vehicle_Dimensions
# The length, width, and height of the vehicle, potentially influencing toll calculations and safety measures.
# 8.Transaction_Amount: The total cost.
# 9.Amount_paid
# The actual amount paid during the transaction, reflecting any discounts, penalties, or outstanding balances.
# 10.Geographical_Location
# The precise latitude and longitude of the tollbooth, enabling mapping and spatial analysis.
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv(r"C:\Users\Admin\Desktop\FastagFraudDetection Data.csv")
df


#  # Vehicle_Plate_Number as  have unique value which do not having dependency

# In[2]:


df=df.drop(['Vehicle_Plate_Number'],axis=1)
df


# In[3]:


df.isnull().sum()


# In[4]:


df.info()


# In[5]:


df.describe()


# # for motorcycle fastag not required in table so,we delete that rows.

# In[6]:


print(df[df["Vehicle_Type"]=="Motorcycle"])


# In[7]:


list1=(df[df["Vehicle_Type"]=="Motorcycle"]).index.values.tolist()
list1


# In[8]:


df.drop(index=list1,inplace=True)


# In[9]:


df.info()


# I found ? in Amount_paid column, so i replace the ? with NAN because pandas and sklearn only handle NaN values

# In[10]:


df["Amount_paid"].value_counts()


# In[11]:


df["Amount_paid"].replace("?",np.nan,inplace=True)


# In[12]:


df["Amount_paid"]=df["Amount_paid"].astype(float) # To convert into Float data type


# In[13]:


NL_mean=df["Amount_paid"].mean()
df["Amount_paid"]=df["Amount_paid"].fillna(NL_mean)  # to replace null value to mean


# In[14]:


df.isnull().sum()


# # Data Visualization

# In[15]:


Fraud_indicator = df['Fraud_indicator'].value_counts()
plt.style.use('default')
plt.figure(figsize=(5, 5))
sns.barplot(x=Fraud_indicator.index, y=Fraud_indicator.values, palette='bright')
plt.title(' Status Fraud_indicator', fontdict={'fontname': 'Georgia', 'fontsize': 10, 'fontweight': 'bold'})
plt.xlabel(' Fraud_indicator', fontdict={'fontname': 'Georgia', 'fontsize': 8})
plt.ylabel('count of Fraud_indicator', fontdict={'fontname': 'Georgia', 'fontsize': 8})
plt.tick_params(labelsize=8)
plt.show()

  #Time-based Analysis:
# In[16]:


# Convert 'Timestamp' to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])


# In[17]:


# Explore daily transaction trends
daily_trends = df.resample('D', on='Timestamp')['Transaction_ID'].count()


# In[18]:


# Visualize daily transaction trends
plt.figure(figsize=(6,4))
daily_trends.plot(kind='line')
plt.title('Daily Transaction Trends')
plt.xlabel('Date')
plt.ylabel('Number of Transactions')
plt.show()


# In[19]:


plt.figure(figsize=(6,4))    # Vehicle type Distribution over Fraud indicator
sns.countplot(x='Vehicle_Type', hue='Fraud_indicator', data=df,color = "pink")
plt.title('Vehicle Type Distribution')
plt.show()


#   Fraud indicator count on Lane Type indicate equally fraud detect on Express and Regular way

# In[20]:


plt.figure(figsize=(6,4))   # Lane Type Distribution
sns.countplot(x='Lane_Type', hue='Fraud_indicator', data=df)
plt.title('Lane Type Distribution')
plt.show()


# In[21]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='Fraud_indicator', y='Transaction_Amount', data=df)
plt.title('Transaction Amount Distribution')
plt.show()


# # Geographical Location Visualization

# In[22]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Geographical_Location', y='Transaction_Amount', hue='Fraud_indicator', data=df)
plt.title('Geographical Location Visualization')
plt.show()


# In[23]:


#Histogram
df.hist(figsize=(10, 8), bins=20)
plt.show()


# In[24]:


df_num=df.select_dtypes(["int","float"])
df_num


# In[25]:


df_cat=df.select_dtypes(["object"])
df_cat


# In[26]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in df_cat:
    df_cat[i]=le.fit_transform(df_cat[i])
df_cat


# # Feature Scaling

# In[27]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_scaled=sc.fit_transform(df_num)
x_scaled


# In[28]:


df_scaled=pd.DataFrame(data=x_scaled,columns=df_num.columns)
df_scaled


# In[29]:


df_new=pd.concat([df_num,df_cat],axis=1)
df_new


# In[30]:


x=df_new.drop(["Fraud_indicator"],axis=1)
x


# In[31]:


y=df_new["Fraud_indicator"]
y


# In[32]:


df['Fraud_indicator'].value_counts()


# # Since the Data is imbalanced , need to do sampling (either Oversampling or Undersampling)¶

# # UnderSampling: Deleting samples from majority class

# In[33]:


from imblearn.under_sampling import RandomUnderSampler


# In[35]:


ros=RandomUnderSampler(random_state=1)
x_ros,y_ros=ros.fit_resample(x,y)


# In[36]:


x=x_ros
y=y_ros


# In[37]:


y


# # Check  Data got balanced by Undersampling:

# In[40]:


Fraud_indicator = y.value_counts()
plt.style.use('default')
plt.figure(figsize=(5, 5))
sns.barplot(x=Fraud_indicator.index, y=Fraud_indicator.values, palette='bright')
plt.title(' Status Fraud_indicator', fontdict={'fontname': 'Georgia', 'fontsize': 10, 'fontweight': 'bold'})
plt.xlabel(' Fraud_indicator', fontdict={'fontname': 'Georgia', 'fontsize': 8})
plt.ylabel('count of Fraud_indicator', fontdict={'fontname': 'Georgia', 'fontsize': 8})
plt.tick_params(labelsize=8)
plt.show()


# # Apply Different ML Models to check Train Data and Test Data Accuracy :

# In[41]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.svm import SVC


# In[42]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)


# In[43]:


logreg = LogisticRegression()
knn = KNeighborsClassifier()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier(random_state=42)
svm = SVC()


# In[44]:


def mymodel(model):
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    print(classification_report(y_test,y_pred))
    return model


# In[45]:


mymodel(logreg)


# In[46]:


train = logreg.score(x_train, y_train) # training acc
test = logreg.score(x_test, y_test) # testing acc
print(f"Traning Result -: {train}")
print(f"Test Result -: {test}")


# In[47]:


mymodel(dt)


# In[48]:


train = dt.score(x_train, y_train) # training acc
test = dt.score(x_test, y_test) # testing acc
print(f"Traning Result -: {train}")
print(f"Test Result -: {test}")


# In[49]:


mymodel(rf)


# In[50]:


train = rf.score(x_train, y_train) # training acc
test = rf.score(x_test, y_test) # testing acc
print(f"Traning Result -: {train}")
print(f"Test Result -: {test}")


# In[52]:


mymodel(svm)


# In[53]:


train = svm.score(x_train, y_train)   # training acc
test = svm.score(x_test, y_test)      # testing acc
print(f"Traning Result -: {train}")
print(f"Test Result -: {test}")


# SVM Hypertunnig by GRIDSEARCHCV for better Accuracy

# In[54]:


from sklearn.model_selection import GridSearchCV


# In[55]:


parameter ={'C':[1,10],'gamma':[1,10],'kernel':['rbf','linear']}


# In[56]:


GS=GridSearchCV(svm,parameter,cv=4,verbose=3)  # Observe the result displayed with verbosity 2
# We see GridSearch has run 4 fold CV for each combination of "c" and "Gamma" passed in the dictionary
# Thus giving score(Accuracy) for each iterations
GS


# In[57]:


GS.fit(x_train,y_train) 


# In[58]:


#To know the best value of parameters in the GridSearchCV, we can run below command:
GS.best_params_  


# In[59]:


svm = SVC(kernel='linear',C=1,gamma=1)


# In[60]:


mymodel(svm)


# # Thus, Hypertuning the SVM using GRIDSEARCHCV, helped us get best parameters to find the better accuracy, easily

# In[ ]:





# In[ ]:




