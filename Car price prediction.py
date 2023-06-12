#!/usr/bin/env python
# coding: utf-8

# In[1]:


#CAR PRICE PREDICTION WITH ML
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics


# In[2]:


url='https://raw.githubusercontent.com/amankharwal/Website-data/master/CarPrice.csv'
df = pd.read_csv(url,index_col=0)
df.head()


# In[23]:


df.shape


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[7]:


plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),annot=True)
plt.title("Correlation between the columns")


# In[8]:


x = df.drop(['price','CarName','fueltype','aspiration','doornumber','carbody','drivewheel','enginelocation','enginetype','cylindernumber','fuelsystem'], axis=1)
y = df['price'].values
print(f"x shape : {x.shape}  y shape : {y.shape}")
    


# In[9]:


#splitting the data
from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test = tts(x,y,test_size=0.25,random_state=42)
print("X-train : ",x_train.shape)
print("X-test : ",x_test.shape)
print("y-train : ",y_train.shape)
print("y-test : ",y_test.shape)


# In[10]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)


# In[11]:


pred_test = model.predict(x_test)


# In[12]:


#MSE is mean squared error , MAE is mean absolute error
print("MSE : ",(metrics.mean_squared_error(pred_test,y_test)))
print("MAE : ",(metrics.mean_absolute_error(pred_test,y_test)))
print("R2 Score : ",(metrics.r2_score(pred_test,y_test)))
    


# In[13]:


pred2 = model.predict([[3,88.6,168.8,64.1,48.8,2548,130,3.47,2.68,9,111,5000,21,27]])
pred2.round(1)


# In[14]:


plt.scatter(y_test, pred_test)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()


# In[15]:


lass_model=Lasso(alpha=1.0)


# In[16]:


lass_model.fit(x_train,y_train)


# In[17]:


pred_test1=lass_model.predict(x_test)


# In[18]:


#MSE is mean squared error , MAE is 
print("MSE : ",(metrics.mean_squared_error(pred_test1,y_test)))
print("MAE : ",(metrics.mean_absolute_error(pred_test1,y_test)))
print("R2 Score : ",(metrics.r2_score(pred_test1,y_test)))


# In[19]:


plt.scatter(y_test, pred_test1)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()


# In[20]:


pred = model.predict([[3,88.6,168.8,64.1,48.8,2548,130,3.47,2.68,9,111,5000,21,27]])
pred.round(1)


# In[ ]:





# In[ ]:





# In[ ]:




