#!/usr/bin/env python
# coding: utf-8

# In[134]:


import pandas as pd
import numpy as np


# In[135]:


df = pd.read_csv("https://github.com/YBIFoundation/Dataset/raw/main/Big%20Sales%20Data.csv")


# In[136]:


df.head()


# In[137]:


df.info()


# In[138]:


df.columns


# In[139]:


df.describe()


# In[140]:


df['Item_Weight'].fillna(df.groupby(['Item_Type'])['Item_Weight'].transform('mean'),inplace = True)


# In[141]:


df.info()


# In[142]:


df.describe()


# In[143]:


import seaborn as sns
sns.pairplot(df)


# # Get categories and count of categories

# In[144]:


df[['Item_Identifier']].value_counts()


# In[145]:


df[['Item_Fat_Content']].value_counts()


# In[146]:


df.replace({'Item_Fat_Content':{'LF':'Low Fat','reg':'Regular','low fat':'Low Fat'}},inplace=True)


# In[147]:


df[['Item_Fat_Content']].value_counts()


# In[148]:


df.replace({'Item_Fat_Content': {'Low Fat':0,'Regular': 1}},inplace=True)


# In[149]:


df[['Item_Type']].value_counts()


# In[154]:


df.replace({'Item_Type':{'Fruits and Vegetables': 0, 'Snack Foods' :0,'Household'   :1,'Frozen Foods' : 0,'Dairy':0,
                         'Baking Goods' :0,'Canned': 0,'Health and Hygiene' :1,'Meat' :0,'Soft Drinks':0,'Breads':0,              
                         'Hard Drinks':0,'Others':2,'Starchy Foods':0,'Breakfast' :0,'Seafood'  :0 }},inplace=True)


# In[155]:


df[['Item_Type']].value_counts()


# In[156]:


df[['Outlet_Identifier']].value_counts()


# In[157]:


df.replace({'Outlet_Identifier':{'OUT027':0,'OUT013':1,'OUT049':2,'OUT046':3,'OUT035':4,'OUT045':5,'OUT018':6,'OUT017':7,'OUT010':8
                                 ,'OUT019':9}},inplace=True)


# In[158]:


df[['Outlet_Identifier']].value_counts()


# In[159]:


df[['Outlet_Size']].value_counts()


# In[160]:


df.replace({'Outlet_Size':{'Small':0,'Medium':1,'High':2}},inplace=True)


# In[161]:


df[['Outlet_Size']].value_counts()


# In[162]:


df[['Outlet_Location_Type']].value_counts()


# In[163]:


df.replace({'Outlet_Location_Type':{'Tier 1':0,'Tier 2':1,'Tier 3':2}},inplace=True)


# In[164]:


df[['Outlet_Location_Type']].value_counts()


# In[165]:


df[['Outlet_Type']].value_counts()


# In[166]:


df.replace({'Outlet_Type':{'Grocery Store':0,'Supermarket Type1':1,'Supermarket Type2':2,'Supermarket Type3':3}},inplace=True)


# In[167]:


df[['Outlet_Type']].value_counts()


# In[168]:


df.head()


# In[169]:




df.info()


# In[170]:


df.shape


# In[171]:


y = df['Item_Outlet_Sales']


# In[172]:


y.shape


# In[173]:


y


# In[174]:


X = df[['Item_Weight','Item_Fat_Content','Item_Visibility','Item_Type',
          'Item_MRP','Outlet_Identifier'
         ,'Outlet_Establishment_Year','Outlet_Size'
         ,'Outlet_Location_Type','Outlet_Type']]


# In[175]:


X = df.drop(['Item_Identifier','Item_Outlet_Sales'],axis = 1)


# In[176]:


X.shape


# In[177]:


X


# In[178]:


from sklearn.preprocessing import StandardScaler


# In[179]:


sc = StandardScaler()


# In[180]:


X_std = df[['Item_Weight','Item_Visibility','Item_MRP','Outlet_Establishment_Year']]


# In[181]:


X_std = sc.fit_transform(X_std)


# In[182]:


X_std


# In[183]:


X[['Item_Weight','Item_Visibility','Item_MRP','Outlet_Establishment_Year']] = pd.DataFrame(X_std,columns=[['Item_Weight','Item_Visibility','Item_MRP','Outlet_Establishment_Year']])


# In[184]:


X


# In[185]:


from sklearn.model_selection import train_test_split


# In[186]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1,random_state=2529)


# In[187]:


X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[188]:


from sklearn.ensemble import RandomForestRegressor


# In[189]:


rfr = RandomForestRegressor(random_state = 2529 )


# In[190]:


rfr.fit(X_train,y_train)


# In[191]:


y_pred = rfr.predict(X_test)


# In[192]:


y_pred.shape


# In[193]:


y_pred


# In[194]:


from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


# In[195]:


mean_squared_error(y_test,y_pred)


# In[196]:


mean_absolute_error(y_test,y_pred)


# In[197]:


r2_score(y_test,y_pred)


# In[198]:


import matplotlib.pyplot as plt
plt.scatter(y_test,y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Predicted Price")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




