#!/usr/bin/env python
# coding: utf-8

# In[2]:





# In[ ]:


import pandas as pd

df = pd.read_csv('/Downloads/TRAIN.csv')
df_test = pd.read_csv('/Downloads/TEST.csv')


# In[ ]:




newdf = df[(df.warehouse_id == "ABC2")]

newdf['year'] = pd.DatetimeIndex(newdf['evsd']).year
newdf['month'] = pd.DatetimeIndex(newdf['evsd']).month
newdf['day'] = pd.DatetimeIndex(newdf['evsd']).day
newdf['dayofweek'] = pd.DatetimeIndex(newdf['evsd']).day_name()
newdf.dayofweek.replace(to_replace=['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'],value=[0,1,2,3,4,5,6], inplace=True)

newdf['dayofweek_o'] = pd.DatetimeIndex(newdf['order_date']).day_name()
newdf.dayofweek_o.replace(to_replace=['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'],value=[0,1,2,3,4,5,6], inplace=True)


newdf_test = df_test[(df_test.warehouse_id == "ABC2")]

newdf_test['year'] = pd.DatetimeIndex(newdf_test['evsd']).year
newdf_test['month'] = pd.DatetimeIndex(newdf_test['evsd']).month
newdf_test['day'] = pd.DatetimeIndex(newdf_test['evsd']).day
newdf_test['dayofweek'] = pd.DatetimeIndex(newdf_test['evsd']).day_name()
newdf_test.dayofweek.replace(to_replace=['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'],value=[0,1,2,3,4,5,6], inplace=True)
newdf_test['dayofweek_o'] = pd.DatetimeIndex(newdf_test['order_date']).day_name()
newdf_test.dayofweek_o.replace(to_replace=['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'],value=[0,1,2,3,4,5,6], inplace=True)

newdf.corr()['quantity_received']



# In[ ]:




from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df1 = newdf[['d_id','dayofweek','month','visibility','quantity_ordered','quantity_submitted','quantity_received']]
df_np=df1.to_numpy()
df_np.shape
x_train,y_train=df_np[:,:6],df_np[:,-1]

df1_test=newdf_test[['d_id','dayofweek','month','visibility','quantity_ordered','quantity_submitted','quantity_received']]
df_test_np=df1_test.to_numpy()
x_test,y_test=df_test_np[:,:6],df_test_np[:,-1]

from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.1)
yhat = iso.fit_predict(x_train)
# select all rows that are not outliers
mask = yhat != -1
x_train, y_train = x_train[mask, :], y_train[mask]
from sklearn.metrics import accuracy_score
#Create a LinearRegression object
lr= LinearRegression()
#Fit X and y 
lr.fit(x_train, y_train)
print(lr.coef_)
p_values = f_regression(x_train,y_train)[1]
print(p_values.round(3))

ypred = lr.predict(x_test)

#Metrics to evaluate your model 
r2_score(y_test, ypred), mean_absolute_error(y_test, ypred), np.sqrt(mean_squared_error(y_test, ypred))

prediction_df=pd.DataFrame({'warehouse_id':newdf_test['warehouse_id'],
                            'distributor_id':newdf_test['distributor_id'],
'order_id':newdf_test['order_id'],
'order_date':newdf_test['order_date'],
'evsd':newdf_test['evsd'],
'po_source_ind':newdf_test['po_source_ind'],
                            'quantity_ordered'   :newdf_test['quantity_ordered'],
                            'quantity_submitted' : newdf_test['quantity_submitted'],
                            'visibility' : newdf_test['visibility'],
                            'quantity_received' : newdf_test['quantity_received'],
                            'LinearRegression_prediction' : ypred})
prediction_df
prediction_df.to_csv('/Users/padmanpr/Work/LinearRegression.csv')

