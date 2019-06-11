#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd 
import numpy as np 


# In[42]:


dataset = pd.read_csv('Dataset.csv')


# In[43]:


dataset.head(20)


# In[44]:


dataset.shape 


# In[45]:


dataset.describe() 


# In[46]:


X = dataset.iloc[:, 1:3].values
y = dataset.iloc[:, 0].values


# In[47]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 


# In[48]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test) 


# In[ ]:





# # #RANDOM FOREST RREGRESSION -->>

# In[49]:


from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor


# In[50]:


regressor = RandomForestRegressor(n_estimators=20, random_state=0) 


# In[51]:


regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


# In[52]:


from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[53]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df


# In[54]:


from sklearn.metrics import confusion_matrix, accuracy_score

print("Training Accuracy = ", regressor.score(X_train, y_train))
print("Test Accuracy = ", regressor.score(X_test, y_test))


# In[55]:


# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = regressor.estimators_[5]
# Pull out one tree from the forest
tree = regressor.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')


# In[30]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  
print(accuracy_score(y_test, y_pred)) 


# # LINEAR REGRESION -->>

# In[161]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train)


# In[162]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[163]:


dataset.plot(x='Humidity', y='MoisturePercentage', style='o')  
plt.title('Humidity vs MoisturePercentage')  
plt.xlabel('Humidity')  
plt.ylabel('MoisturePercentage')  
plt.show()  


# In[164]:


dataset.plot(x='Heatcelcius', y='MoisturePercentage', style='o')  
plt.title('Heat-Air temperature  vs  MoisturePercentage')  
plt.xlabel('Heat in Celcius')  
plt.ylabel('MoisturePercentage')  
plt.show()


# In[165]:


print(regressor.intercept_) 


# In[166]:


print(regressor.coef_) 


# In[167]:


y_pred = regressor.predict(X_test)


# In[168]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df


# In[169]:


print("Training Accuracy = ", regressor.score(X_train, y_train))
print("Test Accuracy = ", regressor.score(X_test, y_test))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




