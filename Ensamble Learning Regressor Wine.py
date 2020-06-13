# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:11:23 2020

@author: Usuario
"""

import pandas as pd
wine=pd.get_dummies(wine,columns=["color"],drop_first=True)
#%%
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as MSE
#%%
X=wine.drop(["quality"],axis=1)
y=wine["quality"]
#%%
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)
lr=LinearRegression()
dt=DecisionTreeRegressor(random_state=1,min_samples_leaf=0.1)
#%%
regressors=[('LR',lr),('DT',dt)]
for rg_name,rg in regressors:
    rg.fit(X_train,y_train)
    y_pred=rg.predict(X_test)
    print(rg_name,MSE(y_test,y_pred))
    

#%%
vr=VotingRegressor(estimators=regressors)
vr.fit(X_train,y_train)
vr_pred=vr.predict(X_test) 
print(MSE(y_test,vr_pred))       