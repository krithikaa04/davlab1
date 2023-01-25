# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 12:59:26 2023

@author: Krithika
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
#from ml_metrics import rmse
from math import sqrt

train= pd.read_csv("train.csv")
test =pd.read_csv("test.csv")

train.info()
train.describe().T

#selecting the features
corr=train.corr()
corr
corr["SalePrice"]

sns.jointplot(x = "OverallQual", y = "SalePrice", data = train, kind = "reg");
sns.jointplot(x = "GrLivArea", y = "SalePrice", data = train, kind = "reg");
sns.jointplot(x = "GarageCars", y = "SalePrice", data = train, kind = "reg");
sns.jointplot(x = "TotalBsmtSF", y = "SalePrice", data = train, kind = "reg");
sns.jointplot(x = "GarageArea", y = "SalePrice", data = train, kind = "reg");

#Modelling
#linear regression with 2 features
lm  = LinearRegression()
X = pd.DataFrame(np.c_[train['GrLivArea'], train['OverallQual']], columns = ['GrLivArea','OverallQual'])
y = train[["SalePrice"]]
x_train, x_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.3,random_state=5)
model = lm.fit(x_train, Y_train)
y_pred=model.predict(x_test)
y_pred[0:5]

# The mean absolute error
print("MAE = %5.3f" % mean_absolute_error(Y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print("R^2 = %0.5f" % r2_score(Y_test, y_pred))
# The mean squared error
print("MSE = %5.3f" % mean_squared_error(Y_test, y_pred))
# Root-Mean-Squared-Error (RMSE)
rms1 = sqrt(mean_squared_error(Y_test,y_pred))
print("RMSE = %5.3f" %rms1)
y_pred=model.predict(x_test)
mse = mean_squared_error(Y_test, model.predict(x_test))

fig, ax = plt.subplots()
ax.scatter(Y_test, y_pred, edgecolors = (0, 0, 0))
ax.text(y_pred.max()-4.5, Y_test.max()-0.1, r"$R^2$ = %.2f, MAE = %.2f" % (r2_score(Y_test, y_pred), mean_absolute_error(Y_test, y_pred)))
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], "k--",lw = 4)
ax.set_xlabel("Measured")
ax.set_ylabel("Predicted")
plt.show()

#linear regression with 5 features
df=train.loc[:,['GrLivArea','OverallQual',"GarageArea","TotalBsmtSF","GarageCars"]]
y = train[["SalePrice"]]
df
x_train, x_test, Y_train, Y_test = train_test_split(df, y, test_size = 0.3,random_state=5)
model = lm.fit(x_train, Y_train)
y_pred=model.predict(x_test)
mse = mean_squared_error(Y_test, model.predict(x_test))

# The mean absolute error
print("MAE = %5.3f" % mean_absolute_error(Y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print("R^2 = %0.5f" % r2_score(Y_test, y_pred))
# The mean squared error
print("MSE = %5.3f" % mean_squared_error(Y_test, y_pred))
# Root-Mean-Squared-Error (RMSE)
rms = sqrt(mean_squared_error(Y_test,y_pred))
print("RMSE = %5.3f" %rms)

fig, ax = plt.subplots()
ax.scatter(Y_test, y_pred, edgecolors = (0, 0, 0))
ax.text(y_pred.max()-4.5, Y_test.max()-0.1, r"$R^2$ = %.2f, MAE = %.2f" % (
r2_score(Y_test, y_pred), mean_absolute_error(Y_test, y_pred)))
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], "k--",
lw = 4)
ax.set_xlabel("Measured")
ax.set_ylabel("Predicted")
plt.show()

#data exploring
final=test.loc[:,['GrLivArea','OverallQual',"GarageArea","TotalBsmtSF","GarageCars"]]
y_pred_test= model.predict(final)
submission  = pd.DataFrame()
submission['Id']=test['Id']
submission["SalePrice"]=y_pred_test
submission.info()
submission.to_csv('submission.csv', index=False) 
submission.head()