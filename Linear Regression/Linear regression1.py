#Demonstaration of linear regression in python
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression


df = pd.read_csv('USA_Housing.csv')

#The commands that follow are used for exploring the datset further
print(df.info())

print(df.describe()) #quick account of statistical information about the data

print(sns.pairplot(df)) #create histograms and corelation scatter plots

print(sns.distplot(df['Price']))  #distribution of price in our dataset , This is also the label we are trying to predict


print(df.columns)

#Performing linear regression

X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]                    #Features for dependency

Y = df['Price'] #Feature to predict


X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.33, random_state=42)

lm = LinearRegression()         #Creating a linear regression object

lm.fit(X_train,Y_train)         #training the model

#Performing predictions of Price
prediction = lm.predict(X_test)

print("Predicted House Prices ")
print(prediction)  #Predicted house prices

plt.scatter(prediction,Y_test)


#Histogram Plot of the residuals
#Normally distributed residuals suggests that the model was the right choice for the data
sns.distplot((Y_test-prediction))


#Regression Evaluation :Most common evaluation metrics
error1 = metrics.mean_absolute_error(Y_test,prediction)

print("Mean absolute error for linear regression ")
print(error1)

error2 = metrics.mean_squared_error(Y_test,prediction)

print("Mean squared error for linear regression ")
print(error2)



error3 = math.sqrt(metrics.mean_squared_error(Y_test,prediction)) #interpreted in terms of target units
print("Root mean squared error for linear regression ")
print(error3)

plt.show()
