#Refer data cleaning file first 
#Data frame used is output of data cleaning.py file 


import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

data1 = pd.read_csv('titanic_train.csv')#import the training data as a dataframe

#beginning Machine Learning


#Label we want to predict usinng logistic regression is whether a passenger survived or nor

X = data1.drop('Survived',axis=1)
Y = data1['Survived']

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.33, random_state=101)


#create an object for logistic regression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
print(classification_report(y_test,predictions))
CM = confusion_matrix(y_test,predictions)



print(type(CM))
print(CM)

accuracy = (CM[0][0] + CM[1][1] )/ CM.sum()   #Accuracy = TP+ TN / Total
print(accuracy)

plt.scatter(y_test,predictions)
plt.show()
