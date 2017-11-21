import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report
df = pd.read_csv("Classified Data")


print (df.head())

#preparing the data :

# For KNN always perform stardaization of the data so that no features dominate

Scaler = StandardScaler()
Scaler.fit(df.drop('TARGET CLASS',axis=1))  # Performing scaling on all feature columns only

scaled_features = Scaler.transform(df.drop('TARGET CLASS',axis= 1))

print(scaled_features)
# data is now prepared



X = scaled_features
y = df['TARGET CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 101)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

predictions = knn.predict(X_test)



#accuarcy metrics
print ("Confusion matrix : ")
cm = (confusion_matrix(y_test,predictions))
print (cm)
print ("Classification report")
print(classification_report(y_test,predictions))


print("ACCURACY ")
# print(cm.sum())
accuracy = (cm[0][0]+cm[1][1] )/ cm.sum()
print(accuracy)


print("ERROR RATE : ")
error = (cm[0][1]+cm[1][0])/cm.sum()
print(error)


#Plotting error rates for different values of n

error_rate = []

for i in range (1,30):

    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append((np.mean(pred_i != y_test)))  # this is the average error rate when k = i



print ("done")

plt.figure(figsize=(10,6))
plt.plot(range(1,30),error_rate,color='black',linestyle='dashed',marker = 'x',markersize=7,markerfacecolor= "red")
plt.title('Error rate vs K value')

plt.show()

























