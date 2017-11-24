import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,confusion_matrix


df = pd.read_csv("kyphosis.csv")
#Exploratory data nalysis

df.info()


subsetdf1 = df[df['Kyphosis']=='absent']
subsetdf2 = df[df['Kyphosis']=='absent']

#Number of people from the dataset where Kyphosis was present

distinct_count = df['Kyphosis'].value_counts()


print(distinct_count)

#plotting a bar graph
# plt.figure(figsize=(10,6))


# plt.bar(range(1,3),distinct_count,linewidth=20, tick_label=['Absent','Present'],color='burlywood')
# plt.title('Distinct count ...')
# plt.show()

#Looking for missing data
print (df.isnull())
#The dataset dosen't contain any null values


# Performing scaling over all the numerical columns:
scaler = StandardScaler()
dropper = scaler.fit(df.drop('Kyphosis',axis=1))
scaled_features = pd.DataFrame(scaler.transform(df.drop('Kyphosis',axis=1)))

print(type(scaled_features))

print("scaled Features ")
print(scaled_features)


df_kyphosis = df['Kyphosis']

frames = [df_kyphosis,scaled_features]
df_new = pd.concat(frames ,axis=1)
print(df_new)  #dataframe containing original kyphosis labels and scaled values




#Checking distribution by age of people in which kyphosis is absent


# sns.distplot(df['Age'])#distribution on full dataset
# plt.show()


df_present = df[(df['Kyphosis']=='present')] #Dataframe containgn only records that have kyphosis = present

print("Data Frame with only present ")
print(df_present)

#Checking distribution by age of people in which kyphosis is present

sns.distplot(df_present['Age'],bins= 10,color='red')#distribution on full dataset
plt.title("Checking distribution by age of people in which kyphosis is present")
plt.show()

df_absent = df[df['Kyphosis']=='absent']
print("Data Frame with only absebnt" )
print(df_absent)



sns.distplot(df_absent['Age'],bins= 10,color='grey')#distribution on full dataset
plt.title("Checking distribution by age of people in which kyphosis is absent")
plt.show()


sns.pairplot(df, hue='Kyphosis')
plt.show()



X = df.drop('Kyphosis',axis =1)
y = df['Kyphosis']



X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3)


dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


predictions = dtree.predict(X_test)


#accuarcy metrics
print("DECISION TREE")
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



rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)


rfc_pred = rfc.predict(X_test)


#accuarcy metrics
print("RANDOM FOREST")
print ("Confusion matrix : ")
cm2 = (confusion_matrix(y_test,rfc_pred))
print (cm2)
print ("Classification report")
print(classification_report(y_test,rfc_pred))





print("ACCURACY ")
# print(cm.sum())
accuracy2 = (cm2[0][0]+cm2[1][1] )/ cm2.sum()
print(accuracy2)


print("ERROR RATE : ")
error2 = (cm2[0][1]+cm2[1][0])/cm2.sum()
print(error2)




