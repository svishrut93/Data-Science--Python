import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt


data1 = pd.read_csv('titanic_train.csv')#import the training data as a dataframe

#Exploratory data analysis
print(data1.info())


#The "survived" column tells us if a passenger survived or not
# encoding used : 0 = not survive / 1= survive
print("Number of survivers(1) and number of non-survivors(0) ")
distinct_count_of_survived = data1['Survived'].value_counts()
print(distinct_count_of_survived) # tells us how many people survived and how many dint///
#the object type of distinct_count_of_survived is a pandas serioes
#print(type(data1['Survived'].value_counts()))
count_sunrvived = distinct_count_of_survived[0]
count_not_survived = distinct_count_of_survived[1]

xaxis = [1,2]
yaxis = [549,342]
labels = ['survived','not survived']
print(count_not_survived)

#Plotting these on a graph
#
# plt.bar(xaxis,yaxis)
#
# plt.ylabel ('count')



#idea on the age of people a=on the titANIC


#here i create a distribution plot of the ages of people on the titanic :
sns.distplot(data1['Age'].dropna(),bins = 30 )
#
#
sns.distplot(data1['Fare'].dropna(),bins=30)

#comparing different attributes : age and far

sns.boxplot(x = 'Pclass',y = 'Age', data=data1)
#

#checking the graph, we find that passengers in the first class tend to be older than passengers in class 2 and 3

#cleaning the data


#Filling the missing data with the mean : Imputation



#
def impute_age(cols):
    Age= cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1 :
            return 37 # Average age of a passenger with first class ticket : Got from the previous box plot
        if Pclass == 2 :
            return 29 #Avaerage age of a passenger with 2nd class ticket : Got from the previous box plot
        else:
            return 24 #Average age of a passenger with 3rd class ticket : Got from previous box plot
    else:
        return Age



data1['Age']=data1[['Age','Pclass']].apply(impute_age,axis=1)   #APPLYING FUNCTION : REMOVING NaN values wityh means

print(data1['Age'])
#



data1.drop('Cabin',axis=1,inplace=True)   #Dropping value : column
print(data1.columns.values)

#dropping anymore missing values, since there are so few left after imputation
data1.dropna(inplace=True)

#dealing with categorical features
#using get dummies to encode the categorical columns : Sex and Embarked


sex = pd.get_dummies(data1['Sex'],drop_first=True)
print(sex)

embark = pd.get_dummies(data1['Embarked'],drop_first=True)
print(embark)

#now concatenate all the 3 dataframes( 1 main , 2 (sex and embark ))
data1= pd.concat([data1,sex,embark],axis=1)

print(data1)
#at this point dummy variables have been inserted
#now that categorical data has been encoded with dummy variables,
# we remove the columns that had the categorical data in the first place

data1.drop(['Sex','Embarked','Name','Ticket'],inplace=True,axis=1)
print(data1)
print("---")
print(data1.columns.values)
print("At this point the dataframe has been cleaned with only "
      "numerical values, and is fit to be fed into an ML algorithm")
plt.show()







