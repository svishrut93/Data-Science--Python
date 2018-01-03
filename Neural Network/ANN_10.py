#Artificail Neural Network
#Installing Theano
# Runs on both CPU ans GPU, also built on top of numpy
#GPU is usually a much better choice for Neural networks as forward and backpropagation involve parallel computations

#Installing Tensorflow

#Installing Keras
#In a way Keras wraps the two libraries Theano and Tensor Flow togetgher
#Deep neural networks in a very few lines of code


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')


X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

print(X)

print("---------")
print(type(X))
print(y)


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])


onehotencoder = OneHotEncoder(categorical_features = [1])  #creating dummy variables
X = onehotencoder.fit_transform(X).toarray()


#Getting rid of the dummy variable trap with 3 categorical variables for country.
# Removing the Oth element from all instances of nd numpy array

X = X [:,1:]




# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

#Train on 8000 observations and test on 2000 observations

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


Xdataframe = pd.DataFrame(X)

print(Xdataframe)

#Importing the keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
#Uses tenbsor flow backend

#initializing our Artificial neural network

classifier = Sequential()

#Adding layers to our ANN- Input layer and he first hidden layer
classifier.add(Dense(output_dim = 6 , init = 'uniform', activation='relu', input_dim=11))

#Adding more hidden layers
classifier.add(Dense(output_dim = 6, init = 'uniform', activation='relu'))

#Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation='sigmoid'))

#Compiling the ANN
classifier.compile(optimizer = "adam",loss = 'binary_crossentropy', metrics = ['accuracy']) # algorithm for stochastic gradient descent

classifier.fit(X_train,y_train, batch_size=10, nb_epoch = 100)

print("fitted")

# Predicting the Test set results
y_pred = classifier.predict(X_test)

print(y_pred)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


print(cm)

accuracy = ( cm[0][0] + cm [1][1] ) / 2000

print (accuracy)
print ("------------------------------------")


