<h1>Logistic Regression </h2>

The dataset used to perform logistic regression is the famous Titanic dataset from kaggle.<br>
While linear regression is useful for predicting continuous values (like housing prices in the previous folder), we cant use it for problems like binary classification 
The model wont be a good fit. 

So we try to solve binary classification using logistic regression. 

The dataset csv files are available for download in this folder. 

The first file : Data_cleaning_LR.py is the cleaning process done on the training set. It contains code with detailed comments on how to clean the data so that it contains numerical values, fit for being fed into a machine learning algorithm.

Also refer to the data cleaning visualization folder to understand how data was interpreted and cleaning performed.

Then check file Logistic Regression.py for code of ML algorithm. 

The data to predict is whether a passenger survived or not. 

We get an accuracy here of 0.809 

Output :
             precision    recall  f1-score   support

          0       0.79      0.92      0.85       174
          1       0.85      0.65      0.74       120

avg / total       0.81      0.81      0.80       294

<class 'numpy.ndarray'>
[[160  14]
 [ 42  78]]
Accuracy = 0.809523809524

Process finished with exit code 0
