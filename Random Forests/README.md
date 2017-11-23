<h1> Random Forests and Decision Trees </h1> 

The datset used is the Kyphosis dataset from Kaggle. 

This contains the information of patients who have been found to have Kyphosis recurrence after performing a surgery. 
The task at hand is to predict if a person has kyphosis or not (Classification Problem)

The file used is uploaded as kyphosis.csv 


The data is cleaned and pre-processed. The visualizations about the data along with outputs are contained inside the data visualization folder. 
After extracting insights about the data with the help of matplotlib and seaborn, I forst used a decision tree classifier for prediction, with the below results. 

<h3>Decision Tree </h3>

![alt text](https://github.com/svishrut93/Data-Science--Python/blob/master/Random%20Forests/Data%20Visualization/Results%20Decision%20Trees.PNG)

<h3>Random Forest  </h3>


![alt text](https://github.com/svishrut93/Data-Science--Python/blob/master/Random%20Forests/Data%20Visualization/Results%20Random%20forests.PNG)


We observe the random forest shows slightly better results than the decision tree. The difference is small because the dataset is small. As the size of the dataset increases, the random forests invariably show better results than decision trees.


![alt text](https://github.com/svishrut93/Data-Science--Python/blob/master/Random%20Forests/Data%20Visualization/Comparision.png)


![alt text](https://github.com/svishrut93/Data-Science--Python/blob/master/Random%20Forests/Data%20Visualization/Distribution%20for%20kyphosis%20absent.png)




