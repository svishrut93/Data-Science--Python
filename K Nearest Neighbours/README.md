<h1> K Nearest Neighbours </h1> 

K NN is performed on the dataset described as "classified data".
This dataset closely remebles anonymised data usually shared during many data science interviews. 

It is first cleaned and standardized to be fed into the ML algorithm. <br>
The target class contains of classifying an observation to belonging in 0 or 1. <br> 

All other features are used to make the prediction (Independent Variables). 
The average accuracy for the above algorithm is in the range of 0.92 - 0.94(for the range of specified K values) <br> 
Different values of K are tried and the error rate generated for each is plotted. The optimal K values comes to around 26. <br> 
This is demonstarted by the below graph <br>

![alt text](https://github.com/svishrut93/Data-Science--Python/blob/master/K%20Nearest%20Neighbours/KNN%20Graph-Error%20Rate.png)


Output : 



Confusion matrix : 
[[145  14]
 [ 12 129]]
Classification report
             precision    recall  f1-score   support

          0       0.92      0.91      0.92       159
          1       0.90      0.91      0.91       141

avg / total       0.91      0.91      0.91       300

ACCURACY 
0.913333333333
ERROR RATE : 
0.0866666666667
done
