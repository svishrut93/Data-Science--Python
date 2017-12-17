import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


X = [1,2,3,4,5,6,7,8,9,10]
Y = [1,4,9,16,25,36,49,64,81,100]


fig,axes = plt.subplots(nrows=1,ncols=2)
axes[0].plot(X,Y)
axes[0].fill_between(X,0,Y,alpha=0.5)
axes[1].fill_between(Y,10,X,color = "#eeefff")


axes[1].plot(Y,X)

plt.tight_layout()
plt.show()




