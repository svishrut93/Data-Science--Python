import numpy as np
import numpy.linalg as lin
import pandas as pd

V1row = np.array([2,1,0,3,1,1])
V2row = np.array([2,3,1,2,3,0])
V3row = np.array([1,0,3,3,1,1])
V4row = np.array([2,3,1,0,3,2])


combinedV = np.array([[2,1,0,3,1,1],
                     [2,3,1,2,3,0],
                     [1,0,3,3,1,1],
                     [2,3,1,0,3,2]])

print (V1row.shape)
print(type(V1row))
print (combinedV.shape)


A = np.transpose(combinedV)
print('A matrix')
print(A)
print(type(A))


#compute : L = A transpose A

L = np.dot(np.transpose(A),A)
print("L matrix ")
print(L)

eigenvals = (lin.eigvals(L))
print("Eigen values of L ")
print("-----------------------")
print(eigenvals)


eigenvectors = lin.eig(L)[1]
print("eigen vectors of L ")
print("-----------------------")
print(eigenvectors)

v1 = np.array(eigenvectors[0])
v2 = np.array(eigenvectors[1])
v3 = np.array(eigenvectors[2])
v4 = np.array(eigenvectors[3])
# v2 = eigenvectors[1]
# v3 = eigenvectors[2]
# v4  = eigenvectors[3]

print(type(eigenvectors))
print("Printing eigen vectors as individual numpy arrays: ")
print("---------------------------------------------------")
print(v1)
print(v2)
print(v3)
print(v4)

#Eigenvectors of the covariance matrix C :
# mu i = Avi
#vi = the eigenvectors calculated


print(A.shape)
print(v1.shape)


#multiplication :
# in tb it is u1 , u2 , u3 , u 4
#remember the number of u's will always be equal to number of eigenvecttors
#very important ! understand this
#what i am doing here
# the outer loop j , iterates over every row of matrix A from text book
# the inner loop i : iterates over every row in the eigenvector numpy array
# remember how numpy stores arrays

rowsum=0

u1list=[]
u2list=[]
u3list=[]
u4list=[]

for j in range(0,len(A)):
    #u1.append(rowsum)
    #print("@@@@@@@@@@@@")
    rowsum=0
    for i in range(0,len(eigenvectors)):

        # print (A[j][i])
        # print(eigenvectors[i][0])
        rowsum = rowsum+(A[j][i]*eigenvectors[i][0])  # i [0] means you are iterating over rows like in tb: works only for v1
        #print (rowsum)
        if (i==len(eigenvectors)-1):
            u1list.append(rowsum)

#print(u1)

for j in range(0,len(A)):
    #u1.append(rowsum)
    #print("@@@@@@@@@@@@")
    rowsum=0
    for i in range(0,len(eigenvectors)):

        # print (A[j][i])
        # print(eigenvectors[i][0])
        rowsum = rowsum+(A[j][i]*eigenvectors[i][1])  # i [1] means you are iterating over rows like in tb: works only for v2
        #print (rowsum)
        if (i==len(eigenvectors)-1):
            u2list.append(rowsum)

#print(u2)


for j in range(0,len(A)):
    #u1.append(rowsum)
    #print("@@@@@@@@@@@@")
    rowsum=0
    for i in range(0,len(eigenvectors)):

        # print (A[j][i])
        # print(eigenvectors[i][0])
        rowsum = rowsum+(A[j][i]*eigenvectors[i][2])  # i [1] means you are iterating over rows like in tb: works only for v2
        #print (rowsum)
        if (i==len(eigenvectors)-1):
            u3list.append(rowsum)

#print(u3)


for j in range(0,len(A)):
    #u1.append(rowsum)
    #print("@@@@@@@@@@@@")
    rowsum=0
    for i in range(0,len(eigenvectors)):

        # print (A[j][i])
        # print(eigenvectors[i][0])
        rowsum = rowsum+(A[j][i]*eigenvectors[i][3])  # i [1] means you are iterating over rows like in tb: works only for v2
        #print (rowsum)
        if (i==len(eigenvectors)-1):
            u4list.append(rowsum)

#print(u4)

#converting all the lists into a numpy array :

print ("Eigenvectors of the covariance matrix C - as lists ")
print(u1list)
print("#####")
print(u2list)
print("#####")
print(u3list)
print("######")
print(u4list)
print("######")


#converting all the lists of eigenvectors of the covariance matrix into a numpy array :
u1 = np.asarray(u1list)
u2 = np.asarray(u2list)
u3 = np.asarray(u3list)
u4 = np.asarray(u4list)

print ("Eigenvectors of the covariance matrix C - as numpy arrays  ")

print(u1)
print(u2)
print(u3)
print(u4)

# wNumpyarrays = np.empty([4][6])
# print(wNumpyarrays)


print("Creating the same type of list programmatically: Eigenvectors of the covariance matrix C - as single numpy array ")
listoflists = [u1list,u2list,u3list,u4list]

uNumpyarrays = np.array(listoflists)

print(uNumpyarrays)


print("We normalize each eigenvector by dividing by its length. This yeilds the unit eigenvectors")
divisor = len(uNumpyarrays[0])

uNumpyarrays = uNumpyarrays/divisor
muNumpyarrays = uNumpyarrays
print("Normalized: Eigenvectors of the covariance matrix C - as single numpy array "
       "this is also our Eigenspace")

print(muNumpyarrays)
print(len(muNumpyarrays))

print("Next we determine the scoring matrix delta = (omega1 , omega2 , omega3 , omega4 ) by projecting the training vectors "
      "onto the eigen space. \nSpecifically computing weight vectors ")

Delta = []


#computing dot product Vi . mu[i]

# uncomment if you wanty to see labels for the different lists as omega1, omega2...
# listnaming = [i for i in range (10)]
# print("List naming: ")
# print(1,2,3,4)
# dct_of_Omega = {} #uncomment if you wanty to see labels for the different lists as omega1, omega2....
biglist = []

omega1 =[]
omega2 =[]
omega3 = []
omega4 = []

for i in range (0,4):
    lll=[]
    for j in range (0,4):
        temp=(np.dot(combinedV[i],muNumpyarrays[j]))
        lll.append(temp)


    # dct_of_Omega['omega_%s'%i] = lll #uncomment if you wanty to see labels for the different lists as omega1, omega2....
    biglist.append(lll)

# print(dct_of_Omega) #uncomment if you wanty to see labels for the different lists as omega1, omega2....


Omega = np.array(biglist)
print(Omega)
# Omega.reshape(4,4)

print("The training vector is formed....\n"
      "when comparing from 78 read it so that every list in omega is a coulm of delta on 78 ")

delta = np.copy(Omega)

print(delta)

#At this point choose the eigenvectors of your choice

#we are choosing only forst 3 principal componenets


#scoring a certain vector :

scoreV1 = np.array([4,0,1,8,3,2])


BigW = []
lll2=[]
for j in range (0,3):   #3 because we consider only 3 principal components

        temp=(np.dot(scoreV1,muNumpyarrays[j]))
        lll2.append(temp)

BigW.append(lll2)


print("W is found as ....")
W = np.array(BigW)
print(W)



# Performing subtractio for ED

ED = []
lll3=[]
for i in range (0,3):

        for j in range (0,3):
            tempo = W[0][i]-delta[i][j]
            lll3.append(temp)

ED.append(lll3)


print (ED)

dist = []
dist1 = np.linalg.norm(W[0]-delta[0][0:3])
dist.append(dist1)

dist2 = np.linalg.norm(W[0]-delta[1][0:3])
dist.append(dist2)
dist3 = np.linalg.norm(W[0]-delta[2][0:3])
dist.append(dist3)
print("The distance is ....")
mindist = min(float(s) for s in dist )

print(mindist)

print()
#X.mu1 ; X.mu2 ; X.mu3






















