import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import  KMeans


df = pd.read_excel("Dataset.xlsx")
print(df)


#Perform standardization
scalar = StandardScaler()
print(df.columns)

scalar.fit(df.drop('Team',axis =1))
scaled_features = scalar.transform(df.drop('Team',axis =1))

scaled_features_df = pd.DataFrame(scaled_features)

print("Scaled features:")

print(scaled_features_df)



kmeans = KMeans(n_clusters=2)
kmeans.fit(scaled_features)
print("kmeans clusters")
print(kmeans.cluster_centers_)
print(kmeans.labels_)

##2nd map
kmeans2 = KMeans(n_clusters=3)
kmeans2.fit(scaled_features)
print("kmeans2 clusters")
print(kmeans2.cluster_centers_)
print(kmeans2.labels_)



fig , axes = plt.subplots(nrows =1 , ncols = 3)
# axes[0]= fig.add_axes ([0.1,0.1,0.8,0.8])
# axes2[1] = fig.add
# axes1.scatter(scaled_features[:0],scaled_features[:1])


axes[0].scatter(scaled_features_df[0],scaled_features_df[1], c= '#004C99')
axes[0].set_title('Original plot')

axes[1].scatter(scaled_features_df[0],scaled_features_df[1],c=kmeans.labels_)   #color labels according to labels obtained from kmeans : c = kmeans.labels_
axes[1].set_title('2 Clusters')
axes[1].scatter(kmeans.cluster_centers_[0][0],kmeans.cluster_centers_[0][1],color= 'blue', marker='x')
axes[1].scatter(kmeans.cluster_centers_[1][0],kmeans.cluster_centers_[1][1],color= 'blue', marker='x')




axes[2].scatter(scaled_features_df[0],scaled_features_df[1],c=kmeans2.labels_)
axes[2].set_title('3 Clusters')
axes[2].scatter(kmeans2.cluster_centers_[0][0],kmeans2.cluster_centers_[0][1],color= 'blue', marker='x')
axes[2].scatter(kmeans2.cluster_centers_[1][0],kmeans2.cluster_centers_[1][1],color= 'blue', marker='x')
axes[2].scatter(kmeans2.cluster_centers_[2][0],kmeans2.cluster_centers_[2][1],color= 'blue', marker='x')
plt.show()




# plt.show()