import pandas as pd
import numpy as np
import torch
from sklearn.cluster import KMeans, MeanShift,estimate_bandwidth,AgglomerativeClustering,DBSCAN
from sklearn import metrics
from sklearn.utils.linear_assignment_ import linear_assignment
import time

def cluster_acc(Y_pred, Y):
  # assert Y_pred.size == Y.size
  D = max(Y_pred.max(), Y.max())+1
  w = np.zeros((D,D), dtype=np.int64)
  for i in range(Y_pred.size):
    w[Y_pred[i], Y[i]] += 1
  # task assignment problem, one person corresponds one task, Harold Kuhn algorithm can solve it,w is the cost matric
  ind = linear_assignment(w.max() - w)
  print('map',ind)
  # +1247+1589
  return (sum([w[i,j] for i,j in ind])*1.0)/(Y_pred.size), w, ind


# # activtion vector
data_dir = "./1123/OUR/"
filename = 'centroid_unknowdata_CICIDS_cluster.csv'
## validataion dataset
# filename = 'centroid_knowdata_CICIDS_1124.csv'
raw_data_filename = data_dir + filename
raw_data = pd.read_csv(raw_data_filename, header=None)
features = raw_data.iloc[:, 0:raw_data.shape[1] - 1]
np_features = np.array(features)
labels = raw_data.iloc[:, raw_data.shape[1] - 1:]
labels = labels.values.ravel()
np_labels = np.array(labels)
df = pd.DataFrame(np_features, index=np_labels)
np_features = df.values
np_labels = df.index.values
# df_drop = df.drop(index=[2,3,4])
# np_features = df_drop.values
# np_labels = df_drop.index.values
print(np_features.shape)
print(np_labels.shape)
print(np_labels)

### First procedure
### determine the cluster number (eps and min_samples are determined by using DBSCAN on test known semantic data)
# Compute DBSCAN
# t0=time.clock()
# db = DBSCAN(eps=0.41, min_samples=180).fit(np_features)
# core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# core_samples_mask[db.core_sample_indices_] = True
# labels = db.labels_
# # Number of clusters in labels, ignoring noise if present.
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# n_noise_ = list(labels).count(-1)
# t1 = time.clock()
# print('cluster time',str(t1-t0))
# print('Estimated number of clusters: %d' % n_clusters_)
# print('Estimated number of noise points: %d' % n_noise_)

### Second procedure
### use kmeans to calculate the cluster centroid
# kmeans result
t2 = time.clock()
ms = KMeans(n_clusters=4, init='k-means++', max_iter = 3000)
y_pred = ms.fit_predict(np_features)
t3 = time.clock()
print('kmeans time',str(t3-t2))
accuracy = metrics.adjusted_rand_score(np_labels,y_pred)
accuracy2 = metrics.adjusted_mutual_info_score(np_labels,y_pred)
print('adjusted_rand_score',accuracy)
print('mutual_info_score_acc',accuracy2)
# accuracy3 = metrics.silhouette_score(np_labels.reshape(-1,1),y_pred)
accuracy3 = metrics.silhouette_score(np_features,y_pred)
print('Silhouette Coefficient',accuracy3)
print(metrics.calinski_harabaz_score(np_features, y_pred))
clust_acc, cost, map_label = cluster_acc(y_pred,np_labels)
print('cluster_acc',clust_acc)
print('cluster centroid',ms.cluster_centers_)
print('cost matric',cost)
gd_centroid = np.zeros((ms.cluster_centers_.shape[0],ms.cluster_centers_.shape[1]))
for i, j in map_label:
    gd_centroid[j] = ms.cluster_centers_[i]
print('gd_centroid',gd_centroid)
