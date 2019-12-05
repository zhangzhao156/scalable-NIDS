import pandas as pd
import numpy as np
import torch
from sklearn.cluster import KMeans, MeanShift,estimate_bandwidth,AgglomerativeClustering,DBSCAN
from sklearn import metrics
from sklearn.metrics import pairwise_distances
import scipy
from scipy.spatial.distance import cdist
from  scipy.spatial.distance import euclidean
from scipy import stats
from sklearn.utils.linear_assignment_ import linear_assignment
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def normalize_KDD99(df):
    result = df.copy()
    for feature_name in df.columns:
        if feature_name == 0 :
            max_value = 58329
            min_value = 0
        elif feature_name == 2 :
            max_value = 69
            min_value = 0
        elif feature_name == 3:
            max_value = 10
            min_value = 0
        elif feature_name == 4:
            # max_value = 1379963888
            max_value = 693375640
            min_value = 0
        elif feature_name == 5:
            # max_value = 1309937401
            max_value = 5203179
            min_value = 0
        elif feature_name == 9:
            max_value = 101
            min_value = 0
        elif feature_name == 10:
            max_value = 5
            min_value = 0
        elif feature_name == 12:
            max_value = 884
            min_value = 0
        elif feature_name == 15:
            max_value = 993
            min_value = 0
        elif feature_name == 16:
            max_value = 100
            min_value = 0
        elif feature_name == 17:
            max_value = 5
            min_value = 0
        elif feature_name == 18:
            max_value = 8
            min_value = 0
        else:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
        if max_value != min_value:
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

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
data_dir = "./"
filename = 'centroid_unknowdata_NSLKDD_4_0703_our.csv'
# filename = 'centroid_unknowdata_KDD99_4_0703_our.csv'
raw_data_filename = data_dir + filename
raw_data = pd.read_csv(raw_data_filename, header=None)
features = raw_data.iloc[:, 0:raw_data.shape[1] - 1]
# no normalization -> get the needed cluster centroid
np_features = np.array(features)
# # normalize the features
# features_norm = normalize(features)
# np_features = np.array(features_norm)
# print(np_features)
labels = raw_data.iloc[:, raw_data.shape[1] - 1:]
labels = labels.values.ravel()
np_labels = np.array(labels)
df = pd.DataFrame(np_features, index=np_labels)
df_drop = df.drop(index=[3])
np_features = df_drop.values
np_labels = df_drop.index.values
print(np_features.shape)
print(np_labels.shape)
print(np_labels)


# use DBSCAN to determine the cluster number
db = DBSCAN(eps=0.18, min_samples=110).fit(np_features)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)



# # use kmeans to find the hidden clusters
# ms = KMeans(n_clusters=2, init='k-means++', max_iter = 3000)
# y_pred = ms.fit_predict(np_features)
# accuracy = metrics.adjusted_rand_score(np_labels,y_pred)
# accuracy2 = metrics.adjusted_mutual_info_score(np_labels,y_pred)
# print('adjusted_rand_score',accuracy)
# print('mutual_info_score_acc',accuracy2)
# accuracy3 = metrics.silhouette_score(np_features,y_pred)
# print('Silhouette Coefficient',accuracy3)
# print(metrics.calinski_harabaz_score(np_features, y_pred))
# clust_acc, cost, map_label = cluster_acc(y_pred,np_labels)
# print('cluster_acc',clust_acc)
# print('cluster centroid',ms.cluster_centers_)
# print('cost matric',cost)
# gd_centroid = np.zeros((ms.cluster_centers_.shape[0],ms.cluster_centers_.shape[1]))
# for i, j in map_label:
    # gd_centroid[j] = ms.cluster_centers_[i]
# print('gd_centroid',gd_centroid)




