# Python File to test your KMeans class

from K_Means_ClusteringNecatiBuhur import KMeans
import numpy as np
import pandas as pd
import os

# Read the data
df = pd.read_csv(os.path.join('winequality-red.csv'),sep=';')

# Get the X and y
X = df.drop(['quality'], axis=1) # First 11 columns
y = df['quality'] # Last labels

# Get the number of clusters
num_of_clusters = len(np.unique(y))

k = KMeans(k=num_of_clusters, max_iters=200, plot_steps=True)
y_predicted = k.predict(X.to_numpy())

from sklearn import metrics
print("Homogeneity: %0.3f" % metrics.homogeneity_score(y, y_predicted))
print("Completeness: %0.3f" % metrics.completeness_score(y, y_predicted))
print("V-measure: %0.3f" % metrics.v_measure_score(y, y_predicted))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(y, y_predicted))
print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(y, y_predicted))
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, y_predicted))





