import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

# Load data from input file
x = np.loadtxt('data_clustering.txt', delimiter=',')

# Estimate the bandwidth of X
bandwidth_x = estimate_bandwidth(x, quantile=0.1, n_samples=len(x))

# Cluster data with MeanShift
mean_shift_model = MeanShift(bandwidth=bandwidth_x, bin_seeding=True)
mean_shift_model.fit(x)

# Extract the centers of clusters
cluster_centers = mean_shift_model.cluster_centers_
print('\nCenters of clusters:\n', cluster_centers)

# Estimate the number of clusters
labels = mean_shift_model.labels_
num_clusters = len(np.unique(labels))
print("\nNumber of clusters in input data =", num_clusters)

# Plot the points and cluster centers
plt.figure()
markers = 'o*xvs'
for i, marker in zip(range(num_clusters), markers):
    # Plot points that belong to the current cluster
    plt.scatter(x[labels == i, 0], x[labels == i, 1], marker=marker, color='black')
    # Plot the cluster center
    cluster_center = cluster_centers[i]
    plt.plot(cluster_center[0], cluster_center[1], marker='o',
             markerfacecolor='black', markeredgecolor='black',
             markersize=15)
plt.title('Clusters')
plt.show()
