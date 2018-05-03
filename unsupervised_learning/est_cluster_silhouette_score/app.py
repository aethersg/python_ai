import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans

# Load data from input file
x = np.loadtxt('data_quality.txt', delimiter=',')
# Initialize variables
scores = []
values = np.arange(2, 10)
# Iterate through the defined range
for num_clusters in values:
    # Train the KMeans clustering model
    kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
    kmeans.fit(x)
    score = metrics.silhouette_score(x, kmeans.labels_, metric='euclidean', sample_size=len(x))
    print("\nNumber of clusters =", num_clusters)
    print("Silhouette score =", score)
    scores.append(score)

# Plot silhouette scores
plt.figure()
plt.bar(values, scores, width=0.7, color='black', align='center')
plt.title('Silhouette score vs number of clusters')
plt.show()

# Extract best score and optimal number of clusters
num_clusters = np.argmax(scores) + values[0]
print('\nOptimal number of clusters =', num_clusters)

# Plot data
plt.figure()
plt.scatter(x[:, 0], x[:, 1], color='black', s=80, marker='o', facecolors='none')
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
plt.title('Input data')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
