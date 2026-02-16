# This program is used to generate the bounding boxes for each cluster we want to make from our point cloud file.
# Below libraries are imported to work with functions associated with them.
# ________________________________________________________________________________________________________________
# references
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
# https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html
# https://www.geeksforgeeks.org/davies-bouldin-index/
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html
# http://www.open3d.org/docs/latest/python_api/open3d.geometry.AxisAlignedBoundingBox.html
# ChatGPT for solving error and issues in code.
# ______________________________________________________________________________________________________________________
# Below libraries are imported to work with functions associated with them.
import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
import json

# Load the combined point cloud and use
combined_ply_file = 'C:/Users/91997/PycharmProjects/pythonProject/main_combined_file_point_cloud.ply'
pcd = o3d.io.read_point_cloud(combined_ply_file)
coordinates = np.asarray(pcd.points)  # Convert the points to a numpy array to be used in clustering.

# Determine the optimal number of clusters using the Davies-Bouldin Index.
# Davies-Bouldin Index is a metric for evaluating the quality of clustering. we can choose a range for clusters.
davies_bouldin_indices = []    # list to store davies bouldin indices.
range_n_clusters = list(range(100, 101))

# The script uses KMeans clustering from the sklearn library to group the points in the cloud into clusters.
# for this a for loop is used  in the range of clusters and davies_bouldin_score is used calculate
# the Davies-Bouldin Index based on the clustering result.
for no_of_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=no_of_clusters, init='k-means++', n_init=10, max_iter=300)
    cluster_labels = kmeans.fit_predict(coordinates)
    db_index = davies_bouldin_score(coordinates, cluster_labels)
    davies_bouldin_indices.append(db_index)

# Plot the Davies-Bouldin indices for better visualization.
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, davies_bouldin_indices, marker='o')
plt.title('Davies-Bouldin Index Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Davies-Bouldin Index')
plt.show()

# Based on the Davies-Bouldin plot, choose the optimal number of clusters
optimal_k = range_n_clusters[np.argmin(davies_bouldin_indices)]

# Perform KMeans clustering using the optimal value. It models the cluster into optimal value which is stored in labels
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, max_iter=100)
kmeans.fit(coordinates)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_  # the centriods of clusters are stored in centroids.

# Create bounding boxes for each cluster.
bounding_boxes = []  # list to save the minimum and maximum coordinates of each bounding box.
grid_boxes = []      # list to save each bounding box.

# for loop used to run in the range of optimal value of clusters.
for i in range(optimal_k):
    cluster_points = coordinates[labels == i]

    # condition to Check if the cluster is not empty
    if len(cluster_points) == 0:
        continue

    # Calculate the minimum and maximum coordinates (bounding box) for the cluster
    min_coords = np.min(cluster_points, axis=0)
    max_coords = np.max(cluster_points, axis=0)

    # Create an AxisAlignedBoundingBox for the cluster
    grid_box = o3d.geometry.AxisAlignedBoundingBox(min_coords, max_coords)
    grid_box.color = (0, 0, 1)  # color is set for each for the bounding box.
    grid_boxes.append(grid_box) # each bounding boxes are appended.

    # function to save bounding boxes.
    bounding_boxes.append({
        'min_coords': min_coords.tolist(),
        'max_coords': max_coords.tolist()
    })

# all the bounding boxes  are saved to a JSON file that will be used to create clusters from different point cloud data.
with open('new_bounding_boxes.json', 'w') as f:
    json.dump(bounding_boxes, f)

# Visualize the combined point cloud and the bounding boxes
o3d.visualization.draw_geometries([pcd] + grid_boxes)

#Print the number of points in each cluster to check number of points.
for i in range(optimal_k):
    print(f"Cluster {i}: {np.sum(labels == i)} points")
