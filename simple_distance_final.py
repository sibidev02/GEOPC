# code for computing simple distances and finding the points above the threshold and visualizing it

# Importing the Open3D library for 3D data processing
import open3d as o3d

# Importing NumPy for numerical operations, particularly with arrays
import numpy as np

# Importing Matplotlib for plotting graphs and visualizations
import matplotlib.pyplot as plt

# Importing KDTree from SciPy's spatial module for efficient nearest-neighbor searches
from scipy.spatial import KDTree

def simple_distance_computation(source_points, target_points):
    """
    Compute Euclidean distances from each point in source_points to the nearest point in target_points.

references used for simple_distance computation:
    -This function uses the KDTree nearest neighbor search algorithm from the SciPy library.
    Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html (accessed on 28th Sept 2023)
    - Euclidean Distance: https://en.wikipedia.org/wiki/Euclidean_distance
    - Open3D: A modern library for 3D data processing: https://www.open3d.org
    -Open3D Python API: http://www.open3d.org/docs/release/python_api/open3d.io.read_point_cloud.html
    -NumPy documentation: https://numpy.org/doc/stable/
    -Matplotlib documentation: https://matplotlib.org/stable/users/colormaps.html
    using chatgpt for solving errors and debugging

    Arguments:
        source_points (np.ndarray): First point cloud as an (N, 3) array.
        target_points (np.ndarray): Second point cloud as an (M, 3) array.

    Returns:
        np.ndarray: Euclidean distances from each point in source_points to the nearest point in target_points.
    """
    tree = KDTree(source_points)
    distances, _ = tree.query(target_points)
    return distances

# Load the point clouds from the file paths
point_cloud_file_path_1 = r"C:\Users\devsibi\PycharmProjects\pythonProject4\lidar\las_data\cluster_3_118.ply"
point_cloud_file_path_2 = r"C:\Users\devsibi\PycharmProjects\pythonProject4\lidar\las_data\cluster_3_115.ply"

# read the point cloud using open3D
point_cloud_1 = o3d.io.read_point_cloud(point_cloud_file_path_1)
point_cloud_2 = o3d.io.read_point_cloud(point_cloud_file_path_2)

# Convert the open3D point cloud format to numpy arrays for easy processing
source_points = np.asarray(point_cloud_1.points)
target_points = np.asarray(point_cloud_2.points)

# Calculate Euclidean distances for each point in source cloud to the nearest point in the target cloud
distances = simple_distance_computation(source_points, target_points)

# Print the number of points in each point cloud
num_points_1 = len(source_points)
num_points_2 = len(target_points)
print(f"Number of points in point cloud 1: {num_points_1}")
print(f"Number of points in point cloud 2: {num_points_2}")

# Print distances from each point in source point cloud  to the nearest point in target point cloud
for i, distance in enumerate(distances):
    print(f"Point {i} distance: {distance}")

# set a distance threshold value
threshold = 0.01

# Identify points in the source cloud that are farther than the threshold from their nearest neighbor
indices_above_threshold = np.where(distances > threshold)[0]
points_above_threshold = len(indices_above_threshold)
print(f"Number of points with distance greater than {threshold}: {points_above_threshold}")

# Make a filtered point cloud if any of the points are over the threshold value
if points_above_threshold > 0:
    # filter the points that exceeds the threshold
    filtered_points = source_points[indices_above_threshold]
    # filter colors(if available)
    filtered_colors = np.asarray(point_cloud_1.colors)[indices_above_threshold] if point_cloud_1.colors else None

    # create a new point cloud object to store the filtered pont cloud
    filtered_point_cloud = o3d.geometry.PointCloud()
    filtered_point_cloud.points = o3d.utility.Vector3dVector(filtered_points)

    # add colors to the new point cloud
    if filtered_colors is not None:
        filtered_point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)

    # Specify the path to save the filtered point cloud
    filtered_point_cloud_file_path = "filtered_point_cloud.ply"

    # Save the filtered point cloud to the specified path
    o3d.io.write_point_cloud(filtered_point_cloud_file_path, filtered_point_cloud)

    print(f"Filtered point cloud saved to {filtered_point_cloud_file_path}")

    # Visualize the filtered point cloud
    o3d.visualization.draw_geometries([filtered_point_cloud])
else:
    print("No points found with distance greater than the threshold.")

# Visualize the original point cloud with color mapped to distances
min_distance = np.min(distances)
max_distance = np.max(distances)

# create color map based on the distances
if min_distance != max_distance:
    colors = plt.get_cmap("jet")((distances - min_distance) / (max_distance - min_distance))
else:
    colors = plt.get_cmap("jet")(np.zeros_like(distances))
# assign colour map to the source point cloud.
point_cloud_1.colors = o3d.utility.Vector3dVector(colors[:, :3])

# Visualize the original point cloud with distances color mapped
o3d.visualization.draw_geometries([point_cloud_1])
