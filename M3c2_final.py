# code for computing M3c2 distances and finding the points above the threshold and visualizing it

# Importing the Open3D library for 3D data processing
import open3d as o3d

# Importing NumPy for numerical operations, particularly with arrays
import numpy as np

# Importing Matplotlib for plotting graphs and visualizations
import matplotlib.pyplot as plt

# Importing KDTree from SciPy's spatial module for efficient nearest-neighbor searches
from scipy.spatial import KDTree

def m3c2_projected_distances(src_points, src_normals, target_points):
    """
    Calculate M3C2-like distances from each point in the source point cloud (`src_points`)
    to the nearest point in the target point cloud (`target_points`), projected along
    the normal vectors of the source points.

references used for M3C2 computation:
    -This function uses the KDTree nearest neighbor search algorithm from the SciPy library.
    Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html (accessed on 28th Sept 2023)
    - Open3D: A modern library for 3D data processing: https://www.open3d.org
    -"M3C2: A Robust Surface-Matching Algorithm"by Lague, B., et al., 2013.
    -Open3D Python API: http://www.open3d.org/docs/release/python_api/open3d.io.read_point_cloud.html
    -NumPy documentation: https://numpy.org/doc/stable/
    -Matplotlib documentation: https://matplotlib.org/stable/users/colormaps.html
    using chatgpt for solving errors and debugging

    Args:
        src_points (np.ndarray): Source point cloud coordinates as (N, 3) array.
        src_normals (np.ndarray): Source point cloud normals as (N, 3) array.
        target_points (np.ndarray): Target point cloud coordinates as (M, 3) array.

    Returns:
        np.ndarray: Projected distances from each point in the source point cloud to the nearest point in the target cloud.
    """
    # Build a KDTree from the target cloud to quickly find nearest neighbors
    target_tree = KDTree(target_points)

    # Prepare an array to store the distances
    projected_distances = np.zeros(len(src_points))

    # Normalize the normal vectors for projection
    src_normals = src_normals / np.linalg.norm(src_normals, axis=1, keepdims=True)



    # Compute the M3C2-like distance for each point in the source cloud
    for i, point in enumerate(src_points):
        normal_vec = src_normals[i]

        # Find the nearest point in the target cloud
        nearest_idx = target_tree.query(point)[1]
        nearest_point = target_points[nearest_idx]

        # Calculate the vector from the source point to the nearest target point
        vector_to_nearest = nearest_point - point

        # Project this vector onto the normal direction to get the distance along the normal
        distance_along_normal = np.dot(vector_to_nearest, normal_vec)

        # Store the projected distance
        projected_distances[i] = distance_along_normal

    return projected_distances

# Load the point clouds from file paths
cloud1_path = r"C:\Users\devsibi\PycharmProjects\pythonProject4\lidar\las_data\cluster_1_118.ply"
cloud2_path = r"C:\Users\devsibi\PycharmProjects\pythonProject4\lidar\las_data\cluster_1_0505.ply"

# Read point clouds using Open3D
source_cloud = o3d.io.read_point_cloud(cloud1_path)
target_cloud = o3d.io.read_point_cloud(cloud2_path)

# Ensure that both point clouds contain data
if len(source_cloud.points) == 0 or len(target_cloud.points) == 0:
    raise ValueError("One or both point clouds are empty. Please load valid data.")

# Estimate normals for the source point cloud
source_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

# Convert the point clouds into NumPy arrays
src_points = np.asarray(source_cloud.points)
src_normals = np.asarray(source_cloud.normals)
tgt_points = np.asarray(target_cloud.points)

# Check if the normals were successfully calculated
if len(src_normals) == 0:
    raise ValueError("Failed to compute normals for the source point cloud.")

# Define a threshold for filtering distances
distance_threshold = 0.001

# Calculate M3C2-like distances, which are projected along the source cloud's normals
m3c2_distances = m3c2_projected_distances(src_points, src_normals, tgt_points)

# Print some information about the point clouds
print(f"Source cloud contains {len(src_points)} points.")
print(f"Target cloud contains {len(tgt_points)} points.")
print(f"Number of normals computed: {len(src_normals)}")

# Identify points with distances that exceed the threshold
outlier_indices = np.where(np.abs(m3c2_distances) > distance_threshold)[0]
outlier_points = src_points[outlier_indices]
print(f"Number of points with distances exceeding {distance_threshold}: {len(outlier_points)}")

# Create and visualize a new point cloud containing only points above the threshold
if len(outlier_points) > 0:
    # Create a new point cloud for points exceeding the threshold
    outlier_cloud = o3d.geometry.PointCloud()
    outlier_cloud.points = o3d.utility.Vector3dVector(outlier_points)

    # Visualize the filtered point cloud
    print(f"Visualizing points with projected distances greater than {distance_threshold}")
    o3d.visualization.draw_geometries([outlier_cloud])
else:
    print(f"No points exceed the distance threshold of {distance_threshold}.")

# Visualize the original point cloud with distances mapped to color
min_dist = np.min(m3c2_distances)
max_dist = np.max(m3c2_distances)

if min_dist != max_dist:
    # Normalize distances for color mapping
    color_mapping = plt.get_cmap("jet")((m3c2_distances - min_dist) / (max_dist - min_dist))
else:
    color_mapping = plt.get_cmap("jet")(np.zeros_like(m3c2_distances))

# Assign the color mapping to the source point cloud
source_cloud.colors = o3d.utility.Vector3dVector(color_mapping[:, :3])

# Visualize the source point cloud with distances color-mapped
o3d.visualization.draw_geometries([source_cloud])