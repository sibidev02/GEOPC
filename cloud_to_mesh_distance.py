# This function takes a 3D mesh and a point cloud as input.
# It computes the signed distance between each point in the cloud and the mesh,
# and returns these distances as a NumPy array. The distances are getting printed.
# This code is made for testing small files, a separate code which allows streaming through
# the created clusters is used for the actual rockfall detection. But the algorithm used is same.


#----------------------------------------------------------------------------------------------------------------------
# References:
# Open3D: A Modern Library for 3D Data Processing https://arxiv.org/pdf/1801.09847
# https://github.com/isl-org/Open3D
# Generating 3D Meshes with Python: https://orbi.uliege.be/bitstream/2268/254933/1/TDS_generate_3D_meshes_with_python.pdf
# 3D Point Cloud Reconstruction: https://ar5iv.labs.arxiv.org/html/2112.12907
# Chatgpt for debugging and fixing issues
#-----------------------------------------------------------------------------------------------------------------------


import open3d as o3d # For 3D data processing,including point clouds and meshes.
import numpy as np # Importing NumPy for numerical computations.
import matplotlib.pyplot as plt # to map point cloud distances to color for visualization.

# Defining a function to calculate cloud to mesh distance from input directories provided.
def cloud_mesh_to_distance(mesh: o3d.t.geometry.TriangleMesh,
                                   cloud: o3d.t.geometry.PointCloud) -> np.ndarray:
    # For computing distances by "casting rays" from the point cloud towards the mesh.
    # Raycasting is a technique used to compute the intersection of rays (straight lines) with objects in a 3D scene.
    # Here Raycasting is used to add the mesh and compute the distance between cloud and mesh.
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)
    # compute the signed distance from point cloud to mesh.
    sdf = scene.compute_signed_distance(cloud.point.positions)
    return sdf.numpy()


# Defining directories for cloud cluster & meshes file.
mesh_file_path = "project/786901_2a_20181115_Mikro-Detail-mesh/cluster_90_mesh.ply"# next year
point_cloud_file_path = "project/786901_2a_20180718_Mikro-Detail/100/cluster_90.ply"# first year

# For reading two files corresponding to point cloud and mesh for different years in the same region,
# The distance calculation is analysed for rockfall detection.
mesh = o3d.io.read_triangle_mesh(mesh_file_path)
point_cloud = o3d.io.read_point_cloud(point_cloud_file_path)

# Converting the mesh and point cloud into Open3D's Tensor types.
mesh_tensor = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
point_cloud_tensor = o3d.t.geometry.PointCloud.from_legacy(point_cloud)

# Calculating the distances between points and surface of mesh.
distances = cloud_mesh_to_distance(mesh_tensor, point_cloud_tensor)

# Converting the obtained value to absolute distance values.
abs_distances = np.abs(distances)

# Printing the number of points in the input point cloud for reference.
num_points = len(point_cloud.points)
print(f"Number of points in the point cloud: {num_points}")

# Printing distances from each point in the point cloud to the mesh as output.
for i, distance in enumerate(abs_distances):
    print(f"Point {i} distance: {distance}")

# Defining the threshold value above which points should be separated.
threshold = 0.02 # vary the values and run the code to find out the effect of threshold on number of points filtered.

# Find indices of points with distances higher than the provided threshold.
indices_above_threshold = np.where(abs_distances > threshold)[0]
points_above_threshold = len(indices_above_threshold)

# Printing the number of such points.
print(f"Number of points with distance greater than {threshold}: {points_above_threshold}")

# Creating a new point cloud with only the points above the threshold.
filtered_points = np.asarray(point_cloud.points)[indices_above_threshold]
filtered_colors = np.asarray(point_cloud.colors)[indices_above_threshold] if point_cloud.colors else None

filtered_point_cloud = o3d.geometry.PointCloud() # creating new point cloud.
filtered_point_cloud.points = o3d.utility.Vector3dVector(filtered_points) # Assigning the filtered points to new point cloud.
if filtered_colors is not None: # Giving colour if the points exist.
    filtered_point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)

# Specifying the path to which the filtered point cloud has to be saved.
filtered_point_cloud_file_path = "project/cloudtomeshdistanceanalysis/0.02/20180718-20181115/filtered_point_cloud_clu90.ply"

# Saving the filtered point cloud to the specified path.
o3d.io.write_point_cloud(filtered_point_cloud_file_path, filtered_point_cloud)

# Printing the cloud save to specified path message.
print(f"Filtered point cloud saved to {filtered_point_cloud_file_path}")

# Visualizing the original and filtered point cloud with color mapped to distances.
colors = plt.get_cmap("jet")((abs_distances - np.min(abs_distances)) / (np.max(abs_distances) - np.min(abs_distances)))
point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])

# Adding grey color to the mesh for better visualization.
mesh.paint_uniform_color([0.7, 0.7, 0.7])

# Visualizing the original point cloud and mesh.
o3d.visualization.draw_geometries(
    [point_cloud, mesh],
    mesh_show_wireframe=True,
    mesh_show_back_face=True
)

# Visualizing the final filtered point cloud.
o3d.visualization.draw_geometries([filtered_point_cloud])
