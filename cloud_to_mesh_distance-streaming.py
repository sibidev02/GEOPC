
# The code calculates the distances from each point in the point cloud to the mesh in provided directories.
# mesh is the 3D mesh object to which distances are computed and cloud is the 3D point cloud from which
# distances are calculated. The code returns a numpy array of distances from each point in the point cloud
# to the mesh.


#----------------------------------------------------------------------------------------------------------------------
# References:
# Open3D: A Modern Library for 3D Data Processing https://arxiv.org/pdf/1801.09847
# https://github.com/isl-org/Open3D
# Generating 3D Meshes with Python: https://orbi.uliege.be/bitstream/2268/254933/1/TDS_generate_3D_meshes_with_python.pdf
# 3D Point Cloud Reconstruction: https://ar5iv.labs.arxiv.org/html/2112.12907
# Chatgpt for debugging and fixing issues
#-----------------------------------------------------------------------------------------------------------------------


import open3d as o3d  # Importing Open3D for 3D data processing.
import numpy as np    # Importing NumPy for numerical computations.
import os             # Importing os for directory management and file handling.

# Defining the function to calculate cloud to mesh distance
def cloud_to_mesh_distance(mesh: o3d.t.geometry.TriangleMesh, cloud: o3d.t.geometry.PointCloud) -> np.ndarray:
    # For computing distances by "casting rays" from the point cloud towards the mesh.
    # Raycasting is a technique used to compute the intersection of rays (straight lines) with objects in a 3D scene.
    # Here Raycasting is used to add the mesh and compute the distance between cloud and mesh.
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)
    # compute the signed distance from point cloud to mesh.
    sdf = scene.compute_signed_distance(cloud.point.positions)
    return sdf.numpy()

# Defining directories for clusters, meshes & filtered points.
clusters_dir = "project/786901_2a_20180718_Mikro-Detail/100"  # Directory path where point cloud clusters are stored.
meshes_dir = "project/786901_2a_20181115_Mikro-Detail-mesh"    # Directory path where corresponding meshes are stored.
output_dir = "project/786901_2a_20181115_Mikro-Detail-filteredpoints"  # Output directory to store filtered point clouds.

# For ensuring the output directory provided exists and if not creating it
os.makedirs(output_dir, exist_ok=True)

# For initializing a counter to keep track of the total number of points filtered across all clusters.
total_filtered_points = 0

# Loop through each cluster and corresponding mesh file starting from 0 to 100.
for i in range(101):  # We can adjust the range if number of cluster in the directory is different.
    cluster_file = f"cluster_{i}.ply"
    mesh_file = f"cluster_{i}_mesh.ply"

    # Generating full paths for the cluster and mesh files.
    cluster_path = os.path.join(clusters_dir, cluster_file)
    mesh_path = os.path.join(meshes_dir, mesh_file)

    # Checking if both the cluster and mesh files exist.
    if os.path.exists(cluster_path) and os.path.exists(mesh_path):
        # Loading the mesh and point cloud.
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        point_cloud = o3d.io.read_point_cloud(cluster_path)

        # Converting the mesh and point cloud into Open3D's Tensor types.
        mesh_tensor = o3d.t.geometry.TriangleMesh.from_legacy(mesh) # Loading the mesh file.
        point_cloud_tensor = o3d.t.geometry.PointCloud.from_legacy(point_cloud) # Loading the point cloud file.

        # Calculate the signed distances from the point cloud to the mesh.
        distances = cloud_to_mesh_distance(mesh_tensor, point_cloud_tensor)

        # Convert to absolute distances as magnitude is not necessary.
        abs_distances = np.abs(distances)

        # Defining the threshold value.
        # The points having a distance higher that this threshold will be rockfall and getting filtered.
        threshold = 0.03 # Tried the values 0.01,0.02,0.03,0.05,0.09.

        # Finding indices of points with distances greater than the threshold.
        indices_above_threshold = np.where(abs_distances > threshold)[0]

        # Create a new point cloud with only the points above the threshold.
        filtered_points = np.asarray(point_cloud.points)[indices_above_threshold]
        # If the point cloud has color information, extract the colors.
        filtered_colors = np.asarray(point_cloud.colors)[indices_above_threshold] if point_cloud.colors else None

        # Creating a new Open3D point cloud object for the filtered points.
        filtered_point_cloud = o3d.geometry.PointCloud()
        filtered_point_cloud.points = o3d.utility.Vector3dVector(filtered_points)

        # If color data exists, set the color of the filtered points.
        if filtered_colors is not None:
            filtered_point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)

        # Save the filtered point cloud.
        filtered_point_cloud_file_path = os.path.join(output_dir, f"filtered_cluster_{i}.ply")
        o3d.io.write_point_cloud(filtered_point_cloud_file_path, filtered_point_cloud)

        # Update the total points filtered.
        total_filtered_points += len(filtered_points)

        # Printing the number of points filtered.
        print(f"Cluster {i}: {len(filtered_points)} points filtered. Saved to {filtered_point_cloud_file_path}")

    else:
        # If the cluster or mesh file doesn't exists a warning message gets printed.
        print(f"Missing cluster or mesh file for index {i}")

# Printing the total number of points filtered across all clusters.
print(f"Total number of points filtered across all clusters: {total_filtered_points}")
# Printing process complete message.
print("Processing complete.")
