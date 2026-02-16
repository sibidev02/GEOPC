# The filtered points which indicates the rockfall event across all clusters are to be stitched together
# to get the rockfall for the whole point cloud. The clusters with filtered points are streamed and
# stitched together to obtain this.

# importing necessary libraries.
import open3d as o3d # Importing Open3D for 3D data processing.
import os # Importing os for directory management and file handling.

# Defining paths for input and output directories.
clusters_dir = "project/786901_2a_20181115_Mikro-Detail-filteredpoints"  # Path for directory containing the point cloud clusters.
output_dir = "project/786901_2a_20181115_Mikro-Detail-filtered_stitched"  # Path for output directory for the combined point cloud.

# For ensuring the output directory exists, if not it will be created.
os.makedirs(output_dir, exist_ok=True)

# Initializing an empty point cloud object.
# This can be used to combine all the individual filtered point clouds from the input cluster directory.
combined_point_cloud = o3d.geometry.PointCloud()

# Variables to track the file with the highest number of points
max_points = 0 # This variable can store the highest number of points found among all individual point cloud files.
max_points_file = "" # This variable can store the filename of the point cloud with the most points.

# List all files in the clusters directory
cluster_files = os.listdir(clusters_dir)
ply_files = [f for f in cluster_files if f.endswith('.ply')] # To work with only ply files in the directory.

# For looping through each cluster file and combining them into a single point cloud file.
for ply_file in ply_files:
    cluster_path = os.path.join(clusters_dir, ply_file)

    # print(f"Loading {cluster_path}...")  # print the path being loaded.
    # For loading the point cloud.
    point_cloud = o3d.io.read_point_cloud(cluster_path)

    # Combining the point cloud.
    # This accumulates all the point clouds together.
    combined_point_cloud += point_cloud
    num_points = len(point_cloud.points)
    print(f"{ply_file} added with {num_points} points.")

    # Checking if this particular file has the most points.
    if num_points > max_points:
        max_points = num_points
        max_points_file = ply_file

# Printing the file with the highest number of points.
print(f"The file with the highest number of points is {max_points_file} with {max_points} points.")

# Saving the combined point cloud as a ply file
combined_point_cloud_file_path = os.path.join(output_dir, "combined_point_cloud.ply")
o3d.io.write_point_cloud(combined_point_cloud_file_path, combined_point_cloud)

# Printing the message showing the path where the file is saved.
print(f"Combined point cloud saved to {combined_point_cloud_file_path}")

# Visualizing the combined point cloud using open 3d.
o3d.visualization.draw_geometries([combined_point_cloud])
