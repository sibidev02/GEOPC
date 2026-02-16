
# We are using this code to combine the biggest four files. The reason is to create a bounding box that covers
# all the point clouds in various years. This will ensure that boundary points and key details are preserved while
# reducing unnecessary duplicates. This has helped to create uniform bounding box which can be applied on all
# point cloud data for making clusters.
# __________________________________________________________________________________________________________________
# References:
# https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html
# https://www.open3d.org/docs/0.9.0/tutorial/Basic/pointcloud.html
# http://www.open3d.org/docs/latest/python_api/open3d.geometry.PointCloud.html
# ChatGPT for solving error and issues in code.
# __________________________________________________________________________________________________________________

import open3d as o3d
# List of .ply file paths to be merged
ply_file_paths = [
    'C:/Users/91997/PycharmProjects/pythonProject/786901_2a_20201118_Mikro-Detail.ply',
    'C:/Users/91997/PycharmProjects/pythonProject/786901_2a_20181115_Mikro-Detail.ply',
    'C:/Users/91997/PycharmProjects/pythonProject/786901_2a_20200505_Mikro-Detail.ply',
    'C:/Users/91997/PycharmProjects/pythonProject/786901_2a_20190322_Mikro-Detail.ply'

]
# Create an empty PointCloud object for the combined file. This will store and combine the points from individual files.
combined_pcd = o3d.geometry.PointCloud()

# Initialize variables to track total points before and after combining and check if points are lost or not.
total_points_individual_files = 0

# for loop to Load each .ply file and add to the combined PointCloud
for ply_file in ply_file_paths:
    pcd = o3d.io.read_point_cloud(ply_file)  # Read point cloud from the .ply file
    numof_points = len(pcd.points)  # Count the number of points in the individual file
    total_points_individual_files += numof_points  # keep the track total number of points across all files.
    print(f"{ply_file} contains {numof_points} points.") # print the no of points in each ply file.

    combined_pcd += pcd  # merges the point cloud from the current file into combined file.

# remove duplicate points using down sampling. The method voxel_down_sample() reduces the number of points by grouping
# nearby points into small "voxels" by representing as a single point. smaller voxel size preserves the details.

    combined_pcd = combined_pcd.voxel_down_sample(voxel_size=0.001)

# after removing duplicates, save the merged point cloud to a new .ply file.
output_ply_file = ('main_combined_file_point_cloud.ply')
o3d.io.write_point_cloud(output_ply_file, combined_pcd)

# Count the total no of points in the combined point cloud.
total_no_points_combined = len(combined_pcd.points)

# Below code is written for verification of total number of points. It checks if the point count before and after
# merging matches. If points are lost (due to downsampling or file issues), it issues a warning.

print(f"\nTotal points in individual files: {total_points_individual_files}")
print(f"Total points in the combined point cloud: {total_no_points_combined}")

# Check if the point counts matches or not.
if total_points_individual_files == total_no_points_combined:
    print("Success: All points from the individual files are present in the combined file.")
else:
    print(f"Warning: {total_points_individual_files - total_no_points_combined} points are missing in the combined file.")
