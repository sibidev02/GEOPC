# In this program we are making clusters using the Json file we made using the bounding boxes program. Each tile is
# saved a seperate file in output directory.
# _________________________________________________________________________________________________________
# references:
# http://www.open3d.org/docs/latest/index.html
# https://www.open3d.org/docs/latest/python_api/open3d.geometry.OrientedBoundingBox.html
# http://www.open3d.org/docs/latest/tutorial/Basic/visualization.html
# https://numpy.org/doc/ and  https://numpy.org/doc/stable/user/basics.indexing.html
# ChatGPT for solving error and issues in code.
# ___________________________________________________________________________________________________________
# Below libraries are imported to work with functions associated with them.
import json
import numpy as np
import open3d as o3d
import os

# Load the second point cloud data from a PLY file
ply_file_path = 'C:/Users/91997/PycharmProjects/pythonProject/786901_2a_20181115_Mikro-Detail.ply'
pcd = o3d.io.read_point_cloud(ply_file_path)
coordinates = np.asarray(pcd.points)  # Convert the points to a numpy array to be used in clustering.

# Load bounding boxes from the JSON file using json load function
with open('new_bounding_boxes.json', 'r') as f:
    bounding_boxes = json.load(f)

# Create AxisAlignedBoundingBox objects from loaded bounding boxes
grid_boxes = []
points_in_grids = []

# Directory to save the grid cells
output_dir = 'nextfile_gridcells2018'
os.makedirs(output_dir, exist_ok=True)

# The for loop iterates through each bounding box to keep the track of bounding box being processed.
# the minimum and maximum coordinates that define the bounding box are used. These are read from the JSON file and
# converted into NumPy arrays
for i, boundbox in enumerate(bounding_boxes):
    min_coords = np.array(boundbox['min_coords'])
    max_coords = np.array(boundbox['max_coords'])

# AxisAlignedBoundingBox function is used to create grid boxes which uses min and mix coords as inputs.
    grid_box = o3d.geometry.AxisAlignedBoundingBox(min_coords, max_coords)
    grid_box.color = (0, 0, 0)  # add color
    grid_boxes.append(grid_box)  # append the grid box

    # Count points within the grid box. for this we use boolean array to check that coordinates lie with in the
    # bounding box. The condition ensures that  points satisfy the bounding box constraints along all three axes.
    maskarr = np.all((coordinates >= min_coords) & (coordinates <= max_coords), axis=1)
    cluster_points = coordinates[maskarr]  # The result of masking operation is stored
    no_points_in_tile = len(cluster_points)  # count the no. of points
    points_in_grids.append(no_points_in_tile)  # append the points

    # Save cluster points to a PLY file. The filtered cluster_points are assigned to this  point cloud
    # converting them back into an Open3D format.
    cluster_pcd = o3d.geometry.PointCloud()
    cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
    cluster_ply_filename = os.path.join(output_dir, f'cluster_{i}.ply')  # saved a new files.
    o3d.io.write_point_cloud(cluster_ply_filename, cluster_pcd)  # file is saved in the output directory

    # Print the number of points in this tile
    print(f"Cluster {i}: Points in Tile: {no_points_in_tile}")

# Visualize the point cloud and bounding boxes to see how the bounding boxes segment the point cloud.
pcd.points = o3d.utility.Vector3dVector(coordinates)
o3d.visualization.draw_geometries([pcd] + grid_boxes)

# Check and print density statistics. It compares the total number of points in the original point cloud to the sum of
# points in all the grids.
total_points = coordinates.shape[0]
total_points_in_grids = np.sum(points_in_grids)
print(f"Total Points: {total_points}, Total Points in Grids: {total_points_in_grids}")

# Check if any points are lost during tiling.
if total_points == total_points_in_grids:
    print("No points lost during tiling.")
else:
    print("Points lost during tiling.")
