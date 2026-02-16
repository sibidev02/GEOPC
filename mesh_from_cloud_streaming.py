
# We have a huge point cloud with millions of points. To perform rockfall analysis, distance between the points
# in the point cloud of subsequent years is calculated. For applying meshing method, the clusters of point cloud
# should be converted in to meshes. This code takes the directory where clusters created from huge point cloud
# as input and gives a directory where the meshes for each cluster is saved as .ply file as output.


#----------------------------------------------------------------------------------------------------------------------
# References:
# Open3D: A Modern Library for 3D Data Processing https://arxiv.org/pdf/1801.09847
# https://stackoverflow.com/questions/62938546/how-to-draw-bounding-boxes-and-update-them-real-time-in-python
# Quadric Simplification Method: https://dl.acm.org/doi/pdf/10.1145/258734.258849
# Mesh clustering and connected triangle analysis: https://doi.org/10.1201/b10688
# Chatgpt for debugging and fixing issues
#-----------------------------------------------------------------------------------------------------------------------


# importing necessary libraries...
import open3d as o3d # For 3D data processing,including point clouds and meshes.
import numpy as np
import os

# Defining a function to process a point cloud file and save a mesh version of it to the provided output directory.
def cloud_to_mesh_streaming(pcd_file_path, output_path):
    # For ensuring the output directory provided exists and if not creating it
    os.makedirs(output_path, exist_ok=True)

    # For Loading the point cloud from the provided .ply file.
    pcd = o3d.io.read_point_cloud(pcd_file_path)

    # Down-sampling the point cloud
    # Reducing the number of points using a voxel grid filter for making the processing faster & efficient.
    # voxel size of 0.02 is chosen after trial and error (reduce the voxel size for retaining more details).
    # For further details please refer the report.
    pcd = pcd.voxel_down_sample(voxel_size=0.02)

    # Computing the bounding box of the original point cloud
    # Computing the smallest box aligned to the axes that contains the entire point cloud.
    # The minimum and maximum bounds are extracted.
    bounding_box = pcd.get_axis_aligned_bounding_box()
    min_bound = np.array(bounding_box.min_bound)
    max_bound = np.array(bounding_box.max_bound)

    # Expanding the bounding box slightly.
    # For ensuring all points are captured.
    # Expansion factor 0.1 is chosen after several trials.
    expansion_factor = 0.1
    min_bound -= expansion_factor
    max_bound += expansion_factor

    # Define and crop the point cloud using the expanded bounding box
    # Creating a new expanded bounding box and cropping the point cloud to fit within it.
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
    cropped_pcd = pcd.crop(bbox)

    # For checking if the cropped point cloud is empty.
    # If empty showing a message to adjust the bounding box.
    if len(cropped_pcd.points) == 0:
        raise ValueError("The cropped point cloud is empty, adjust the bounding box.")

    # Estimating surface normals on the cropped point cloud for mesh reconstruction.
    cropped_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # For checking if normals are correctly estimated.
    # If not giving message.
    if not cropped_pcd.has_normals():
        raise ValueError("Normals not estimated correctly.")

    # Poisson surface reconstruction
    # Generating a triangle mesh from the point cloud using the normals.
    # Depth of octree controls the resolution of the output mesh.
    # depth=9 is chosen after trials. For more details please refer the report.
    # Increase depth for finer mesh with more details.
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cropped_pcd, depth=9)

    # For removing low-density vertices.
    # Vertices with the least density of 5% are removed.
    densities = np.asarray(densities)
    vertices_to_remove = densities < np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # Calculating triangle areas.
    # calculated using the cross-product formula for triangle area.
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    triangle_areas = np.zeros(len(triangles))

    for i, triangle in enumerate(triangles):
        p1 = vertices[triangle[0]]
        p2 = vertices[triangle[1]]
        p3 = vertices[triangle[2]]
        triangle_areas[i] = np.linalg.norm(np.cross(p2 - p1, p3 - p1)) / 2

    # Remove small area triangles.
    # For cleaning the mesh triangles with least 5% area are removed.
    area_threshold = np.quantile(triangle_areas, 0.05)
    triangles_to_remove = triangle_areas < area_threshold
    mesh.remove_triangles_by_mask(triangles_to_remove)
    mesh.remove_unreferenced_vertices()

    # Cluster connected triangles and remove small clusters
    # Cluster with less than 200 triangles removed. Reduce cluster threshold for retaining more details.
    triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_threshold = 100
    clusters_to_keep = np.where(cluster_n_triangles >= cluster_threshold)[0]
    triangles_to_keep_mask = np.isin(triangle_clusters, clusters_to_keep)

    triangles = np.asarray(mesh.triangles)
    vertices_to_keep = np.unique(triangles[triangles_to_keep_mask].flatten())
    vertices_to_keep_mask = np.zeros(len(mesh.vertices), dtype=bool)
    vertices_to_keep_mask[vertices_to_keep] = True
    mesh.remove_vertices_by_mask(~vertices_to_keep_mask)

    # Simplifying the mesh.
    # Optional for easy processing.
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=int(len(mesh.triangles) * 0.25))

    # Saving the mesh as .ply in the specified output directory.
    mesh_filename = os.path.join(output_path, os.path.basename(pcd_file_path).replace('.ply', '_mesh.ply'))
    o3d.io.write_triangle_mesh(mesh_filename, mesh)

    # Printing a message with output file path if the mesh is saved.
    print(f"Mesh saved successfully in {output_path}.")

# Setting the input directory with clusters and output directory to save meshes here.
input_dir = "test/786901_2a_20181115_Mikro-Detail/100"
output_dir = "test/786901_2a_20181115_Mikro-Detail-mesh"

# Loop through each cluster file and process for creating meshes for all the clusters simultaneously.
# Streaming through clusters.
for cluster_file in os.listdir(input_dir):
    if cluster_file.endswith(".ply"):
        pcd_file_path = os.path.join(input_dir, cluster_file)
        cloud_to_mesh_streaming(pcd_file_path, output_dir)
