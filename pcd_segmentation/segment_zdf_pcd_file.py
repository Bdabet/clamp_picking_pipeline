import open3d as o3d
import numpy as np
import os
import zivid
import cv2
import sys

# Determine the root path 
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
from config import root_path


big_clamp_pcd_path = os.path.join(root_path, "files", "clamps_pcds", "clamps_pcd","big_clamp_aligned.pcd")




def load_zdf_point_cloud(zdf_path, app = None):
    if app is None:
        app = zivid.Application()
    frame = zivid.Frame(zdf_path)
    point_cloud = frame.point_cloud()
    xyz = point_cloud.copy_data("xyz")
    rgba = point_cloud.copy_data("rgba")
    height = frame.point_cloud().height
    width = frame.point_cloud().width
    return xyz, rgba, height, width

def apply_transformation(xyz, transformation_matrix):
    points = xyz.reshape(-1, 3)
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    #points_homogeneous = points_homogeneous * 0.001 # convert to millimeters
    transformed_points_homogeneous = points_homogeneous.dot(transformation_matrix.T)
    transformed_points = transformed_points_homogeneous[:, :3]
    return transformed_points.reshape(xyz.shape)

def visualize_point_cloud(xyz, rgba):
    rgb = rgba[:, :, :3] / 255.0
    points = xyz.reshape(-1, 3)
    colors = rgb.reshape(-1, 3)
    valid_indices = ~np.isnan(points).any(axis=1) & (points[:, 2] != 0)
    points = points[valid_indices]
    colors = colors[valid_indices]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name="Full Point Cloud")

def segment_pcd_masks(zdf_path, masks_dir, save_dir, transformation_matrix=None, app = None):
    
    xyz, rgba, height, width = load_zdf_point_cloud(zdf_path, app)
    if transformation_matrix is not None:
        xyz = apply_transformation(xyz, transformation_matrix)
    # visualize_point_cloud(xyz, rgba)
    rgb = rgba[:, :, :3] / 255.0
    depth = xyz[:, :, 2]

    for mask_filename in os.listdir(masks_dir):
        mask_path = os.path.join(masks_dir, mask_filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        mask_points = []
        mask_colors = []
        for v in range(height):
            for u in range(width):
                if mask[v, u] > 0:
                    z = depth[v, u]
                    if np.isnan(z) or z == 0:
                        continue
                    x = xyz[v, u, 0]
                    y = xyz[v, u, 1]
                    z = xyz[v, u, 2]
                    mask_points.append([x, y, z])
                    mask_colors.append(rgb[v, u])

        mask_points = np.array(mask_points)
        mask_colors = np.array(mask_colors)

        if len(mask_points) == 0:
            print(f"No valid points found in mask {mask_filename}")
            continue

        mask_pcd = o3d.geometry.PointCloud()
        mask_pcd.points = o3d.utility.Vector3dVector(mask_points)
        mask_pcd.colors = o3d.utility.Vector3dVector(mask_colors)

        if is_mask_valid(mask_pcd):
            mask_pcd_path = os.path.join(save_dir, mask_filename.replace(".png", ".pcd"))
            o3d.io.write_point_cloud(mask_pcd_path, mask_pcd)
            #o3d.visualization.draw_geometries([mask_pcd], window_name=mask_filename)
            print(f"Saved segmented point cloud to {mask_pcd_path}")
        else:
            print(f"{mask_filename} is not valid")
       

def is_mask_valid(segmented_pcd):

    # Perform statistical outlier removal
    cl, ind = segmented_pcd.remove_statistical_outlier(nb_neighbors=250, std_ratio=2.0)

    # Select inlier points 
    segmented_pcd_inliers = segmented_pcd.select_by_index(ind)

    segmented_pcd_array = np.asarray(segmented_pcd_inliers.points)
    
    # find bodunding lengths of segmented pcd
    segmented_pcd_ranges = np.max(segmented_pcd_array, axis = 0) - np.min(segmented_pcd_array, axis = 0)
    # print(segmented_pcd_ranges)

    # get bounding lengths of maximum possible pcd
    big_clamp_pcd = o3d.io.read_point_cloud(big_clamp_pcd_path)
    big_clamp_pcd_array = np.asarray(big_clamp_pcd.points)
    max_ranges = np.max(big_clamp_pcd_array, axis = 0) - np.min(big_clamp_pcd_array, axis = 0)

    range_difference = segmented_pcd_ranges - max_ranges

    if (np.max(range_difference/max_ranges) > 3):
        return False
    else:
        return True

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],  window_name="inlier/outlier")

def create_pcd_excluding_masks(zdf_path, masks_dir, clamps_data, save_path= None,  transformation_matrix=None, app=None):
    # Load the ZDF point cloud
    xyz, rgba, height, width = load_zdf_point_cloud(zdf_path, app)
    
    # Prepare the RGB and depth data
    rgb = rgba[:, :, :3] / 255.0
    depth = xyz[:, :, 2]
    
    # Initialize a mask to track points to exclude
    exclusion_mask = np.zeros((height, width), dtype=bool)
    
    # Process each mask and update the exclusion_mask
    # Extract all relevant mask names from the dictionary
    relevant_masks = []
    
    for clamp_data in clamps_data.values():

        relevant_masks.append(clamp_data["mask name"][:-4])
    
    # print("recorded relevant masks are",relevant_masks)
    for mask_filename in os.listdir(masks_dir):
        if mask_filename[:-4] in relevant_masks:
            mask_path = os.path.join(masks_dir, mask_filename)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            exclusion_mask = exclusion_mask | (mask > 0)  # Combine masks
    
    # Collect points that are NOT covered by any mask
    remaining_points = []
    remaining_colors = []
    for v in range(height):
        for u in range(width):
            if not exclusion_mask[v, u]:  # Check if the point is not excluded
                z = depth[v, u]
                if np.isnan(z) or z == 0:
                    continue
                x = xyz[v, u, 0]
                y = xyz[v, u, 1]
                z = xyz[v, u, 2]
                remaining_points.append([x, y, z])
                remaining_colors.append(rgb[v, u])
    
    # Convert remaining points and colors to numpy arrays
    remaining_points = np.array(remaining_points)
    remaining_colors = np.array(remaining_colors)
    
    # Create a point cloud
    if len(remaining_points) == 0:
        print("No valid points found after excluding masks.")
        return
    
    remaining_pcd = o3d.geometry.PointCloud()
    remaining_pcd.points = o3d.utility.Vector3dVector(remaining_points)
    remaining_pcd.colors = o3d.utility.Vector3dVector(remaining_colors)
    
    remaining_pcd.points = o3d.utility.Vector3dVector(np.asarray(remaining_pcd.points) * 0.001)
    remaining_pcd.transform(transformation_matrix)
    
    # Save the resulting point cloud
    if save_path:
        o3d.io.write_point_cloud(save_path, remaining_pcd)
    

    remaining_pcd = remaining_pcd.voxel_down_sample(voxel_size=0.001)
    cl, ind =  remaining_pcd.remove_statistical_outlier(nb_neighbors=70,std_ratio=2)
    display_inlier_outlier(remaining_pcd, ind)
    segmented_pcd_inliers = remaining_pcd.select_by_index(ind)
    
    o3d.visualization.draw_geometries([remaining_pcd])
  

    return segmented_pcd_inliers

def segment_single_pcd_mask(zdf_path, masks_dir, mask_filename, save_dir, transformation_matrix=None, app = None):
    xyz, rgba, height, width = load_zdf_point_cloud(zdf_path, app)
    if transformation_matrix is not None:
        xyz = apply_transformation(xyz, transformation_matrix)
    # visualize_point_cloud(xyz, rgba)
    rgb = rgba[:, :, :3] / 255.0
    depth = xyz[:, :, 2]

    
    mask_path = os.path.join(masks_dir, mask_filename)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    mask_points = []
    mask_colors = []
    for v in range(height):
        for u in range(width):
            if mask[v, u] > 0:
                z = depth[v, u]
                if np.isnan(z) or z == 0:
                    continue
                x = xyz[v, u, 0]
                y = xyz[v, u, 1]
                z = xyz[v, u, 2]
                mask_points.append([x, y, z])
                mask_colors.append(rgb[v, u])

    mask_points = np.array(mask_points)
    mask_colors = np.array(mask_colors)

    

    mask_pcd = o3d.geometry.PointCloud()
    mask_pcd.points = o3d.utility.Vector3dVector(mask_points)
    mask_pcd.colors = o3d.utility.Vector3dVector(mask_colors)

    if is_mask_valid(mask_pcd):
        mask_pcd_path = os.path.join(save_dir, mask_filename.replace(".png", ".pcd"))
        o3d.io.write_point_cloud(mask_pcd_path, mask_pcd)
        #o3d.visualization.draw_geometries([mask_pcd], window_name=mask_filename)
        print(f"Saved segmented point cloud to {mask_pcd_path}")
    else:
        print(f"{mask_filename} is not valid")

if __name__ == "__main__": 
    
    #testing
    x = 0


    