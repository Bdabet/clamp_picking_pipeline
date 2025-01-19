import json
import open3d as o3d
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import euclidean
from scipy.spatial.transform import Rotation as R
import os
import random
import sys

# Determine the root path 
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from config import root_path

from  scene_representation.representation_creation import clamps, scene_pcd


def find_nearby_clamps(input_clamp, twodim = False, grasping = False):

    relevant_clamps = []

    #get bounding box points of input clamp
    if twodim == True:
        bounding_box = np.asarray(input_clamp.bounding_box_pcd.points)[:,:2]
    elif grasping == True:
        bounding_box = np.asarray(input_clamp.bounding_box_extended_pcd.points)
    else:
        bounding_box = np.asarray(input_clamp.bounding_box_pcd.points)
        

   

    #check clamps bounding boxes to find clmaps in the vicinity of the current clamp
    for clamp in clamps:

        if clamp.clamp_number != input_clamp.clamp_number:

            for point in np.asarray(clamp.clamp_curve_pcd.points):
                
                if twodim == True:

                    point = point[:2]
                    

                if Delaunay(bounding_box).find_simplex(point) >= 0:

                    
                    #print(f"clamp {input_clamp.clamp_number} is in viscinity of clamp {clamp.clamp_number}")
                    relevant_clamps.append(clamp)
                    break #dont need to check other points if one point is found to be in checked volume"""
    
    return relevant_clamps
    
     
def is_clamp_entangled(input_clamp):

    clamp_is_entangled = False

    entangled_clamps = []

    nearby_clamps = find_nearby_clamps(input_clamp)

    bounding_volume = np.asarray(input_clamp.bounding_volume_pcd.points)

    for clamp in nearby_clamps:

        if clamp.clamp_number != input_clamp.clamp_number:

            for point in np.asarray(clamp.clamp_curve_pcd.points):
                    

                if Delaunay(bounding_volume).find_simplex(point) >= 0:

                    entangled_clamps.append(clamp)
                    clamp_is_entangled = True
                    print(f"clamp {input_clamp.clamp_number} is entangled with clamp {clamp.clamp_number}")
                    break #dont need to check other points if one point is found to be in checked volume

    return clamp_is_entangled, entangled_clamps


def is_clamp_entangled_visualized(input_clamp, pcd):
    clamp_is_entangled = False
    entangled_clamps = []
    
    
    
    nearby_clamps = find_nearby_clamps(input_clamp)
    bounding_volume = np.asarray(input_clamp.bounding_volume_pcd.points)

    # Create a list to store red points to be added
    red_points = []

    for clamp in nearby_clamps:
        if clamp.clamp_number != input_clamp.clamp_number:
            for point in np.asarray(clamp.clamp_curve_pcd.points):
                if Delaunay(bounding_volume).find_simplex(point) >= 0:
                    red_points.append(point)  # Add the point inside bounding box to red points
                    entangled_clamps.append(clamp)
                    clamp_is_entangled = True
                    print(f"Clamp {input_clamp.clamp_number} is entangled with clamp {clamp.clamp_number}")
                    # break  # Don't need to check other points if one point is found to be in the bounding volume

    # Add red points to the PCD
    if red_points:
        red_pcd = o3d.geometry.PointCloud()
        red_pcd.points = o3d.utility.Vector3dVector(np.array(red_points))
        red_pcd.paint_uniform_color([1, 0, 0])  # Color red

        # Combine original PCD with red points
        pcd += red_pcd

        # Save the updated PCD file
        # updated_pcd_file = pcd_file.replace('.pcd', '_updated.pcd')
        # o3d.io.write_point_cloud(updated_pcd_file, pcd)
        # print(f"Updated PCD file saved to {updated_pcd_file}")

    return clamp_is_entangled, entangled_clamps, pcd


def is_clamp_occluded(input_clamp):

    clamp_is_occluded = False

    occluding_clamps = []

    nearby_clamps = find_nearby_clamps(input_clamp, twodim=True)

                
                    
    #extract x and y coordinates of input clamp curve
    input_clamp_curve_coord = np.asarray(input_clamp.clamp_curve_pcd.points)[:,:2]
    

    for clamp in nearby_clamps:

        if clamp.clamp_number != input_clamp.clamp_number:
            
            clamp_curve_coord = np.asarray(clamp.clamp_curve_pcd.points)[:,:2]

            # Define tolerance
            tolerance = 1

            shared_entries = [(coord1, coord2) for coord1 in input_clamp_curve_coord for coord2 in clamp_curve_coord 
                                if (np.abs(np.subtract(coord1, coord2)) < np.array([tolerance, tolerance])).all()]
            
            if np.asarray(shared_entries).size > 0:
                
                input_clamp_height = np.mean(np.asarray(input_clamp.bounding_box_pcd.points)[:,2])
                clamp_height = np.mean(np.asarray(clamp.bounding_box_pcd.points)[:,2])
                if clamp_height > input_clamp_height:
                    occluding_clamps.append(clamp)
                    clamp_is_occluded = True

    return clamp_is_occluded, occluding_clamps           


def middlemost_true_region(regions):
    
    # Get the total number of regions
    total_regions = len(regions)
    
    # Find the middle index of the entire dictionary
    middle_index = total_regions // 2
    
    # Convert the keys to a list
    region_keys = list(regions.keys())
    
    # Find the indices of all True regions
    true_indices = [i for i, key in enumerate(region_keys) if regions[key]]
    
    if not true_indices:
        return None  # Return None if no True regions exist
    
    # Find the True region closest to the middle index
    closest_index = min(true_indices, key=lambda x: abs(x - middle_index))
    
    return region_keys[closest_index], closest_index


def find_accessible_points(input_clamp, search_radius=0.014, visualize_nearby_points = False):
    
    accessible_points = []
    accessible_points_idxs = []

    # Find nearby clamps
    nearby_clamps = find_nearby_clamps(input_clamp, grasping=True)
    
    if nearby_clamps:
        # Iterate through grasping points and angles
        for idx, ((point_name, point), angle) in enumerate(zip(input_clamp.grasping_points.items(), input_clamp.grasping_points_angles.values())):
            point_accessible = True
            point = np.array(point, dtype=np.float64)

            for clamp in nearby_clamps:
                # Build KD-Tree for the current clamp's PCD
                kd_tree = o3d.geometry.KDTreeFlann(clamp.clamp_pcd)
                
                # Search for points within the radius
                [_, idxs, _] = kd_tree.search_radius_vector_3d(point, search_radius)

                if visualize_nearby_points:
                    np.asarray(clamp.clamp_pcd.colors)[idxs[1:], :] = [0, 1, 0]
                    print("Visualize the point cloud.")
                    o3d.visualization.draw_geometries([clamp.clamp_pcd])
                
                if len(idxs) > 0:
                    # Point is inaccessible due to proximity to nearby points
                    point_accessible = False
                    print(f"point {point_name} is inaccessible.")
                    break
            
            # If point remains accessible after checking all clamps
            if point_accessible:
                accessible_points.append((idx, point, angle))
                accessible_points_idxs.append(idx)
    else:
        # If no nearby clamps, all points are accessible
        for idx, (point, angle) in enumerate(zip(input_clamp.grasping_points.values(), input_clamp.grasping_points_angles.values())):
            accessible_points.append((idx, np.array(point, dtype=np.float64), angle))
            accessible_points_idxs.append(idx)
    
    return accessible_points, accessible_points_idxs


def find_grasping_pose_points(input_clamp, accessible_points, input_pcd=None, visualize=False):
    """
    Choose the most suitable grasping point from the accessible points and optionally highlight it in the PCD.
    
    Parameters:
        input_clamp: The clamp being evaluated.
        accessible_points: List of accessible points with their indices and angles.
        input_pcd: Optional; The point cloud of the scene containing clamps.
        visualize: Optional; Whether to visualize the point cloud with the highlighted grasping point.
        
    Returns:
        tuple: Chosen grasping point's position and angle.
    """
    if not accessible_points:
        print("No accessible grasping points found.")
        return None, None

    # Choose the middlemost accessible point
    if input_clamp.clamp_orientation == "h":
        mid_idx = len(accessible_points) // 2
        chosen_point_info = accessible_points[mid_idx]
    elif input_clamp.clamp_orientation == "v":
        chosen_point_info = max(accessible_points, key=lambda p: p[1][2])
    
    idx, position, angle = chosen_point_info
    print(f"Chosen grasping point is at index {idx} with position {position} and angle {angle}")

    # Highlight the chosen grasping point if visualization is enabled and input_pcd is provided
    if visualize and input_pcd is not None:
        grasping_point_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.007)
        grasping_point_sphere.translate(position)  # Position the sphere at the grasping point
        grasping_point_sphere.paint_uniform_color([0, 1, 0])  # Red color for the sphere

        # Visualize the scene with the highlighted point
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(input_pcd)  # Add the original point cloud
        vis.add_geometry(grasping_point_sphere)  # Add the sphere
        vis.run()
        vis.destroy_window()
    elif visualize:
        print("Visualization skipped because no input_pcd was provided.")

    return position, angle


def find_accessible_points_with_walls(input_clamp, accessible_points_idxs, search_radius=0.014, distance_threshold = 0.004, input_pcd=None, visualize_nearby_points=False, scene_pcd = None):
    """
    Finds accessible grasping points for a clamp, considering a given point cloud (PCD).

    Parameters:
    - input_clamp: The clamp being checked for accessible points.
    - search_radius: Radius to search for nearby points in the input PCD.
    - distance_threshold: Maximum Euclidean distance to mark a point as inaccessible.
    - input_pcd: The Open3D PointCloud object to search for nearby points.
    - visualize_nearby_points: If True, highlights nearby points for visualization.
    
    Returns:
    - List of accessible points as tuples (index, point, angle).
    """

    accessible_points = []
    
    if input_pcd is None:
        raise ValueError("An input PCD must be provided.")


    # Build KD-Tree for the input PCD
    kd_tree = o3d.geometry.KDTreeFlann(input_pcd)

    # Iterate through grasping points and angles
    for idx, ((point_name, point), angle) in enumerate(zip(input_clamp.grasping_points.items(), input_clamp.grasping_points_angles.values())):

        
        if idx in accessible_points_idxs:
            
            point_accessible = True 
            point = np.array(point, dtype=np.float64)

            # Search for points within the radius
            [_, idxs, distances] = kd_tree.search_radius_vector_3d(point, search_radius)

            if visualize_nearby_points:
                # Highlight nearby points in red
                # np.asarray(input_pcd.colors)[idxs[1:], :] = [1, 0, 0]
                print(f"Visualizing nearby points for {point_name}.")
                if scene_pcd is None:
                    o3d.visualization.draw_geometries([input_pcd])
                else:
                    o3d.visualization.draw_geometries([input_pcd, scene_pcd])

            color_idx = []
            for i in idxs[1:]:
                  
                nearby_point = np.asarray(input_pcd.points)[i]
                # distance = euclidean(point, nearby_point)
                
                z_distance = abs(point[2] - nearby_point[2])
            
                if z_distance > distance_threshold and nearby_point[2] < point[2]: # z is below threshold and point is below graping point
                    continue # avoid designation of a point as inaccessible due to point from lower surface 
                else:
                # Point is inaccessible due to proximity to another point
                    color_idx.append(i)
                    point_accessible = False
                    #np.asarray(input_pcd.colors)[color_idx[1:], :] = [1, 0, 0]
                    np.asarray(input_pcd.colors)[color_idx, :] = [1, 0, 0]

                    #print(f"Visualizing nearby points for {point_name}.")
                    #o3d.visualization.draw_geometries([input_pcd])
                    # print(f"Point {point_name} is inaccessible")
                    # o3d.visualization.draw_geometries([input_pcd])
                    # break
                
        
                

            # If point remains accessible after checks
            if point_accessible:
                accessible_points.append((idx, point, angle))
            
        
    o3d.visualization.draw_geometries([input_pcd])

    return accessible_points



