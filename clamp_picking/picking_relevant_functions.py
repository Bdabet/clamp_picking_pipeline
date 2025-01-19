import sys
import os
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)


from scene_analysis.scene_analysis import is_clamp_occluded, is_clamp_entangled


def determine_clamps_state(clamps):


    preferred_clamps = []
    occluded_clamps = []
    non_occluded_clamps = []
    entangled_clamps = []
    non_entangled_clamps = []


    # find state of each clamp (occluded, entangled)
    for clamp in clamps:

        occlusion_state, _ = is_clamp_occluded(clamp)
        if occlusion_state:
            occluded_clamps.append(clamp)
        else:
            non_occluded_clamps.append(clamp)
        entanglement_state, _ = is_clamp_entangled(clamp)
        if entanglement_state:
            entangled_clamps.append(clamp)
        else:
            non_entangled_clamps.append(clamp)
        
        if not occlusion_state and not entanglement_state:
            preferred_clamps.append(clamp)
            


    print("entangled_clamps are")
    for clamp in entangled_clamps:
        print(clamp.clamp_number)
    print("occluded_clamps are")
    for clamp in occluded_clamps:
        print(clamp.clamp_number)
    
    return non_entangled_clamps,non_occluded_clamps, preferred_clamps

def points_in_main_frame(pose, distances, clamp_orienation = None):
   
    # Step 1: Extract starting point and rotation angles
    x, y, z, roll, pitch, yaw = pose

    # to test
    if clamp_orienation == "v":
        roll = 180
        

    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)
    start_point = np.array([x, y, z])
    
    # Step 2: Create the rotation matrix from roll, pitch, and yaw
    rotation_matrix = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    
    
    
    unit_vector = np.array([0, 0, 1])  # Along z-axis
    

    # Step 4: Compute points in the main coordinate system
    points_main = []
    for d in distances:
        point_rotated = d * unit_vector  # Point in rotated frame
        point_main = rotation_matrix @ point_rotated  # Transform to main frame
        point_main += start_point  # Offset by the starting point
        points_main.extend(point_main)
    
    points_main = np.concatenate((points_main, np.array([np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw)])), axis = None)
    
    return np.array(points_main)

def turn_z_downwards(grasping_pose, clamp_orientation):
    """
    Adjust the grasping pose to ensure the z-axis points downwards.

    Parameters:
        grasping_pose (list): A list representing the pose [x, y, z, roll, pitch, yaw].
        clamp_orientation (str): Orientation of the clamp ('h' for horizontal, otherwise vertical).

    Returns:
        list: Adjusted grasping pose with z-axis pointing downwards.
    """
    # Create a local copy to avoid modifying the original pose
    adjusted_pose = grasping_pose[:]
    print("adjusted pose",adjusted_pose)
    pointing_upwards = False
    
    # Horizontal clamp case
    if clamp_orientation == "h":
        if abs(adjusted_pose[3]) < 90 and abs(adjusted_pose[4]) < 90:  # If z-axis is pointing upwards
            adjusted_pose[3] += 180  # Flip TCP 180 degrees around x-axis
            print("Adjusted roll by 180 to ensure z-axis points downwards (horizontal case).")
            pointing_upwards = True
        
    
    #vertical clamp case
    else:
        adjusted_pose[3] = adjusted_pose[3] - 90 #offset by 90 to start with 0 in vertical position
        # convert negative angles to their equivalent positive angles 
        # if adjusted_pose[3] <0 :
        #     adjusted_pose[3] = (adjusted_pose[3] +360) %360
        #     print(adjusted_pose)
        # if adjusted_pose[5] < 0 :
        #     adjusted_pose[5] = (adjusted_pose[5] +360) %360  


        if (-90 < adjusted_pose[3]%360 < 90) and (-90 < adjusted_pose[4]%360 < 90):  # If y-axis is pointing upwards
            adjusted_pose[3] += 180  # Flip TCP 90 degrees around x-axis

            # switch y and z axes
            # adjusted_pose[4] = grasping_pose[5]
            # adjusted_pose[5] = grasping_pose[4]
            print("Adjusted roll by 180 to ensure y-axis points downwards (vertical case).")

    # Vertical clamp case
    # else:
    #     print(abs(adjusted_pose[3] % 360))
    #     if ( 
    #         (abs(adjusted_pose[3] % 360) > 45 and abs(adjusted_pose[3] % 360) < 90) or 
    #        (abs(adjusted_pose[3] % 360) > 180 and abs(adjusted_pose[3] % 360) < 315)
    #        ):
            
    #         adjusted_pose[3] -= 90
    #         print("Adjusted roll by (+) 90 to ensure z-axis points downwards (vertical case).")
    
    #     elif ( 
    #         (abs(adjusted_pose[3] % 360) > 90 and abs(adjusted_pose[3] % 360) < 135) or 
    #        (abs(adjusted_pose[3] % 360) > 225 and abs(adjusted_pose[3] % 360) < 360)
    #        ):
    #         adjusted_pose[3] += 90
    #         print("Adjusted roll by (-) 90 to ensure z-axis points downwards (vertical case).")
        
    return adjusted_pose, pointing_upwards

def turn_z_downwards_adjusted(grasping_pose, clamp_orientation):

    pointing_upwards = False
    adjusted_pose = grasping_pose[:]

    if clamp_orientation == "v":
        adjusted_pose[3] = adjusted_pose[3] - 90

    if 270 <= adjusted_pose[3]%360 <= 360 or 0 <= adjusted_pose[3]%360 <= 90:
        adjusted_pose[3] = adjusted_pose[3] + 180
        pointing_upwards = True
    
    return adjusted_pose, pointing_upwards

def generate_close_points(base_point, num_points, max_offset=0.01):
    
    base_point = np.array(base_point)
    offsets = np.random.uniform(-max_offset, max_offset, size=(num_points, 3))
    close_points = base_point + offsets
    return close_points.tolist()

def rotate_around_local_z(grasping_pose, clamp, pointing_upwards, grasping_region_angle, robot_angle_offset = 45):
    
    # Current rotation based on the grasping pose
    r_current = R.from_euler('xyz', grasping_pose[3:], degrees=True)

    # Define the rotation about the local z-axis
    if clamp.clamp_orientation == "h":
        if pointing_upwards:
            grasping_region_angle = -grasping_region_angle
        r_local_z = R.from_euler('z', grasping_region_angle-robot_angle_offset, degrees=True)
    else:
        r_local_z = R.from_euler('z', -robot_angle_offset, degrees=True)  # Opposite of gripper displacement

    # Combine the rotations
    r_new = r_current * r_local_z

    # debug 
    # print("r_new", r_new.as_euler('xyz', degrees=True))

    # Update the final grasping pose
    grasping_pose = np.concatenate((grasping_pose[:3], r_new.as_euler('xyz', degrees=True)), axis=None)

    return grasping_pose

def visualize_grasping_positions(scene_pcd, concentrated_points, upper_grasping_pose, final_grasping_pose):
   
    # Combine concentrated points with the initial and final grasping positions
    grasping_points = np.concatenate(
        (concentrated_points, np.asarray([upper_grasping_pose[:3], final_grasping_pose[:3]])), axis=0
    )

    # Create a new point cloud for the grasping points
    grasping_points_pcd = o3d.geometry.PointCloud()
    grasping_points_pcd.points = o3d.utility.Vector3dVector(grasping_points)

    # Add color to the grasping points (red)
    grasping_points_pcd.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0] for _ in grasping_points])

    # Append the grasping points to the existing scene point cloud
    scene_pcd.points = o3d.utility.Vector3dVector(
        np.vstack((np.asarray(scene_pcd.points), np.asarray(grasping_points_pcd.points)))
    )

    # Visualize the updated point cloud
    o3d.visualization.draw_geometries([scene_pcd], window_name="PCD with Grasping Points")

    return scene_pcd

