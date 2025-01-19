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


# Clamp pcds paths
clamp_pcds_files_path = os.path.join(root_path, "files", "clamps_pcds_scaled")

# initialze global variables 

clamps = [] #list of clamp objects
scene_pcd = o3d.geometry.PointCloud() #pcd where all clamps are added


class Clamp():
    """
    Represents a clamp with associated bounding regions and attributes.

    Attributes:
        size (str): 's', 'm', or 'b' indicating small, medium, or big clamps.
        clamp_transformation (numpy.ndarray): Transformation matrix for the clamp.
        clamp_pose (list): Pose of the clamp (position and orientation, angles in radians).
        clamp_pcd (open3d.geometry.PointCloud): Clamp's point cloud.
        bounding_box_pcd (open3d.geometry.PointCloud): oriented bounding box encapsulating the clamp
        bounding_box_extended_pcd
        bounding_volume_pcd
        clamp_curve_pcd (open3d.geometry.PointCloud): simple line representation of clamp rubber part
        h_grasp_bounding_volume e(open3d.geometry.PointCloud): covers region for possible hrorizontal picking positions
        v_grasp_bounding_volume(open3d.geometry.PointCloud): covers region for possible vertical picking positions

    """

    number_of_clamps = 0

    # Predefined center points for each clamp size
    CLAMP_CENTERS = {
        "s": [5.74323225/1000, 13.14538544/1000, 0.25742388/1000],
        "m": [11.8132830/1000, 19.5173681/1000, 8.585e-06/1000],
        "b": [20.1405449/1000, 27.1667296/1000, 2.7179718e-05/1000],
    }

    def __init__(self, size, pose=None, combined_transformation=None):
        # Initialize attributes
        self.clamp_size = size  # "s", "m", or "b"
        if combined_transformation is not None:
            self.clamp_transformation = combined_transformation
            self.clamp_pose = extract_pose_from_transformation(combined_transformation)
        elif pose is not None:
            self.clamp_transformation = pose_to_transform_matrix_rpy(pose[:3], pose[3:])
            self.clamp_pose = pose
        else:
            raise ValueError("Either pose or transformation must be provided.")
        
        # find clamp orientaion 
        self.clamp_orientation = self._find_clamp_orientation()

        # increment number of clamps by 1 with each initialization
        Clamp.number_of_clamps += 1
        self.clamp_number = Clamp.number_of_clamps

        # Load point clouds 
        self.clamp_pcd = o3d.io.read_point_cloud(get_pcd_path("clamp", self.clamp_size))
        # self.clamp_pcd.points = o3d.utility.Vector3dVector(np.asarray(self.clamp_pcd.points) * 0.001)
        self.bounding_volume_pcd = o3d.io.read_point_cloud(get_pcd_path("inner_bounding_volume", self.clamp_size))
        self.bounding_box_pcd = o3d.io.read_point_cloud(get_pcd_path("bounding_box", self.clamp_size))
        self.bounding_box_extended_pcd = o3d.io.read_point_cloud(get_pcd_path("bounding_box_extended", self.clamp_size))
        self.clamp_curve_pcd = o3d.io.read_point_cloud(get_pcd_path("clamp_curve", self.clamp_size))
        

        # load json files       
        self.h_grasping_regions = get_json_path("clamp_grasping_regions_h", self.clamp_size)
        self.v_grasping_regions = get_json_path("clamp_grasping_regions_v", self.clamp_size)
        self.grasping_axis_rotations = get_json_path("axis_rotations", self.clamp_size)
        self.grasping_points = get_json_path("grasping_points", self.clamp_size)
        self.grasping_points_angles = get_json_path("grasping_points_angles", self.clamp_size)

         # Initialize the clamp center
        self.clamp_center_pcd = self._initialize_center() # center for visualization
        self.clamp_center = self.CLAMP_CENTERS[size]
        self.clamp_color = None

    def _initialize_center(self):
        """Initialize the center point cloud for the clamp."""
        if self.clamp_size not in self.CLAMP_CENTERS:
            raise ValueError(f"Invalid clamp size: {self.clamp_size}")
        center_pcd = o3d.geometry.PointCloud()
        center_pcd.points = o3d.utility.Vector3dVector([self.CLAMP_CENTERS[self.clamp_size]])
        return center_pcd

    def _find_clamp_orientation(self):
           

        # two iterations (x and y axes angles)
        for idx in range(2):

            axis_angle = abs(self.clamp_pose[idx+3]%360) # find the angle of tilt of x, then y axes
            

            if ((axis_angle>=(315) and axis_angle<=(360)) or 
                (axis_angle>=( 0 ) and axis_angle<=(45))  or 
                (axis_angle>=(135) and axis_angle<=(225))):

                orientation = "h"
            else:
                orientation = "v"
                break
        
        return orientation
    
    def add_clamp_to_representation(self):
        """Add clamp to scene with new orientation and transforms."""
       
        if self.clamp_transformation is not None:

            # Transform all point clouds
            self.clamp_pcd.transform(self.clamp_transformation)
            self.clamp_curve_pcd.transform(self.clamp_transformation)
            self.bounding_volume_pcd.transform(self.clamp_transformation)
            self.bounding_box_pcd.transform(self.clamp_transformation)
            self.bounding_box_extended_pcd.transform(self.clamp_transformation)
            self.clamp_center_pcd.transform(self.clamp_transformation)


            self.clamp_center = transform_xyz_coordinate(self.clamp_transformation, self.clamp_center)


            # transform grasping regions json files
            for region_name, region_points in self.h_grasping_regions.items():
                tranformed_coordinates = []
                
                for coordinate in region_points:
                    
                    new_coordinate = transform_xyz_coordinate(self.clamp_transformation, coordinate)
                    
                    tranformed_coordinates.append(new_coordinate)

                self.h_grasping_regions[region_name] = np.asarray(tranformed_coordinates)

                
            for region_name, region_points in self.v_grasping_regions.items():
                tranformed_coordinates = []
                
                for coordinate in region_points:
                    
                    new_coordinate = transform_xyz_coordinate(self.clamp_transformation, coordinate)
                    #print(new_coordinate)
                    tranformed_coordinates.append(new_coordinate)

                self.v_grasping_regions[region_name] = np.asarray(tranformed_coordinates)
            
            for points_name, point in self.grasping_points.items():

                tranformed_coordinates = []
                new_coordinate = transform_xyz_coordinate(self.clamp_transformation, point)
                self.grasping_points[points_name] = np.asarray(new_coordinate)

        
         
        # add random color to clamps
        if not self.clamp_pcd.has_colors():
            self.clamp_pcd.colors = o3d.utility.Vector3dVector(np.ones((len(self.clamp_pcd.points), 3)))  # Initialize if no colors
        
        np.asarray(self.clamp_pcd.colors)[:] = [random.randint(50,100)/100,
                                                random.randint(50,100)/100, 
                                                random.randint(50,100)/100]  # Set all points of clamp_pcd to a random uniform color

        # Ensure scene_pcd has colors, initialize if not present
        if not scene_pcd.has_colors():
            scene_pcd.colors = o3d.utility.Vector3dVector(
                np.ones((len(scene_pcd.points), 3)))  # Default color (white)
        #add clamp to main pcd
        scene_pcd.points = o3d.utility.Vector3dVector(
            np.vstack((np.asarray(scene_pcd.points), self.clamp_pcd.points)))
        
        scene_pcd.colors = o3d.utility.Vector3dVector(
            np.vstack((np.asarray(scene_pcd.colors), np.asarray(self.clamp_pcd.colors))))
    
    def show_all_clamp_pcds(self):

        # visualize one pcds one after the other
        o3d.visualization.draw_geometries([self.clamp_pcd, self.clamp_curve_pcd ], window_name="clamp curve")
        o3d.visualization.draw_geometries([self.clamp_pcd, self.bounding_box_pcd ], window_name="bounding box")
        o3d.visualization.draw_geometries([self.clamp_pcd, self.bounding_box_extended_pcd ], window_name="bounding box extended")
        o3d.visualization.draw_geometries([self.clamp_pcd, self.bounding_volume_pcd ], window_name="bounding volume")
        o3d.visualization.draw_geometries([self.clamp_pcd, self.clamp_center_pcd ], window_name="clamp center")

def get_pcd_path(pcd_type,clamp_size):

    
    if clamp_size == "s":
        size = "small"
    elif clamp_size == "m":
        size = "medium"
    elif clamp_size == "b":
        size = "big"
    else:
        print("size is invalid")

    if pcd_type == "clamp":
        return  os.path.join(root_path, clamp_pcds_files_path, "clamps_pcd",f"{size}_clamp_aligned.pcd" )
    elif pcd_type == "bounding_box":
        return os.path.join(root_path, clamp_pcds_files_path, "clamps_bounding_box",f"{size}_clamp_bounding_box.pcd")
    elif pcd_type == "bounding_box_extended":
        return os.path.join(root_path, clamp_pcds_files_path, "clamps_bounding_box_extended",f"{size}_clamp_bounding_box_extended.pcd")
    elif pcd_type == "inner_bounding_volume":
        return os.path.join(root_path, clamp_pcds_files_path, "clamps_inner_bounding_volume",f"{size}_clamp_inner_bounding_volume.pcd")
    elif pcd_type == "clamp_curve":
        return os.path.join(root_path, clamp_pcds_files_path, "clamps_curve", f"{size}_clamp_curve.pcd" )
    elif pcd_type == "h_grasping_region":
        return os.path.join(root_path, clamp_pcds_files_path, "clamps_grasping_regions_h" , f"{size}_clamp_grasping_regions_h.pcd" )
    elif pcd_type == "v_grasping_region":
        return os.path.join(root_path, clamp_pcds_files_path, "clamps_grasping_regions_v" , f"{size}_clamp_grasping_regions_v.pcd" )

def get_json_path(json_file_type, clamp_size):
     

    if clamp_size == "s":
        size = "small"
    elif clamp_size == "m":
        size = "medium"
    elif clamp_size == "b":
        size = "big"
    else:
        print("size is invalid")

    if json_file_type == "clamp_grasping_regions_h":
        with open(os.path.join(root_path, "files", "json_files_scaled", "clamp_grasping_regions", f"{size}_clamp_grasping_regions_h.json"), 'r') as fp:
            return json.load(fp)
    elif json_file_type == "clamp_grasping_regions_v":
        with open(os.path.join(root_path, "files", "json_files_scaled", "clamp_grasping_regions", f"{size}_clamp_grasping_regions_v.json"), 'r') as fp:
            return json.load(fp)
    elif json_file_type == "axis_rotations":
        with open(os.path.join(root_path, "files", "json_files_scaled", "regions_angles", f"{size}_clamp_regions_angles.json"), 'r') as fp:
            return json.load(fp)
    elif json_file_type == "grasping_points":
        with open(os.path.join(root_path, "files", "json_files_scaled", "clamp_grasping_points", f"{size}_clamp_grasping_points.json"), 'r') as fp:
            return json.load(fp)
    elif json_file_type == "grasping_points_angles":
        with open(os.path.join(root_path, "files", "json_files_scaled", "clamp_grasping_points_angles", f"{size}_clamp_grasping_points_angles.json"), 'r') as fp:
            return json.load(fp)



def transform_xyz_coordinate(transformation,coordinate):

        # Convert the current bounding box coordinate to a NumPy array
        coordinate = np.asarray(coordinate)
        
        # Append 1 to create homogeneous coordinates
        coordinate = np.append(coordinate, 1)
        
        # Reshape to a 4x1 column vector
        column_matrix = coordinate.reshape(4, 1)
        
        # Apply the transformation
        transformed_coord = np.dot(transformation, column_matrix)

        # Update the bounding box coordinates, excluding the homogeneous part
        transformed_coord = transformed_coord[:3].flatten()
        #print(f"transformed c ={transformed_coord}")
        return transformed_coord 


def extract_pose_from_transformation(matrix, origin = None):
    """
    Extracts the pose (position and orientation) from a 4x4 transformation matrix.
    
    Parameters:
    matrix (np.ndarray): 4x4 transformation matrix
    
    
    """
    # Check if the matrix is 4x4
    if matrix.shape != (4, 4):
        raise ValueError("Input matrix must be a 4x4 transformation matrix.")
    
    # Extract translation (position) from the last column of the matrix
    if origin is None:
        position = transform_xyz_coordinate(matrix, [0, 0, 0])
    else:
        position = transform_xyz_coordinate(matrix, origin[:3])
    
    
    # Extract rotation (orientation) part from the upper-left 3x3 matrix
    rotation_matrix = matrix[:3, :3]
    
    # Convert rotation matrix to Euler angles (RPY)
    r = R.from_matrix(rotation_matrix)
    if origin is None:
        rpy = r.as_euler('xyz', degrees = True)  # roll, pitch, yaw in degrees
    else:
        rpy = r.as_euler('xyz')
        rpy =  origin[3:] + r.as_euler('xyz', degrees=True)

    return np.concatenate((position, rpy))


def pose_to_transform_matrix_rpy(position, rpy):
    """
    Converts a pose (position and orientation as roll, pitch, yaw) into a 4x4 transformation matrix.
    
    Args:
    - position: tuple or list of (x, y, z) coordinates.
    - rpy: tuple or list of (roll, pitch, yaw) angles in radians.
    
    Returns:
    - A 4x4 transformation matrix as a numpy array.
    """
    # Extract position
    x, y, z = position
    
    # Create rotation matrix from roll, pitch, yaw
    rotation_matrix = R.from_euler('xyz', rpy).as_matrix()
    
    # Build the 4x4 transformation matrix
    transform_matrix = np.eye(4)  # Start with an identity matrix
    transform_matrix[:3, :3] = rotation_matrix  # Insert rotation
    transform_matrix[:3, 3] = [x, y, z]  # Insert translation
    
    return transform_matrix


if __name__ == "__main__":

    #testing
    print(os.path.join(root_path, "files", "json_files", "big_clamp_grasping_regions_h.json"))
    



