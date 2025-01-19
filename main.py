import os
import numpy as np
import open3d as o3d
import pickle
import zivid
from scipy.spatial.transform import Rotation as R
from config import root_path
from rgb_segmentation.fine_tuned_sam2_segmentation import crop_and_segment, convert_zdf
from pcd_segmentation.segment_zdf_pcd_file import segment_pcd_masks, create_pcd_excluding_masks
from ransac_and_icp.ransac_icp_determine_clamp_transf import find_clamp_size, find_best_fit
from scene_representation.representation_creation import Clamp, clamps, scene_pcd
from scene_analysis.scene_analysis import  find_grasping_pose_points, find_accessible_points, find_accessible_points_with_walls
from clamp_picking.picking_relevant_functions import determine_clamps_state, points_in_main_frame, turn_z_downwards_adjusted, generate_close_points, visualize_grasping_positions, rotate_around_local_z


def load_paths(captured_scene_name):
    """
    Load all necessary file paths for the project based on the root path and scene name.
    """
    files_root_path = os.path.join(root_path, "files")
    paths = {
        "images": os.path.join(files_root_path, "images"),
        "zdf_files": os.path.join(files_root_path, "zdf_files"),
        "generated_png_masks": os.path.join(files_root_path, "generated_png_masks"),
        "segmented_pcds": os.path.join(files_root_path, "segmented_pcd_files"),
        "pickle_output": os.path.join(files_root_path, "pickle_files", f"{captured_scene_name}.pkl"),
        "scene_full_pcd": os.path.join(files_root_path, "scene_full_pcd")
    }
    return paths

def delete_old_seg_results():

    for file in os.listdir(paths["generated_png_masks"]):
        os.remove(os.path.join(paths["generated_png_masks"], file))
    print("deleted old masks")
    for file in os.listdir(paths["segmented_pcds"]):
        os.remove(os.path.join(paths["segmented_pcds"], file))
    print("deleted segmented pcds")

def segment_rgb_image(image_path, num_samples=500, display_image = True):
    """
    Segment the RGB image using the SAM2 model and generate masks.
    
    Args:
        image_path (str): Path to the input RGB image.
        num_samples (int): Number of samples for segmentation refinement.
    """

    
    crop_and_segment(image_path, num_samples = num_samples, display_image = display_image)
    print("Loaded and segmented RGB image")

def segment_point_cloud(zdf_path, mask_dir, save_path, pcd_transformation = None, app = None):
    """
    Segment the point cloud data using RGB image masks.
    
    Args:
        zdf_path (str): Path to the ZDF file.
        mask_dir (str): Directory containing the masks generated from the RGB image.
        save_path (str): Directory where the segmented point cloud files will be saved.
    """
    segment_pcd_masks(zdf_path, mask_dir, save_path, transformation_matrix = pcd_transformation, app=app)
    print("Segmented point cloud using RGB masks")   

def ransac_icp_pipeline(segmented_pcds_path, output_path, pcd_transformation = None, display_pcd = False, z_threshold = 0): 
    """
    Perform RANSAC and ICP fitting for all segmented point cloud files.
    
    Args:
        segmented_pcds_path (str): Directory containing segmented PCD files.
        output_path (str): Path to save the ICP results as a pickle file.
        pcd_transformation: tranformation of pcd to robot's base
        Z_threshold: height above which clamps are found
    """
    icp_data = {}
    for idx, pcd_file_name in enumerate(os.listdir(segmented_pcds_path)):
        target_path = os.path.join(segmented_pcds_path, pcd_file_name)

        # load current pcd and scale it down and tranform it according to transformation matrix
        current_point_cloud = o3d.io.read_point_cloud(target_path)
        current_point_cloud.points = o3d.utility.Vector3dVector(np.asarray(current_point_cloud.points) * 0.001)
        current_point_cloud.transform(pcd_transformation)

        # Get z-coordinates of all points
        z_values = np.asarray(current_point_cloud.points)[:, 2]
        

        # Check if any z-coordinate is under the z threshold
        if np.all(z_values < z_threshold):
            print(f"mask {pcd_file_name} is invalid due to not satisfying z_threshold")
            continue # skip invalud masks

        # Determine the size of the clamp by comparing with the predefined models.
        size = find_clamp_size(target_path, pcd_tranformation = pcd_transformation, display_pcd=display_pcd)

        # Perform multiple RANSAC and ICP iterations to find the best fit transformation.
        combined_transformation, pose, icp_fine_tune = find_best_fit(
            target_path, size, pcd_transformation = pcd_transformation,number_of_iterations=1, display_pcd= display_pcd
        )

        # if not is_size_valid(target_path, size, combined_transformation , calibration_transformation = pcd_transformation):
        #     print(print(f"PCD {pcd_file_name} was discarded due to invalid size"))
        #     continue
        
        # Only save the results if the fitness of the transformation is above a threshold.
        if icp_fine_tune.fitness > 0.19:
            icp_data[f"clamp{idx+1}"] = {
                "clamp size": size,
                "combined transformation": combined_transformation,
                "pose": pose,
                "mask name" : pcd_file_name
            }
            valid_mask_names.append(pcd_file_name)
        else:
            print(f"PCD {pcd_file_name} was discarded due to bad fit result")
    
    

    # Save the RANSAC and ICP results as a pickle file for later use.
    with open(output_path, 'wb') as fp:
        pickle.dump(icp_data, fp)

def create_scene_representation(icp_data, scene_pcd, clamps):
    """
    Create the scene representation by adding clamps to the scene point cloud.
    
    Args:
        icp_data (dict): Results from the RANSAC and ICP pipeline.
        scene_pcd (o3d.geometry.PointCloud): Scene point cloud representation.
        clamps (list): List to store Clamp objects for scene analysis.
    """
    for data in icp_data.values():
        # Extract clamp size and transformation matrix from the ICP data.

        clamp_size = data["clamp size"][0]
        clamp_transformation_matrix = np.asarray(data["combined transformation"])
        clamps.append(Clamp(clamp_size, combined_transformation=clamp_transformation_matrix))

    # add clamp into scene pcd with its accompanying pcds
    for clamp in clamps:
        clamp.add_clamp_to_representation()

def visualize_scene(scene_pcd, transformation_matrix, real_scene_path = None):
    """
    Visualize the scene representation along with the real point cloud.
    
    Args:
        scene_pcd (o3d.geometry.PointCloud): Generated scene representation.
        real_scene_path (str): Path to the real scene's point cloud file.
    """
    real_scene = o3d.io.read_point_cloud(real_scene_path)
    real_scene.points = o3d.utility.Vector3dVector(np.asarray(real_scene.points) * 0.001)

   
    transformation_matrix = transformation_matrix

    real_scene.transform(transformation_matrix)
    o3d.visualization.draw_geometries([scene_pcd, real_scene], window_name="Scene PCD")


    # creat scene with remaing points in pcd not coverd by a mask
    
def pick_appropriate_clamp(other_points_pcd = None, real_scene_path = None):

    
    
    # determine states of identified clamps
    non_entangled_clamps, non_occluded_clamps, preferred_clamps = determine_clamps_state(clamps)

    if preferred_clamps:
        # sort preferred clamps by descending height
        preferred_clamps.sort(key = lambda clamp: clamp.clamp_center[2], reverse = True)

    if non_occluded_clamps:
        # sort non_occluded clamps by descending height
        non_occluded_clamps.sort(key = lambda clamp: clamp.clamp_center[2], reverse = True)

    if non_entangled_clamps:
        # sort non_entangled clamps by descending height
        non_entangled_clamps.sort(key = lambda clamp: clamp.clamp_center[2], reverse = True)

    # extend clamps according to priority starting with highest priority clamps to least priority
    preferred_clamps.extend(non_occluded_clamps)
    preferred_clamps.extend(non_entangled_clamps)
    preferred_clamps.extend(clamps)

    # sort all clamps by descending height
    preferred_clamps.sort(key = lambda clamp: clamp.clamp_center[2], reverse = True)
    
    print("preferred clamps are")
    for preferred_clamp in preferred_clamps:
        print(preferred_clamp.clamp_number)
    
 

    # find grasping point by seaching for accesible region
    real_scene = o3d.io.read_point_cloud(real_scene_path)
    overlay_pcd = scene_pcd + real_scene
    for preferred_clamp in preferred_clamps:


        
        accessible_points, accessible_points_indexes = find_accessible_points(preferred_clamp, visualize_nearby_points = False)
        accessible_points = find_accessible_points_with_walls(preferred_clamp, accessible_points_indexes, search_radius=0.014, input_pcd = other_points_pcd, visualize_nearby_points = False, scene_pcd = overlay_pcd)

        for point in accessible_points:
            print(f"point {point} is accessible")


        if not accessible_points:
                print("couldnt find accesible point searching with smaller radius")
                # search with smaller radius
                accessible_points, accessible_points_indexes = find_accessible_points(preferred_clamp, search_radius = 0.01, visualize_nearby_points = False) # 10 mm radius
                accessible_points = find_accessible_points_with_walls(preferred_clamp, accessible_points_indexes, input_pcd= other_points_pcd, search_radius = 0.01, visualize_nearby_points = False)

        if not accessible_points:
            print("coud not find accesible point for firts preferred clamp, trying next clamp")
            continue
        if accessible_points:
            position, angle = find_grasping_pose_points(preferred_clamp, accessible_points)
            grasping_pcd = scene_pcd+other_points_pcd
            find_grasping_pose_points(preferred_clamp, accessible_points, input_pcd = grasping_pcd, visualize = True)

            print(f"clamp attemting to pick is clamp {preferred_clamp.clamp_number} with orientation {preferred_clamp.clamp_orientation}")
            
            break # leave for loop when first clamp with accesible region is found


    # find grapsing pose by giving specific grasping region
    # position, angle = find_grasping_pose(preferred_clamps[0], grasping_region_number= 3)
    # print("angle", angle)

    # join grapsing position with angles
    initial_grasping_pose = list(position)
    pose_angles = preferred_clamp.clamp_pose[3:].copy()
    initial_grasping_pose.extend(pose_angles)

    # extract clamp pose and grasping region position
    print("clamp pose", preferred_clamp.clamp_pose)
    print("positon at grasping region", position)
    
    # turn x  so that z axis is pointing downwards
    downward_oriented_pose, pointing_upwards = turn_z_downwards_adjusted(initial_grasping_pose, preferred_clamp.clamp_orientation)
    print("downward_oriented_pose",downward_oriented_pose)


    # rotate around z axis for gripper offset and regiion angles in horizontal case
    nondisplaced_grapsing_pose = rotate_around_local_z(downward_oriented_pose, preferred_clamp, pointing_upwards, angle, robot_angle_offset = 45)

    final_grasping_pose = points_in_main_frame(nondisplaced_grapsing_pose.copy(), [-0.17], preferred_clamp.clamp_orientation)
    
    upper_grasping_pose = points_in_main_frame(nondisplaced_grapsing_pose.copy(), [-0.27], preferred_clamp.clamp_orientation)

        

    print("upper picking pose",upper_grasping_pose)
    print("final picking pose",final_grasping_pose)

    # visualize grasping postions
    concentrated_points = generate_close_points(downward_oriented_pose[:3], 200, 0.001)
    visualize_grasping_positions(scene_pcd, concentrated_points, upper_grasping_pose, final_grasping_pose )

        


if __name__ == "__main__":

    np.set_printoptions(suppress=True)

    app = zivid.Application()

    preferred_clamps = []
    occluded_clamps = []
    non_occluded_clamps = []
    entangled_clamps = []
    non_entangled_clamps = []
    valid_mask_names = []



    # define transfer matrix to transform Camera PCD to robot's base frame
    transformation_matrix = np.array([
        [0.7421,    0.5991,   -0.3005,    0.6151],
        [0.6680,   -0.6246,    0.4044,   -0.2738],
        [0.0546,   -0.5009,   -0.8638,    0.3521],
        [0,         0,         0,    1.0000]
    ])
    
    




    # Define the name of the captured scene to process.
    captured_scene_name = "clamps_in_box_4"
    # Load all required paths.
    paths = load_paths(captured_scene_name)
    image_path = os.path.join(paths["images"], f"{captured_scene_name}.png")
    zdf_path = os.path.join(paths["zdf_files"], f"{captured_scene_name}.zdf")
    scene_pcd_path = os.path.join(paths["scene_full_pcd"], f"{captured_scene_name}.ply")
    

    # Step 0: delete segmented images and pcds from previous runs

    delete_old_seg_results()
    
    # Step 1: Etxract and then segment the RGB image.
    
    convert_zdf(zdf_path, image_path, scene_pcd_path, app)
    
    segment_rgb_image(image_path, display_image = False)


    # Step 2: Segment the point cloud data using the generated masks.
    
    segment_point_cloud(zdf_path, paths["generated_png_masks"], paths["segmented_pcds"], pcd_transformation = None, app = app)

    # Step 3: Perform RANSAC and ICP to determine clamp poses.

    ransac_icp_pipeline(paths["segmented_pcds"], paths["pickle_output"], pcd_transformation = transformation_matrix, display_pcd= False, z_threshold = 0.03)


    # Step 4: Create the scene representation using the extracted poses.

    with open(paths["pickle_output"], 'rb') as fp:
        icp_data = pickle.load(fp)

    
    create_scene_representation(icp_data, scene_pcd, clamps)


    # Step 5: Visualize the generated scene and the real scene.


    visualize_scene(scene_pcd, transformation_matrix, scene_pcd_path)


    # step 6 Create filtered PCD removing points belonging to unidetifed walls

    pcd_without_clamps = create_pcd_excluding_masks(zdf_path, paths["generated_png_masks"], icp_data, transformation_matrix = transformation_matrix ,app = app)
    

    # step 6: determine which clamp to pick and determine pose to be sent to robot

    pick_appropriate_clamp(pcd_without_clamps, scene_pcd_path)


      
    




