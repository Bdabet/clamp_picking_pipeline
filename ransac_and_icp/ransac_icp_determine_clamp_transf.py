import os
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys

# Determine the root path 
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
from config import root_path


clamps_pcds_path = os.path.join(root_path, "files", "clamps_pcds_scaled", "clamps_pcd")


def translate_to_origin(pcd):
    centroid = np.mean(np.asarray(pcd.points), axis=0)
    pcd.translate(-centroid)
    return centroid, pcd

def ransac_icp_fitting(source_path, target_path, no_of_conv_iterations, reverse = False, pcd_tranformation = None, find_size=False, display_pcd=False, display_info=True):

    if not reverse:
        source = o3d.io.read_point_cloud(source_path)
        target = o3d.io.read_point_cloud(target_path)

         # scale down target pcd to millimeter and transform it according to calibration matrix
        target.points = o3d.utility.Vector3dVector(np.asarray(target.points) * 0.001)
        if pcd_tranformation is not None: 
            target.transform(pcd_tranformation)

        # Translate target point cloud to the origin
        centroid, target = translate_to_origin(target)
    else:
        source = o3d.io.read_point_cloud(target_path)
        target = o3d.io.read_point_cloud(source_path)

        # scale down source (reverse) pcd to millimeter and trnsform it according to calibration matrix
        source.points = o3d.utility.Vector3dVector(np.asarray(source.points) * 0.001)

        if pcd_tranformation is not None: 
            source.transform(pcd_tranformation)

        # Translate target point cloud to the origin
        centroid, source = translate_to_origin(source)


    if display_pcd:
        o3d.visualization.draw_geometries([source, target], window_name="target Alignment at Origin")

    # Downsample the point clouds
    voxel_size = 0.0015  # Scaled down by 1000
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)

    def compute_fpfh(pcd, voxel_size):
        radius_normal = voxel_size * 2  # Scaled appropriately
        radius_feature = voxel_size * 5  # Scaled appropriately

        # Estimate normals
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        # Compute FPFH feature
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return fpfh

    source_fpfh = compute_fpfh(source_down, voxel_size)
    target_fpfh = compute_fpfh(target_down, voxel_size)

    # Apply RANSAC for target alignment 
    distance_threshold = 0.002375  # Scaled down by 1000
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,  
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.7),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)], 
        o3d.pipelines.registration.RANSACConvergenceCriteria(no_of_conv_iterations, 0.99))

    if display_info:
        print("RANSAC target Alignment")
        print(result_ransac)

    if find_size:
        return result_ransac

    # Estimate normals for the target point cloud (required for ICP)
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.0001, max_nn=30))  # Scaled by 1000

    # ICP refinement
    icp_fine_tune = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    # Transform the source point cloud
    source.transform(icp_fine_tune.transformation)
    
    
    if not reverse:
        source.translate(centroid)
        target_original = o3d.io.read_point_cloud(target_path)
        target_original.points = o3d.utility.Vector3dVector(np.asarray(target_original.points) * 0.001)
        if pcd_tranformation is not None: 
            target_original.transform(pcd_tranformation)
        if display_pcd:
            # Visualize the final alignment
            o3d.visualization.draw_geometries([target_original, source], window_name="Final Alignment")
    else:
        
        if display_pcd:
            # Visualize the final alignment
            o3d.visualization.draw_geometries([target, source], window_name="Final Alignment")


    

    centroid_translation = np.eye(4)
    centroid_translation[:3, 3] = centroid  # Convert the centroid to a 4x4 translation matrix

    # Combine the centroid and ICP transformation by matrix multiplication
    combined_transformation = np.dot(centroid_translation, icp_fine_tune.transformation)


    if display_info:
        print(f"combined transformation after = {combined_transformation}")

    return combined_transformation, icp_fine_tune

def find_clamp_size(target_path,  pcd_tranformation = None, display_pcd = False, display_info = True):

    if display_info == False:
        # limit Open3D output to errors
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    size_list = ["small","medium","big"]
    result_ransac = None

    for idx in range(3):
        
        for i in range(1):
        
            current_size = size_list[idx]

            source_path = os.path.join(clamps_pcds_path, f"{current_size}_clamp_aligned.pcd")
            target_path = target_path
            # if is_size_valid(target_path, source_path):
            current_result_ransac = ransac_icp_fitting(source_path, target_path, pcd_tranformation = pcd_tranformation,  find_size = True, no_of_conv_iterations = 10000000, display_pcd = display_pcd, display_info = display_info)

            if result_ransac is None:
                result_ransac = current_result_ransac
                size = current_size
                

            elif len(current_result_ransac.correspondence_set) > len(result_ransac.correspondence_set):
                result_ransac = current_result_ransac
                size = current_size

    
    if display_info:
        print(f"size is {size}")

    return size

def find_best_fit(target_path, size, number_of_iterations, reverse = False, pcd_transformation = None, display_pcd = False, display_info = True):

    if display_info == False:
        # limit Open3D output to errors
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    icp_fine_tune = None

    for _ in range(number_of_iterations):
        
        # Load source and target point clouds
        source_path = os.path.join(clamps_pcds_path, f"{size}_clamp_aligned.pcd")
        target_path = target_path
        

        current_combined_transformation, current_icp_fine_tune = ransac_icp_fitting(source_path, target_path,  reverse = reverse, pcd_tranformation = pcd_transformation, no_of_conv_iterations = 10000000, display_pcd = display_pcd, display_info = display_info)
        

        if icp_fine_tune is None:
            icp_fine_tune = current_icp_fine_tune
            combined_transformation = current_combined_transformation
            R_matrix = combined_transformation[:3, :3]
        elif len(current_icp_fine_tune.correspondence_set) > len(icp_fine_tune.correspondence_set):
            combined_transformation = current_combined_transformation
            icp_fine_tune = current_icp_fine_tune
            R_matrix = combined_transformation[:3, :3]  # Extract the 3x3 rotation matrix
        
        
        translation_vector = combined_transformation[:3, 3]  # Extract the 1x3 translation vector

        
        rotation = R.from_matrix(R_matrix) # Convert the rotation matrix to Euler angles (in radians)
        euler_angles = rotation.as_euler('xyz', degrees=True)  # 'xyz' convention, change if needed

        pose = [translation_vector[0], translation_vector[1], translation_vector[2], euler_angles[0], euler_angles[1], euler_angles[2]]

    
    return combined_transformation, pose, icp_fine_tune
            
def is_size_valid(segmented_pcd_path, size, icp_transformation, calibration_transformation = None):

    segmented_pcd = o3d.io.read_point_cloud(segmented_pcd_path)
    
    if calibration_transformation is not None: 
            segmented_pcd.transform(calibration_transformation)

    segmented_pcd.transform(icp_transformation)
    comparison_pcd_path = os.path.join(clamps_pcds_path, f"{size}_clamp_aligned.pcd")

    

    
    # Perform statistical outlier removal
    cl, ind = segmented_pcd.remove_statistical_outlier(nb_neighbors=250, std_ratio=2.0)
    # scale down segmented pcd
    segmented_pcd.points = o3d.utility.Vector3dVector(np.asarray(segmented_pcd.points) * 0.001)

    # Select inlier points 
    segmented_pcd_inliers = segmented_pcd.select_by_index(ind)

    segmented_pcd_array = np.asarray(segmented_pcd_inliers.points)
    
    # find bodunding lengths of segmented pcd
    segmented_pcd_ranges = np.max(segmented_pcd_array, axis = 0) - np.min(segmented_pcd_array, axis = 0)
    #print(segmented_pcd_ranges)

    # get bounding lengths of maximum possible pcd
    comparison_clamp_pcd = o3d.io.read_point_cloud(comparison_pcd_path)
    comparison_clamp_pcd_array = np.asarray(comparison_clamp_pcd.points)
    max_ranges = np.max(comparison_clamp_pcd_array, axis = 0) - np.min(comparison_clamp_pcd_array, axis = 0)

    range_difference = segmented_pcd_ranges - max_ranges
    print("range difference", range_difference)
    print("max ranges", max_ranges)
    print(np.max(abs(range_difference/max_ranges)))
    if (np.max(abs(range_difference/max_ranges)) > 2):
        return False
    else:
        return True

def ransac_icp_fitting_box(source_path, target_path, no_of_conv_iterations, reverse = False, pcd_tranformation = None, find_size=False, display_pcd=False, display_info=True):

    if not reverse:
        source = o3d.io.read_point_cloud(source_path)
        target = o3d.io.read_point_cloud(target_path)

         # scale down target pcd to millimeter and transform it according to calibration matrix
        # target.points = o3d.utility.Vector3dVector(np.asarray(target.points) * 0.001)
        if pcd_tranformation is not None: 
            target.transform(pcd_tranformation)

        # Translate target point cloud to the origin
        centroid, target = translate_to_origin(target)
    else:
        source = o3d.io.read_point_cloud(target_path)
        target = o3d.io.read_point_cloud(source_path)

        # scale down source (reverse) pcd to millimeter and trnsform it according to calibration matrix
        source.points = o3d.utility.Vector3dVector(np.asarray(source.points) * 0.001)

        if pcd_tranformation is not None: 
            source.transform(pcd_tranformation)

        # Translate target point cloud to the origin
        centroid, source = translate_to_origin(source)


    if display_pcd:
        o3d.visualization.draw_geometries([source, target], window_name="target Alignment at Origin")

    # Downsample the point clouds
    voxel_size = 1.5  # Scaled down by 1000
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)

    def compute_fpfh(pcd, voxel_size):
        radius_normal = voxel_size * 2  # Scaled appropriately
        radius_feature = voxel_size * 5  # Scaled appropriately

        # Estimate normals
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        # Compute FPFH feature
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return fpfh

    source_fpfh = compute_fpfh(source_down, voxel_size)
    target_fpfh = compute_fpfh(target_down, voxel_size)

    # Apply RANSAC for target alignment 
    distance_threshold = 2.375  # Scaled down by 1000
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,  
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.7),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)], 
        o3d.pipelines.registration.RANSACConvergenceCriteria(no_of_conv_iterations, 0.99))

    if display_info:
        print("RANSAC target Alignment")
        print(result_ransac)

    if find_size:
        return result_ransac

    # Estimate normals for the target point cloud (required for ICP)
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))  # Scaled by 1000

    # ICP refinement
    icp_fine_tune = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    # Transform the source point cloud
    source.transform(icp_fine_tune.transformation)
    
    
    if not reverse:
        source.translate(centroid)
        target_original = o3d.io.read_point_cloud(target_path)
        target_original.points = o3d.utility.Vector3dVector(np.asarray(target_original.points))
        if pcd_tranformation is not None: 
            target_original.transform(pcd_tranformation)
        if display_pcd:
            # Visualize the final alignment
            o3d.visualization.draw_geometries([target_original, source], window_name="Final Alignment")
    else:
        
        if display_pcd:
            # Visualize the final alignment
            o3d.visualization.draw_geometries([target, source], window_name="Final Alignment")


    

    centroid_translation = np.eye(4)
    centroid_translation[:3, 3] = centroid  # Convert the centroid to a 4x4 translation matrix

    # Combine the centroid and ICP transformation by matrix multiplication
    combined_transformation = np.dot(centroid_translation, icp_fine_tune.transformation)


    if display_info:
        print(f"combined transformation after = {combined_transformation}")

    return combined_transformation, icp_fine_tune

if __name__ == "__main__":

   # testing
   x = 0