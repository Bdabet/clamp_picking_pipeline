#include <filesystem>
#include <open3d/Open3D.h>
#include <iostream>
#include <Eigen/Dense> 
#include <nlohmann/json.hpp>
#include <fstream>



const std::filesystem::path PROJECT_ROOT = std::filesystem::path(__FILE__).parent_path().parent_path();


const std::string bigClampPcdPath = (PROJECT_ROOT/ "files" / "clamps_pcds_scaled").string();



// The shared scene point cloud 
auto scene_pcd = std::make_shared<open3d::geometry::PointCloud>();


// ---------------------------------------------------------------------------
// Helper: get the path to a .pcd file given its type and clamp size
// ---------------------------------------------------------------------------

std::string get_pcd_path(const std::string& pcd_type, const std::string& clamp_size) {

    // Map "s"/"m"/"b" → "small"/"medium"/"big"
    std::string size;
    if      (clamp_size == "s") size = "small";
    else if (clamp_size == "m") size = "medium";
    else if (clamp_size == "b") size = "big";
    else {
        std::cerr << "Invalid clamp size: " << clamp_size << "\n";
        return "";
    }


    std::filesystem::path base = PROJECT_ROOT / "files" / "clamps_pcds_scaled";
    // Build the file path based on type
    if      (pcd_type == "clamp")
        return (base / "clamps_pcd"                  / (size + "_clamp_aligned.pcd")).string();
    else if (pcd_type == "bounding_box")
        return (base / "clamps_bounding_box"          / (size + "_clamp_bounding_box.pcd")).string();
    else if (pcd_type == "bounding_box_extended")
        return (base / "clamps_bounding_box_extended" / (size + "_clamp_bounding_box_extended.pcd")).string();
    else if (pcd_type == "inner_bounding_volume")
        return (base / "clamps_inner_bounding_volume" / (size + "_clamp_inner_bounding_volume.pcd")).string();
    else if (pcd_type == "clamp_curve")
        return (base / "clamps_curve"                 / (size + "_clamp_curve.pcd")).string();
    else if (pcd_type == "h_grasping_region")
        return (base / "clamps_grasping_regions_h"    / (size + "_clamp_grasping_regions_h.pcd")).string();
    else if (pcd_type == "v_grasping_region")
        return (base / "clamps_grasping_regions_v"    / (size + "_clamp_grasping_regions_v.pcd")).string();

    std::cerr << "Unknown pcd_type: " << pcd_type << "\n";
    return "";
}


// ---------------------------------------------------------------------------
// Helper: load a json file given its type and clamp size
// 
nlohmann::json get_json_path(const std::string& json_file_type, const std::string& clamp_size) {

    // Map "s"/"m"/"b" → "small"/"medium"/"big"  (same as before)
    std::string size;
    if      (clamp_size == "s") size = "small";
    else if (clamp_size == "m") size = "medium";
    else if (clamp_size == "b") size = "big";
    else {
        std::cerr << "Invalid clamp size: " << clamp_size << "\n";
        return {};   // return empty json object — equivalent to returning None
    }

    std::filesystem::path base = PROJECT_ROOT / "files" / "json_files_scaled";

    // Build the file path based on type
    std::filesystem::path filepath;
    if      (json_file_type == "clamp_grasping_regions_h")
        filepath = base / "clamp_grasping_regions" / (size + "_clamp_grasping_regions_h.json");
    else if (json_file_type == "clamp_grasping_regions_v")
        filepath = base / "clamp_grasping_regions" / (size + "_clamp_grasping_regions_v.json");
    else if (json_file_type == "axis_rotations")
        filepath = base / "regions_angles"          / (size + "_clamp_regions_angles.json");
    else if (json_file_type == "grasping_points")
        filepath = base / "clamp_grasping_points"   / (size + "_clamp_grasping_points.json");
    else if (json_file_type == "grasping_points_angles")
        filepath = base / "clamp_grasping_points_angles" / (size + "_clamp_grasping_points_angles.json");
    else {
        std::cerr << "Unknown json_file_type: " << json_file_type << "\n";
        return {};
    }

    // Open and parse the file
    std::ifstream file(filepath.string());  // open the file
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filepath << "\n";
        return {};
    }

    nlohmann::json data = nlohmann::json::parse(file);  // parse file
    return data;
}





// ---------------------------------------------------------------------------
// Transform a 3D coordinate using a 4x4 transformation matrix
// ---------------------------------------------------------------------------

Eigen::Vector3d transform_xyz_coordinate(const Eigen::Matrix4d& transformation,
                                         const Eigen::Vector3d& coordinate) {

    // Append 1 to make it a homogeneous 4D vector 
    Eigen::Vector4d homogeneous;
    homogeneous << coordinate.x(), coordinate.y(), coordinate.z(), 1.0;

    // Matrix multiplication — same as np.dot(transformation, column_matrix)
    Eigen::Vector4d transformed = transformation * homogeneous;

    // Return only the first 3 components — same as transformed_coord[:3].flatten()
    return transformed.head<3>();
}

// ---------------------------------------------------------------------------
// Convert position + RPY angles into a 4x4 transformation matrix
// ---------------------------------------------------------------------------


Eigen::Matrix4d pose_to_transform_matrix_rpy(const Eigen::Vector3d& position,
                                              const Eigen::Vector3d& rpy) {
    // rpy = (roll, pitch, yaw) in radians
    // equivalent to R.from_euler('xyz', rpy)
    Eigen::Matrix3d rotation_matrix =
        (Eigen::AngleAxisd(rpy.x(), Eigen::Vector3d::UnitX())  // roll  → rotate around X
       * Eigen::AngleAxisd(rpy.y(), Eigen::Vector3d::UnitY())  // pitch → rotate around Y
       * Eigen::AngleAxisd(rpy.z(), Eigen::Vector3d::UnitZ())) // yaw   → rotate around Z
        .toRotationMatrix();

    // Build the 4x4 transformation matrix

    Eigen::Matrix4d transform_matrix = Eigen::Matrix4d::Identity();  // np.eye(4)
    transform_matrix.block<3, 3>(0, 0) = rotation_matrix;           // insert rotation
    transform_matrix.block<3, 1>(0, 3) = position;                  // insert translation

    return transform_matrix;
}

// ---------------------------------------------------------------------------
// Extract pose (position + RPY angles) from a 4x4 transformation matrix
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Overload 1 — no origin 
// ---------------------------------------------------------------------------
Eigen::VectorXd extract_pose_from_transformation(const Eigen::Matrix4d& matrix) {

    // Extract translation 
    Eigen::Vector3d position = transform_xyz_coordinate(matrix, Eigen::Vector3d::Zero());

    // Extract rotation matrix from upper left 3x3 block
    Eigen::Matrix3d rotation_matrix = matrix.block<3, 3>(0, 0);

    // Convert rotation matrix to RPY angles in degrees
    Eigen::Vector3d rpy = rotation_matrix.eulerAngles(0, 1, 2) * (180.0 / EIGEN_PI);


    // Concatenate position and rpy into a single 6D vector
    Eigen::VectorXd pose(6);
    pose << position, rpy;

    return pose;
}

// ---------------------------------------------------------------------------
// Overload 2 — with origin 
// ---------------------------------------------------------------------------
Eigen::VectorXd extract_pose_from_transformation(const Eigen::Matrix4d& matrix,
                                                  const Eigen::VectorXd& origin) {

    // Extract position from origin's xyz
    Eigen::Vector3d origin_pos = origin.head<3>();
    Eigen::Vector3d position = transform_xyz_coordinate(matrix, origin_pos);

    // Extract rotation and convert to RPY in degrees
    Eigen::Matrix3d rotation_matrix = matrix.block<3, 3>(0, 0);
    Eigen::Vector3d rpy_matrix = rotation_matrix.eulerAngles(0, 1, 2) * (180.0 / EIGEN_PI);

    // Add origin's rotation to the extracted RPY
    Eigen::Vector3d origin_rpy = origin.tail<3>();
    Eigen::Vector3d rpy = origin_rpy + rpy_matrix;

    // Concatenate into 6D pose vector
    Eigen::VectorXd pose(6);
    pose << position, rpy;

    return pose;
}


