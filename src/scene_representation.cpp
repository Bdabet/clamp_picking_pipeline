#include <filesystem>
#include <open3d/Open3D.h>
#include <iostream>
#include <Eigen/Dense> 



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

    // root_path would come from your config — hardcode or pass it in for now
    std::filesystem::path base = PROJECT_ROOT / "files" / "clamps_pcds_scaled";

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
// Transform a 3D coordinate using a 4x4 transformation matrix
// ---------------------------------------------------------------------------

Eigen::Vector3d transform_xyz_coordinate(const Eigen::Matrix4d& transformation,
                                         const Eigen::Vector3d& coordinate) {

    // Append 1 to make it a homogeneous 4D vector — same as np.append(coordinate, 1)
    Eigen::Vector4d homogeneous;
    homogeneous << coordinate.x(), coordinate.y(), coordinate.z(), 1.0;

    // Matrix multiplication — same as np.dot(transformation, column_matrix)
    Eigen::Vector4d transformed = transformation * homogeneous;

    // Return only the first 3 components — same as transformed_coord[:3].flatten()
    return transformed.head<3>();
}