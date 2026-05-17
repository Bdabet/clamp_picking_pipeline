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



// ---------------------------------------------------------------------------
// Clamp class
// ---------------------------------------------------------------------------

class Clamp {
public:
    // -----------------------------------------------------------------------
    // Static attributes
    // -----------------------------------------------------------------------
    static int number_of_clamps;
    // predefined clamp centers 
    static const std::map<std::string, std::vector<double>> CLAMP_CENTERS;

    // -----------------------------------------------------------------------
    // Instance attributes
    // -----------------------------------------------------------------------

    // basic attributes
    std::string clamp_size;           // "s", "m", or "b"
    std::string clamp_orientation;    // "h" or "v"
    int clamp_number;
    std::string clamp_color;          

    // transformation and pose
    Eigen::Matrix4d clamp_transformation;
    Eigen::VectorXd clamp_pose;       // 6D vector [x, y, z, roll, pitch, yaw]

    // clamp center
    std::vector<double> clamp_center;                          
    std::shared_ptr<open3d::geometry::PointCloud> clamp_center_pcd;

    // point clouds 
    std::shared_ptr<open3d::geometry::PointCloud> clamp_pcd;
    std::shared_ptr<open3d::geometry::PointCloud> bounding_volume_pcd;
    std::shared_ptr<open3d::geometry::PointCloud> bounding_box_pcd;
    std::shared_ptr<open3d::geometry::PointCloud> bounding_box_extended_pcd;
    std::shared_ptr<open3d::geometry::PointCloud> clamp_curve_pcd;

    // json data 
    nlohmann::json h_grasping_regions;
    nlohmann::json v_grasping_regions;
    nlohmann::json grasping_axis_rotations;
    nlohmann::json grasping_points;
    nlohmann::json grasping_points_angles;



    // -----------------------------------------------------------------------
    // Constructor 
    // -----------------------------------------------------------------------

    Clamp(const std::string& size,
          std::optional<Eigen::VectorXd> pose = std::nullopt,
          std::optional<Eigen::Matrix4d> combined_transformation = std::nullopt) {

        // must provide either pose or transformation
        if (!pose.has_value() && !combined_transformation.has_value()) {
            throw std::invalid_argument("Either pose or transformation must be provided.");
        }

        // set clamp size
        clamp_size = size;

        // equivalent to Python's if/elif block
        if (combined_transformation.has_value()) {
            clamp_transformation = combined_transformation.value();
            clamp_pose = extract_pose_from_transformation(clamp_transformation);
        } 
        else if (pose.has_value()) {
            Eigen::Vector3d position = pose.value().head<3>();
            Eigen::Vector3d rpy      = pose.value().tail<3>();
            clamp_transformation     = pose_to_transform_matrix_rpy(position, rpy);
            clamp_pose               = pose.value();
        }

        // find clamp orientation
        clamp_orientation = find_clamp_orientation();

        // increment number of clamps
        number_of_clamps++;
        clamp_number = number_of_clamps;

        // load point clouds from disk
        // equivalent to: self.clamp_pcd = o3d.io.read_point_cloud(...)
        clamp_pcd                = std::make_shared<open3d::geometry::PointCloud>();
        bounding_volume_pcd      = std::make_shared<open3d::geometry::PointCloud>();
        bounding_box_pcd         = std::make_shared<open3d::geometry::PointCloud>();
        bounding_box_extended_pcd= std::make_shared<open3d::geometry::PointCloud>();
        clamp_curve_pcd          = std::make_shared<open3d::geometry::PointCloud>();

        open3d::io::ReadPointCloud(get_pcd_path("clamp",                  size), *clamp_pcd);
        open3d::io::ReadPointCloud(get_pcd_path("bounding_box",           size), *bounding_volume_pcd);
        open3d::io::ReadPointCloud(get_pcd_path("bounding_box_extended",  size), *bounding_box_extended_pcd);
        open3d::io::ReadPointCloud(get_pcd_path("inner_bounding_volume",  size), *bounding_volume_pcd);
        open3d::io::ReadPointCloud(get_pcd_path("clamp_curve",            size), *clamp_curve_pcd);

        // load json files
        h_grasping_regions     = get_json_path("clamp_grasping_regions_h", size);
        v_grasping_regions     = get_json_path("clamp_grasping_regions_v", size);
        grasping_axis_rotations= get_json_path("axis_rotations",           size);
        grasping_points        = get_json_path("grasping_points",          size);
        grasping_points_angles = get_json_path("grasping_points_angles",   size);

        // initialize clamp center
        clamp_center_pcd = initialize_center();
        clamp_center     = CLAMP_CENTERS.at(size);  
        clamp_color      = "";                      
    }

    private:

    // -----------------------------------------------------------------------
    // internal helpers
    // -----------------------------------------------------------------------

    std::string find_clamp_orientation() {

        std::string orientation = "h";  // default

        // two iterations for x and y axes — same as Python's for idx in range(2)
        for (int idx = 0; idx < 2; idx++) {

            // extract the angle for x (idx=0) or y (idx=1) axis, fmod => float remainder
            double axis_angle = std::fmod(std::abs(clamp_pose(idx + 3)), 360.0);


            if ((axis_angle >= 315 && axis_angle <= 360) ||
                (axis_angle >=   0 && axis_angle <=  45) ||
                (axis_angle >= 135 && axis_angle <= 225)) {
                orientation = "h";
            } 
            else {
                orientation = "v";
                break;  
            }
        }

        return orientation;
    }


    std::shared_ptr<open3d::geometry::PointCloud> initialize_center() {

        // check size is valid
        if (CLAMP_CENTERS.find(clamp_size) == CLAMP_CENTERS.end()) {
            throw std::invalid_argument("Invalid clamp size: " + clamp_size);
        }


        // create a new point cloud with just the center point
        auto center_pcd = std::make_shared<open3d::geometry::PointCloud>();

        // get the center coordinates for this clamp size
        std::vector<double> center = CLAMP_CENTERS.at(clamp_size);

        // add the center point to the point cloud
        center_pcd->points_.push_back(Eigen::Vector3d(center[0], center[1], center[2]));


        return center_pcd;
    }


    public:
    // -----------------------------------------------------------------------
    // public fucntions
    // -----------------------------------------------------------------------

    void add_clamp_to_representation() {


        if (clamp_transformation.isZero()) {
            std::cerr << "Transformation matrix is zero, cannot add clamp\n";
            return;
        }


        // transform all point clouds
        clamp_pcd->Transform(clamp_transformation);
        clamp_curve_pcd->Transform(clamp_transformation);
        bounding_volume_pcd->Transform(clamp_transformation);
        bounding_box_pcd->Transform(clamp_transformation);
        bounding_box_extended_pcd->Transform(clamp_transformation);
        clamp_center_pcd->Transform(clamp_transformation);

        // transform clamp center coordinate
        Eigen::Vector3d center_eigen(clamp_center[0], clamp_center[1], clamp_center[2]);
        Eigen::Vector3d transformed_center = transform_xyz_coordinate(clamp_transformation, center_eigen);
        clamp_center = {transformed_center.x(), transformed_center.y(), transformed_center.z()};

        // transform h_grasping_regions

        for (auto& [region_name, region_points] : h_grasping_regions.items()) {


            std::vector<Eigen::Vector3d> transformed_points;

            for (auto& coordinate : region_points) {
                Eigen::Vector3d coord(coordinate[0], coordinate[1], coordinate[2]);
                Eigen::Vector3d new_coord = transform_xyz_coordinate(clamp_transformation, coord);
                transformed_points.push_back(new_coord);
            }

            // update the json with transformed coordinates
            region_points = nlohmann::json::array();
            for (const auto& p : transformed_points) {
                region_points.push_back({p.x(), p.y(), p.z()});
            }
        }

        // same for v_grasping_regions
        for (auto& [region_name, region_points] : v_grasping_regions.items()) {
            std::vector<Eigen::Vector3d> transformed_points;

            for (auto& coordinate : region_points) {
                Eigen::Vector3d coord(coordinate[0], coordinate[1], coordinate[2]);
                Eigen::Vector3d new_coord = transform_xyz_coordinate(clamp_transformation, coord);
                transformed_points.push_back(new_coord);
            }

            region_points = nlohmann::json::array();
            for (const auto& p : transformed_points) {
                region_points.push_back({p.x(), p.y(), p.z()});
            }
        }

        // transform grasping points
        for (auto& [point_name, point] : grasping_points.items()) {
            Eigen::Vector3d coord(point[0], point[1], point[2]);
            Eigen::Vector3d new_coord = transform_xyz_coordinate(clamp_transformation, coord);
            point = {new_coord.x(), new_coord.y(), new_coord.z()};
        }

        // add random color to clamp

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist(50, 100);

        double r = dist(gen) / 100.0;
        double g = dist(gen) / 100.0;
        double b = dist(gen) / 100.0;

        // set uniform color on clamp pcd

        clamp_pcd->PaintUniformColor(Eigen::Vector3d(r, g, b));


        // add clamp to scene_pcd
        *scene_pcd += *clamp_pcd;

    }


};

// ---------------------------------------------------------------------------
// Static members of Clamps class
// ---------------------------------------------------------------------------
int Clamp::number_of_clamps = 0;

const std::map<std::string, std::vector<double>> Clamp::CLAMP_CENTERS = {
    {"s", {5.74323225/1000, 13.14538544/1000, 0.25742388/1000}},
    {"m", {11.8132830/1000, 19.5173681/1000,  8.585e-06/1000}},
    {"b", {20.1405449/1000, 27.1667296/1000,  2.7179718e-05/1000}}
};