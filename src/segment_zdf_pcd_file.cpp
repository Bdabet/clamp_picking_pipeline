#include <open3d/Open3D.h>
#include <Zivid/Zivid.h>
#include <iostream>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <filesystem>


const std::filesystem::path PROJECT_ROOT = std::filesystem::path(__FILE__)
                                           .parent_path()
                                           .parent_path();

const std::string bigClampPcdPath = (PROJECT_ROOT
                                    / "files"
                                    / "clamps_pcds"
                                    / "clamps_pcd"
                                    / "big_clamp_aligned.pcd").string();


struct PointCloudData {
    std::vector<float> xyz;   // flattened: x,y,z per point
    std::vector<uint8_t> rgba; // flattened: r,g,b,a per point
    size_t height;
    size_t width;
};

PointCloudData loadZdfPointCloud(const std::string& zdfPath, Zivid::Application* app = nullptr) {
    
    // C++ doesn't allow returning a temporary optionally-created object easily,
    // so we manage the Application lifetime outside, or use a static/shared one.
    static Zivid::Application defaultApp;
    Zivid::Application& activeApp = (app != nullptr) ? *app : defaultApp;

    Zivid::Frame frame(zdfPath);
    auto pointCloud = frame.pointCloud();

    auto xyz  = pointCloud.copyData<Zivid::PointXYZ>();
    auto rgba = pointCloud.copyData<Zivid::ColorRGBA>();

    size_t height = pointCloud.height();
    size_t width  = pointCloud.width();

    // Flatten into plain vectors for easy downstream use
    std::vector<float> xyzFlat(height * width * 3);
    std::vector<uint8_t> rgbaFlat(height * width * 4);

    for (size_t i = 0; i < height * width; ++i) {
        xyzFlat[i*3 + 0] = xyz(i).x;
        xyzFlat[i*3 + 1] = xyz(i).y;
        xyzFlat[i*3 + 2] = xyz(i).z;

        rgbaFlat[i*4 + 0] = rgba(i).r;
        rgbaFlat[i*4 + 1] = rgba(i).g;
        rgbaFlat[i*4 + 2] = rgba(i).b;
        rgbaFlat[i*4 + 3] = rgba(i).a;
    }

    return PointCloudData{ xyzFlat, rgbaFlat, height, width };
}

void applyTransformation(std::vector<float>& xyz, 
                         const std::array<std::array<float, 4>, 4>& transformMatrix) {
    size_t numPoints = xyz.size() / 3;

    for (size_t i = 0; i < numPoints; ++i) {
        float x = xyz[i*3 + 0];
        float y = xyz[i*3 + 1];
        float z = xyz[i*3 + 2];

        // Apply 4x4 matrix to homogeneous point [x, y, z, 1]
        xyz[i*3 + 0] = transformMatrix[0][0]*x + transformMatrix[0][1]*y + transformMatrix[0][2]*z + transformMatrix[0][3];
        xyz[i*3 + 1] = transformMatrix[1][0]*x + transformMatrix[1][1]*y + transformMatrix[1][2]*z + transformMatrix[1][3];
        xyz[i*3 + 2] = transformMatrix[2][0]*x + transformMatrix[2][1]*y + transformMatrix[2][2]*z + transformMatrix[2][3];
    }
}

void visualizePointCloud(const std::vector<float>& xyz,
                         const std::vector<uint8_t>& rgba) {

    size_t numPoints = xyz.size() / 3;

    std::vector<Eigen::Vector3d> points;
    std::vector<Eigen::Vector3d> colors;

    for (size_t i = 0; i < numPoints; ++i) {
        float x = xyz[i*3 + 0];
        float y = xyz[i*3 + 1];
        float z = xyz[i*3 + 2];

        // equivalent of: valid_indices = ~np.isnan(points).any(axis=1) & (points[:, 2] != 0)
        if (std::isnan(x) || std::isnan(y) || std::isnan(z) || z == 0.0f) {
            continue;
        }

        float r = rgba[i*4 + 0] / 255.0f;
        float g = rgba[i*4 + 1] / 255.0f;
        float b = rgba[i*4 + 2] / 255.0f;

        points.push_back(Eigen::Vector3d(x, y, z));
        colors.push_back(Eigen::Vector3d(r, g, b));
    }

    // Build Open3D point cloud
    auto pcd = std::make_shared<open3d::geometry::PointCloud>();
    pcd->points_ = points;
    pcd->colors_ = colors;

    open3d::visualization::DrawGeometries({pcd}, "Full Point Cloud");
}

// helper function — equivalent of np.max/min over a vector of Vector3d, per axis
std::array<double, 3> pointwiseMax(const std::vector<Eigen::Vector3d>& points) {
    std::array<double, 3> maxVals = {-std::numeric_limits<double>::infinity(),
                                     -std::numeric_limits<double>::infinity(),
                                     -std::numeric_limits<double>::infinity()};
    for (const auto& p : points) {
        maxVals[0] = std::max(maxVals[0], p[0]);
        maxVals[1] = std::max(maxVals[1], p[1]);
        maxVals[2] = std::max(maxVals[2], p[2]);
    }
    return maxVals;
}

std::array<double, 3> pointwiseMin(const std::vector<Eigen::Vector3d>& points) {
    std::array<double, 3> minVals = {std::numeric_limits<double>::infinity(),
                                      std::numeric_limits<double>::infinity(),
                                      std::numeric_limits<double>::infinity()};
    for (const auto& p : points) {
        minVals[0] = std::min(minVals[0], p[0]);
        minVals[1] = std::min(minVals[1], p[1]);
        minVals[2] = std::min(minVals[2], p[2]);
    }
    return minVals;
}

bool isMaskValid(const std::shared_ptr<open3d::geometry::PointCloud>& segmentedPcd) {

    // Statistical outlier removal
    auto [inlierPcd, ind] = segmentedPcd->RemoveStatisticalOutliers(250, 2.0);

    // Get inlier points
    const auto& inlierPoints = inlierPcd->points_;

    // Compute ranges of segmented pcd (max - min per axis)
    auto segMax = pointwiseMax(inlierPoints);
    auto segMin = pointwiseMin(inlierPoints);
    std::array<double, 3> segRanges = {segMax[0] - segMin[0],
                                       segMax[1] - segMin[1],
                                       segMax[2] - segMin[2]};

    // Load big clamp reference pcd and compute its ranges
    auto bigClampPcd = open3d::io::CreatePointCloudFromFile(bigClampPcdPath);
    const auto& bigClampPoints = bigClampPcd->points_;

    auto clampMax = pointwiseMax(bigClampPoints);
    auto clampMin = pointwiseMin(bigClampPoints);
    std::array<double, 3> clampRanges = {clampMax[0] - clampMin[0],
                                          clampMax[1] - clampMin[1],
                                          clampMax[2] - clampMin[2]};

    // Check if any axis exceeds 3x the reference range
    for (size_t i = 0; i < 3; ++i) {
        double rangeDiff = segRanges[i] - clampRanges[i];
        if ((rangeDiff / clampRanges[i]) > 3.0) {
            return false;
        }
    }
    return true;
}


void segmentPcdMasks(const std::string& zdfPath,
                     const std::string& masksDir,
                     const std::string& saveDir,
                     const std::array<std::array<float, 4>, 4>* transformMatrix = nullptr,
                     Zivid::Application* app = nullptr) {

    // Load point cloud
    auto data = loadZdfPointCloud(zdfPath, app);
    auto& xyz  = data.xyz;
    auto& rgba = data.rgba;
    size_t height = data.height;
    size_t width  = data.width;

    // Apply transformation if provided
    if (transformMatrix != nullptr) {
        applyTransformation(xyz, *transformMatrix);
    }

    // Iterate over mask files in the directory
    for (const auto& entry : std::filesystem::directory_iterator(masksDir)) {
        std::string maskPath     = entry.path().string();
        std::string maskFilename = entry.path().filename().string();

        // Load and resize mask
        cv::Mat mask = cv::imread(maskPath, cv::IMREAD_GRAYSCALE);
        cv::resize(mask, mask, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);

        std::vector<Eigen::Vector3d> maskPoints;
        std::vector<Eigen::Vector3d> maskColors;

        for (size_t v = 0; v < height; ++v) {
            for (size_t u = 0; u < width; ++u) {
                if (mask.at<uint8_t>(v, u) == 0) continue;

                size_t idx = v * width + u;
                float z = xyz[idx*3 + 2];

                if (std::isnan(z) || z == 0.0f) continue;

                float x = xyz[idx*3 + 0];
                float y = xyz[idx*3 + 1];

                float r = rgba[idx*4 + 0] / 255.0f;
                float g = rgba[idx*4 + 1] / 255.0f;
                float b = rgba[idx*4 + 2] / 255.0f;

                maskPoints.push_back(Eigen::Vector3d(x, y, z));
                maskColors.push_back(Eigen::Vector3d(r, g, b));
            }
        }

        if (maskPoints.empty()) {
            std::cout << "No valid points found in mask " << maskFilename << "\n";
            continue;
        }

        auto maskPcd = std::make_shared<open3d::geometry::PointCloud>();
        maskPcd->points_ = maskPoints;
        maskPcd->colors_ = maskColors;

        if (isMaskValid(maskPcd)) {
            // Replace .png with .pcd
            std::string saveName = maskFilename;
            saveName.replace(saveName.find(".png"), 4, ".pcd");
            std::string savePath = saveDir + "/" + saveName;

            open3d::io::WritePointCloud(savePath, *maskPcd);
            std::cout << "Saved segmented point cloud to " << savePath << "\n";
        } else {
            std::cout << maskFilename << " is not valid\n";
        }
    }
}

std::shared_ptr<open3d::geometry::PointCloud> createPcdExcludingMasks(
    const std::string& zdfPath,
    const std::string& masksDir,
    const std::map<std::string, std::map<std::string, std::string>>& clampsData,
    const std::string& savePath = "",
    const std::array<std::array<float, 4>, 4>* transformMatrix = nullptr,
    Zivid::Application* app = nullptr) {

    // Load point cloud
    auto data = loadZdfPointCloud(zdfPath, app);
    auto& xyz  = data.xyz;
    auto& rgba = data.rgba;
    size_t height = data.height;
    size_t width  = data.width;

    // Build exclusion mask — flat 2D bool grid
    std::vector<bool> exclusionMask(height * width, false);

    // Extract relevant mask names from clamps_data (strip extension)
    std::vector<std::string> relevantMasks;
    for (const auto& [key, clampData] : clampsData) {
        std::string maskName = clampData.at("mask name");
        relevantMasks.push_back(maskName.substr(0, maskName.size() - 4)); // strip .png
    }

    // Iterate masks directory and combine into exclusion mask
    for (const auto& entry : std::filesystem::directory_iterator(masksDir)) {
        std::string maskFilename = entry.path().filename().string();
        std::string maskStem     = maskFilename.substr(0, maskFilename.size() - 4);

        // equivalent of: if mask_filename[:-4] in relevant_masks
        bool isRelevant = std::find(relevantMasks.begin(), relevantMasks.end(), maskStem) 
                          != relevantMasks.end();
        if (!isRelevant) continue;

        cv::Mat mask = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
        cv::resize(mask, mask, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);

        for (size_t v = 0; v < height; ++v) {
            for (size_t u = 0; u < width; ++u) {
                if (mask.at<uint8_t>(v, u) > 0) {
                    exclusionMask[v * width + u] = true;
                }
            }
        }
    }

    // Collect points NOT covered by any mask
    std::vector<Eigen::Vector3d> remainingPoints;
    std::vector<Eigen::Vector3d> remainingColors;

    for (size_t v = 0; v < height; ++v) {
        for (size_t u = 0; u < width; ++u) {
            if (exclusionMask[v * width + u]) continue;

            size_t idx = v * width + u;
            float z = xyz[idx*3 + 2];
            if (std::isnan(z) || z == 0.0f) continue;

            float x = xyz[idx*3 + 0];
            float y = xyz[idx*3 + 1];

            float r = rgba[idx*4 + 0] / 255.0f;
            float g = rgba[idx*4 + 1] / 255.0f;
            float b = rgba[idx*4 + 2] / 255.0f;

            // scale to meters (multiply by 0.001)
            remainingPoints.push_back(Eigen::Vector3d(x, y, z) * 0.001);
            remainingColors.push_back(Eigen::Vector3d(r, g, b));
        }
    }

    if (remainingPoints.empty()) {
        std::cout << "No valid points found after excluding masks.\n";
        return nullptr;
    }

    auto remainingPcd = std::make_shared<open3d::geometry::PointCloud>();
    remainingPcd->points_ = remainingPoints;
    remainingPcd->colors_ = remainingColors;

    // Apply transformation if provided
    if (transformMatrix != nullptr) {
        Eigen::Matrix4d eigenTransform;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                eigenTransform(i, j) = (*transformMatrix)[i][j];
        remainingPcd->Transform(eigenTransform);
    }

    // Save if path provided
    if (!savePath.empty()) {
        open3d::io::WritePointCloud(savePath, *remainingPcd);
    }

    // Downsample and remove outliers
    auto downsampledPcd = remainingPcd->VoxelDownSample(0.001);
    auto [cleanPcd, ind] = downsampledPcd->RemoveStatisticalOutliers(70, 2.0);

    // displayInlierOutlier(*downsampledPcd, ind);

    open3d::visualization::DrawGeometries({remainingPcd});

    return cleanPcd;
}






int main() {


    return 0;
}