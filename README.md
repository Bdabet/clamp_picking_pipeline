This repository contains the complete pipeline for segmentation, pose estimation, and scene analysis of clamps in cluttered environments, developed as part of my master's thesis.

Clamp Segmentation and Scene Analysis Pipeline

This repository contains the complete pipeline for segmentation, pose estimation, and scene analysis of clamps in cluttered environments, developed as part of my master's thesis. The project integrates state-of-the-art segmentation techniques, 3D point cloud processing, and advanced alignment algorithms to enable precise identification and representation of clamps in real-world or synthetic scenarios.
Key Features
1. Segmentation of Clamps

    RGB Image Segmentation: Utilizes a fine-tuned SAM2 (Segment Anything Model) for segmenting RGB images to generate high-quality masks.
    Point Cloud Segmentation: Applies generated masks to segment 3D point clouds captured by Zivid sensors, ensuring accurate spatial representation of clamps.

2. RANSAC and ICP Alignment

    Clamp Size Determination: Matches segmented point clouds to predefined clamp models using RANSAC and iterative closest point (ICP) alignment.
    Pose Estimation: Computes precise clamp poses in the 3D scene, enabling downstream robotic applications.
    Validation: Filters invalid clamps based on size thresholds, fit quality, and geometric constraints.

3. Scene Representation

    Point Cloud Integration: Combines segmented clamp data with the overall scene point cloud to build a complete representation.
    Clamp Representation: Adds clamp objects into the scene with accurate size, pose, and orientation.

4. Scene Analysis and Grasp Planning

    Accessible Points Detection: Identifies graspable points and accessible areas in the scene for robotic manipulation.
    Grasp Pose Generation: Computes optimal grasping poses and visualizes potential grasp points for each clamp.

Pipeline Overview

    Data Input: Captures RGB images and 3D point clouds using a Zivid sensor.
    Segmentation: Segments clamps in the RGB image and applies masks to the point cloud for 3D segmentation.
    Pose Estimation: Aligns segmented clamps with predefined models using RANSAC and ICP.
    Scene Representation: Constructs a comprehensive scene point cloud with clamp objects integrated.
    Scene Analysis: Analyzes the scene for graspable areas and potential manipulation points.

Repository Structure

    /rgb_segmentation/: Scripts for SAM2-based RGB segmentation and mask generation.
    /pcd_segmentation/: Tools for applying masks to Zivid point clouds and segmenting clamps in 3D.
    /ransac_and_icp/: RANSAC and ICP pipelines for clamp size estimation and pose determination.
    /scene_representation/: Functions for integrating clamps into the scene point cloud.
    /scene_analysis/: Methods for analyzing the scene and generating grasping poses.
    /clamp_picking/: Tools for visualizing grasping points and aligning clamps for manipulation.
    /config.py: Configuration file specifying root paths and parameters.

Requirements

    Python 3.9+
    Open3D
    PyTorch
    Zivid SDK
    NumPy, SciPy


Instructions for Setting Up and Running the Pipeline

    Download Required Files:
        Download the necessary files from the provided link.

    Place Files in the Project Folder:
        Insert the entire downloaded folder into the main root directory of the project, which should be located at \picking_pipeline.

    Update Configuration Path:
        Open config.py and set the root_path variable to the actual path of your project root directory. This ensures the pipeline uses the correct directory structure.

    Prepare Zivid Camera Scene for Processing:
        To run the pipeline with a scene captured using a Zivid camera (in .zdf format):
            Place the .zdf file inside the \files\zdf_files directory.
            Open main.py and locate the variable captured_scene_name.
            Change the value of captured_scene_name to match the name of the .zdf file without the prefix. For example, if the file is scene_001.zdf, set captured_scene_name = "scene_001".
