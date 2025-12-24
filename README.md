

Clamp Segmentation and Scene Analysis Pipeline

This repository contains the complete pipeline for segmentation, pose estimation, and scene analysis of clamps in cluttered environments, developed as part of my project work.

Key Features
1. Segmentation of Clamps

    RGB Image Segmentation: Utilizes a fine-tuned SAM-2 (Segment Anything Model) for segmenting RGB images to generate high-quality masks.
    Point Cloud Segmentation: Applies generated masks to segment 3D point clouds captured by Zivid sensors, ensuring accurate spatial representation of clamps.

    ### Example of successful and failed segmentation masks

    <p align="center">
    <!-- Successful segmentations -->
    <img src="media\valid_masks\1.png" width="220">
    <img src="media\valid_masks\2.png" width="220">
    <img src="media\valid_masks\5.png" width="220">
    </p>
    

    <p align="center">
    <!-- Failed segmentations -->
    <img src="media\invalid_masks\1.png" width="220">
    <img src="media\invalid_masks\2.png" width="220">
    <img src="media\invalid_masks\3.png" width="220">
    </p>


    <p align="center">
    <em>Top: examples of successful clamp segmentations; bottom: examples of failure cases.</em>
    </p>




    Observed performance: For 10 scenes with spaced-out clamps, all clamps were detected (relevant mask ratio 1.00) and 91% of clamps had masks suitable for successful model fitting (valid mask ratio 0.91). In 23 more challenging scenes with closely packed clamps, the relevant and valid mask ratios were 0.78 and 0.70, respectively, mainly due to occlusions and entanglements.


2. RANSAC and ICP Alignment

    Clamp Size Determination: Matches segmented point clouds to predefined clamp models using RANSAC and iterative closest point (ICP) alignment.
    Pose Estimation: Computes precise clamp poses in the 3D scene, enabling downstream robotic applications.
    Validation: Filters invalid clamps based on size thresholds, fit quality, and geometric constraints.

    Pose and size accuracy: Across 23 test images with closely spaced clamps, 95% of segmented clamps had visually correct poses (average pose accuracy ratio 0.95) with a mean point-to-model offset on the order of 1 cm, and clamp size classification was correct for 90% of cases

3. Scene Representation

    Point Cloud Integration: Combines segmented clamp data with the overall scene point cloud to build a complete representation.
    Clamp Representation: Adds clamp objects into the scene with accurate size, pose, and orientation.

    ### Example of some created scene representations


    <p align="center">
    <!-- Example 1: real scene (left) + representation (right) -->
    <img src="media\scene_representations\image_11_real.png" width="320">
    <img src="media\scene_representations\image_11.png" width="320">
    </p>

    <p align="center">
    <!-- Example 2 -->
    <img src="media\scene_representations\image_12_real.png" width="320">
    <img src="media\scene_representations\image_12.png" width="320">
    </p>

    <p align="center">
    <!-- Example 3 -->
    <img src="media\scene_representations\image_23_real.png" width="320">
    <img src="media\scene_representations\image_23.png" width="320">
    </p>

    <p align="center">
    <em>Left: real Zivid point cloud scenes; right: corresponding scene representations with fitted clamp models.</em>
    </p>


    Scene-level quality: Fitted clamp models closely overlay the recorded point clouds in most test scenes, confirming that the scene representation preserves clamp size, pose, and state information sufficiently for downstream grasp planning

4. Scene Analysis and Grasp Planning

    Accessible Points Detection: Identifies graspable points and accessible areas in the scene for robotic manipulation.
    Grasp Pose Generation: Computes optimal grasping poses and visualizes potential grasp points for each clamp.

    Picking success: In 6 experimental scenes (38 clamps total), the robot executed 35 picking attempts with an overall success rate of 92%, with failures mainly caused by inaccurate pose estimates or undetected occlusions in a few challenging scenes.
   
    ### A sequence of picking attempts


    <p align="center">
    <!-- Example 1: real scene (left) + representation (right) -->
    <img src="media\gripping_sequence.png" width="320">
    </p>


    Wall avoidance in boxes: In 7 box scenes, the wall avoidance component successfully rejected grasp points too close to walls in almost all cases, with failures appearing primarily where box walls were poorly represented in the point cloud (e.g., distant or occluded walls).

Pipeline Overview

    Data Input: Captures RGB images and 3D point clouds using a Zivid sensor.
    Segmentation: Segments clamps in the RGB image and applies masks to the point cloud for 3D segmentation.
    Pose Estimation: Aligns segmented clamps with predefined models using RANSAC and ICP.
    Scene Representation: Constructs a comprehensive scene point cloud with clamp objects integrated.
    Scene Analysis: Analyzes the scene for graspable areas and potential manipulation points.

Repository Structure

    /rgb_segmentation/: Scripts for SAM2-based RGB segmentation and mask generation.
    /pcd_segmentation/: Tools for applying masks to Zivid point clouds and segmenting clamps in 3D.
    /ransac_and_icp/: RANSAC and ICP pipelines for clamp size estimation and pose determination using Open3D's library.
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
