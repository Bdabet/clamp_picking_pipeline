import numpy as np
import torch
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import matplotlib.pyplot as plt
import os
import zivid

# to adjust remove mask references
# add cuda changes



root_path = r"C:\Users\Bahaeddine\Desktop\TUHH\Semester 4\picking_process"
sam2_checkpoint = os.path.join(root_path, "files", "sam_running_files", "sam2_hiera_small.pt")
sam2_weights = os.path.join(root_path, "files", "sam_running_files", "model_4.torch") # model_6.torch could also be tested
save_dir = os.path.join(root_path, "files", "generated_png_masks")
model_cfg = "sam2_hiera_s.yaml"



def convert_zdf(file_path, img_output_path, pcd_output_path, app = None) -> None:
    """Convert a ZDF file to a specified format (PLY or 2D image).

    Args:
        file_path: Path to the ZDF file.
        output_path: Path to save the converted file.
        output_format: Format to convert to ("ply" or "2d").

    Raises:
        ValueError: If an unsupported format is specified.

    """
    if app is None:
        app = zivid.Application()
    
    print(f"Reading point cloud from file: {file_path}")
    frame = zivid.Frame(file_path)

    print(f"Saving the frame to {pcd_output_path}")
    frame.save(pcd_output_path)

    point_cloud = frame.point_cloud()
    print(f"Saving the frame to {img_output_path}")
    bgra = point_cloud.copy_data("bgra")
    cv2.imwrite(img_output_path, bgra[:, :, :3])
    
       
def get_points(mask, num_points):
    points = []
    height, width = mask.shape
    y_coords = np.linspace(0, height - 1, int(np.sqrt(num_points)))
    x_coords = np.linspace(0, width - 1, int(np.sqrt(num_points)))
    for y in y_coords:
        for x in x_coords:
            points.append([[x, y]])
            if len(points) >= num_points:
                return np.array(points)
    return np.array(points)

def crop_and_segment(image_path, crop_box = None, num_samples=250, max_size=1024, display_image = True, save = True):
    
    
    img = cv2.imread(image_path)[..., ::-1]  # Read img as RGB
    mask = cv2.imread(image_path, 0)  # Mask of the region we want to segment

    # Resize img to a maximum size of 1024
    r = np.min([max_size / img.shape[1], max_size / img.shape[0]])
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)), interpolation=cv2.INTER_NEAREST)


    
    
    if crop_box is not None:
        left, upper, right, lower = crop_box
        cropped_image = img[upper:lower, left:right]
        cropped_mask = mask[upper:lower, left:right]
    else:
        cropped_image = img
        cropped_mask = mask
    
    if display_image == True:
        plt.imshow(cropped_image)
        plt.show()

    # Generate points within the cropped mask
    input_points = get_points(cropped_mask, num_samples)

    # Load SAM2 model and predictor
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cpu")
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.model.load_state_dict(torch.load(sam2_weights, map_location="cpu"))

    # Predict masks
    with torch.no_grad():
        predictor.set_image(cropped_image)
        masks, scores, _ = predictor.predict(
            point_coords=input_points,
            point_labels=np.ones([input_points.shape[0], 1])
        )

    # Define color threshold for white
    white_threshold = 100

    # Sort and filter masks by score and white color content
    sorted_masks = masks[:, 0][np.argsort(scores[:, 0])][::-1].astype(bool)
    seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)
    occupancy_mask = np.zeros_like(sorted_masks[0], dtype=bool)
    
    for i, mask in enumerate(sorted_masks):
       
        masked_pixels = cropped_image[mask]  # Extract pixels covered by the mask
        

        # Calculate the number of pixels exceeding the white threshold
        white_like_pixels = (masked_pixels >= white_threshold).all(axis=1)
        
        white_like_count = np.sum(white_like_pixels)
        total_nonzero_pixels = np.count_nonzero(masked_pixels)

        # Check if the majority of non-zero pixels exceed the threshold
        if white_like_count / total_nonzero_pixels <= 0.05:  # Skip if 5% or fewer are white-like
            continue

        # Combine mask into segmentation map
        if (mask * occupancy_mask).sum() / mask.sum() > 0.15:
            continue
        mask[occupancy_mask] = 0
        seg_map[mask] = i + 1
        occupancy_mask[mask] = 1

        original_size_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        
        if crop_box is not None:
            original_size_mask[upper:lower, left:right] = mask
        else:
            original_size_mask = mask
        # Save the mask as an img file
        mask_filename = os.path.join(save_dir, f"mask_{i+1}.png")
        mask_image = (original_size_mask.astype(np.uint8) * 255)  # Convert mask to binary img (0 or 255)
        if save == True:
            cv2.imwrite(mask_filename, mask_image)

        
       

    # Place the combined segmentation back into the original image’s coordinates
    full_seg_map = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    if crop_box is not None:    
        full_seg_map[upper:lower, left:right] = seg_map
    else:
        full_seg_map = seg_map
    
    

    # Create a colored segmentation map for visualization
    rgb_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for id_class in range(1, full_seg_map.max() + 1):
        rgb_image[full_seg_map == id_class] = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]

    
    if display_image == True:
        cv2.imshow("annotation", rgb_image)
        cv2.imshow("mix", (rgb_image / 2 + img / 2).astype(np.uint8))
        cv2.imshow("original_image", img)
        cv2.waitKey()
    

    return rgb_image,img

if __name__ == "__main__": 

    # testing 
    
    image_path = r"C:\Users\Bahaeddine\Desktop\TUHH\Semester 4\picking_process\files\images\two_entangled_2.png"
    
    crop_box = (420, 150, 730, 500)  # Define crop box (left, upper, right, lower)
    num_samples = 500

    crop_and_segment(image_path, display_image=True,  save = True, crop_box= crop_box)

