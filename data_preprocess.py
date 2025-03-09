import os
import cv2
import numpy as np

# Data augmentation function
def data_augmentation(image):
    augmented_images = []
    # Horizontal flip
    flipped_horizontal = cv2.flip(image, 1)
    augmented_images.append(("flipped_horizontal", flipped_horizontal))
    # Vertical flip
    flipped_vertical = cv2.flip(image, 0)
    augmented_images.append(("flipped_vertical", flipped_vertical))
    # Add noise
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)  # Gaussian noise
    noisy_image = cv2.add(image, noise)
    augmented_images.append(("noisy", noisy_image))
    return augmented_images

# GT multi-scale transformation function
def gt_multiscale_transform(image, scales=[1.25, 1.5, 2, 2.5]):
    transformed_images = []
    h, w, _ = image.shape
    target_h, target_w = 448, 640  # Target dimensions
    for scale in scales:
        # Calculate the scaled dimensions
        new_h, new_w = int(h * scale), int(w * scale)
        # Resize the image
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # Crop the center region
        start_x_center = (new_w - target_w) // 2
        start_y_center = (new_h - target_h) // 2
        center_crop = resized_image[start_y_center:start_y_center + target_h, start_x_center:start_x_center + target_w]
        transformed_images.append((f"scale_{scale}_center", center_crop))
        # Crop the top-left region
        top_left_crop = resized_image[0:target_h, 0:target_w]
        transformed_images.append((f"scale_{scale}_top_left", top_left_crop))
        # Crop the top-right region
        top_right_crop = resized_image[0:target_h, new_w - target_w:new_w]
        transformed_images.append((f"scale_{scale}_top_right", top_right_crop))
        # Crop the bottom-left region
        bottom_left_crop = resized_image[new_h - target_h:new_h, 0:target_w]
        transformed_images.append((f"scale_{scale}_bottom_left", bottom_left_crop))
        # Crop the bottom-right region
        bottom_right_crop = resized_image[new_h - target_h:new_h, new_w - target_w:new_w]
        transformed_images.append((f"scale_{scale}_bottom_right", bottom_right_crop))
    return transformed_images

# Process all images in the folder
def process_images_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".bmp"):
            # Read the image
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            # Resize the image to 640x448
            image = cv2.resize(image, (640, 448), interpolation=cv2.INTER_LINEAR)
            # Get the base name of the image (without extension)
            base_name = os.path.splitext(filename)[0]

            # Data augmentation
            augmented_images = data_augmentation(image)
            for aug_name, aug_image in augmented_images:
                output_path = os.path.join(output_folder, f"{base_name}_{aug_name}.bmp")
                cv2.imwrite(output_path, aug_image)

            # GT multi-scale transformation
            transformed_images = gt_multiscale_transform(image)
            for region_name, transformed_image in transformed_images:
                output_path = os.path.join(output_folder, f"{base_name}_{region_name}.bmp")
                cv2.imwrite(output_path, transformed_image)

# Set input and output folder paths
input_folder = "path/to/train/data"
output_folder = "path/to/output"

# Process images
process_images_in_folder(input_folder, output_folder)