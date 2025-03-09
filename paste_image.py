import os
import random
import math
from PIL import Image

# Set the path and parameters
image_folder_1 = 'path/to/thermal/train/GT/'  
image_folder_2 = 'path/to/thermal/train/LR_x8/'  
image_folder_3 = 'path/to/visible/train/'  
output_folder_1 = './output1'  
output_folder_2 = './output2'  
output_folder_3 = './output3'  
num_images = 10000  # Number of images generated

original_size_1 = (640, 448)  
original_size_2 = (80, 56)    
original_size_3 = (640, 448)  

# Create output folder
os.makedirs(output_folder_1, exist_ok=True)
os.makedirs(output_folder_2, exist_ok=True)
os.makedirs(output_folder_3, exist_ok=True)

# Get all the image files
image_files_1 = [os.path.join(image_folder_1, f) for f in os.listdir(image_folder_1) if f.endswith('.bmp')]
image_files_2 = [os.path.join(image_folder_2, f) for f in os.listdir(image_folder_2) if f.endswith('.bmp')]
image_files_3 = [os.path.join(image_folder_3, f) for f in os.listdir(image_folder_3) if f.endswith('.bmp')]

# Build image correspondence based on the first 9 characters
image_dict_1 = {os.path.basename(f)[:9]: f for f in image_files_1}
image_dict_2 = {os.path.basename(f)[:9]: f for f in image_files_2}
image_dict_3 = {os.path.basename(f)[:9]: f for f in image_files_3}

# Ensure that the images in the three folders correspond one-to-one
common_keys = set(image_dict_1.keys()).intersection(set(image_dict_2.keys())).intersection(set(image_dict_3.keys()))
image_files_1 = [image_dict_1[key] for key in common_keys]
image_files_2 = [image_dict_2[key] for key in common_keys]
image_files_3 = [image_dict_3[key] for key in common_keys]

# Generate images
for i in range(num_images):
    # Randomly select 2-3 pictures
    num_selected = random.randint(2, 3)  
    selected_keys = random.sample(common_keys, num_selected)
    selected_images_1 = [image_dict_1[key] for key in selected_keys]
    selected_images_2 = [image_dict_2[key] for key in selected_keys]
    selected_images_3 = [image_dict_3[key] for key in selected_keys]

    # Generate transparency coefficients
    alphas = [random.uniform(0, 1) for _ in range(num_selected)]
    total_alpha = sum(alphas)
    alphas = [a / total_alpha for a in alphas] 

    # Deal with GT folder
    canvas_size_1 = (original_size_1[0] * 2, original_size_1[1] * 2)
    canvas_1 = Image.new('RGBA', canvas_size_1, (0, 0, 0, 0))  # Fully transparent canvas

    # Deal with LR folder
    canvas_size_2 = (original_size_2[0] * 2, original_size_2[1] * 2)
    canvas_2 = Image.new('RGBA', canvas_size_2, (0, 0, 0, 0))  # Fully transparent canvas

    # Deal with guide folder
    canvas_size_3 = (original_size_3[0] * 2, original_size_3[1] * 2)
    canvas_3 = Image.new('RGBA', canvas_size_3, (0, 0, 0, 0))  # Fully transparent canvas

    # Calculate the center of the canvas
    center_x = canvas_size_1[0] // 2
    center_y = canvas_size_1[1] // 2

    # Maximum distance from the center (adjust to control how close to the center)
    max_distance = 80  # Maximum distance from the center in pixels

    # Paste each image in turn
    for idx, (img_path_1, img_path_2, img_path_3) in enumerate(zip(selected_images_1, selected_images_2, selected_images_3)):
        img_1 = Image.open(img_path_1).resize(original_size_1)
        img_alpha_1 = img_1.convert("RGBA")
        img_alpha_1.putalpha(int(255 * alphas[idx]))  # Set transparency

        img_2 = Image.open(img_path_2).resize(original_size_2)
        img_alpha_2 = img_2.convert("RGBA")
        img_alpha_2.putalpha(int(255 * alphas[idx]))  # Set transparency

        img_3 = Image.open(img_path_3).resize(original_size_3)
        img_alpha_3 = img_3.convert("RGBA")
        img_alpha_3.putalpha(int(255 * alphas[idx]))  # Set transparency

        # For the first image, place its center at the center of the canvas
        if idx == 0:
            position_1_x = center_x - original_size_1[0] // 2
            position_1_y = center_y - original_size_1[1] // 2
        else:
            # For other images, generate random positions near the center
            angle = random.uniform(0, 2 * math.pi)  # Random angle
            distance = random.uniform(0, max_distance)  # Random distance within max_distance

            # Calculate position relative to the center
            offset_x = int(distance * math.cos(angle))
            offset_y = int(distance * math.sin(angle))

            # Ensure positions are within the canvas bounds
            position_1_x = max(0, min(center_x + offset_x - original_size_1[0] // 2, canvas_size_1[0] - original_size_1[0]))
            position_1_y = max(0, min(center_y + offset_y - original_size_1[1] // 2, canvas_size_1[1] - original_size_1[1]))

        position_1 = (position_1_x, position_1_y)
        position_2 = (
            int(position_1[0] * (original_size_2[0] / original_size_1[0])),  # Scale x coordinate
            int(position_1[1] * (original_size_2[1] / original_size_1[1]))   # Scale y coordinate
        )
        position_3 = position_1  # Use the same position as folder 1

        # Paste the image to the canvas
        canvas_1.paste(img_alpha_1, position_1, img_alpha_1)
        canvas_2.paste(img_alpha_2, position_2, img_alpha_2)
        canvas_3.paste(img_alpha_3, position_3, img_alpha_3)

    # Cut the center of the canvas
    left_1 = (canvas_size_1[0] - original_size_1[0]) // 2
    top_1 = (canvas_size_1[1] - original_size_1[1]) // 2
    right_1 = left_1 + original_size_1[0]
    bottom_1 = top_1 + original_size_1[1]
    cropped_canvas_1 = canvas_1.crop((left_1, top_1, right_1, bottom_1))

    left_2 = (canvas_size_2[0] - original_size_2[0]) // 2
    top_2 = (canvas_size_2[1] - original_size_2[1]) // 2
    right_2 = left_2 + original_size_2[0]
    bottom_2 = top_2 + original_size_2[1]
    cropped_canvas_2 = canvas_2.crop((left_2, top_2, right_2, bottom_2))

    left_3 = (canvas_size_3[0] - original_size_3[0]) // 2
    top_3 = (canvas_size_3[1] - original_size_3[1]) // 2
    right_3 = left_3 + original_size_3[0]
    bottom_3 = top_3 + original_size_3[1]
    cropped_canvas_3 = canvas_3.crop((left_3, top_3, right_3, bottom_3))

    # Save the generated image
    output_path_1 = os.path.join(output_folder_1, f'output_{i:05d}.bmp')
    cropped_canvas_1.convert("RGB").save(output_path_1, format="BMP")  
    print(f'Saved {output_path_1}')

    output_path_2 = os.path.join(output_folder_2, f'output_{i:05d}.bmp')
    cropped_canvas_2.convert("RGB").save(output_path_2, format="BMP")  
    print(f'Saved {output_path_2}')

    output_path_3 = os.path.join(output_folder_3, f'output_{i:05d}.bmp')
    cropped_canvas_3.convert("RGB").save(output_path_3, format="BMP") 
    print(f'Saved {output_path_3}')

print('All images generated.')