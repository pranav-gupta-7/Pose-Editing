from PIL import Image
import numpy as np

# Load the images
image_with_obj_removed = Image.open("./outputs/inpainting/rbg_image_obj_removed.jpg")
mask_image = Image.open("./outputs/inpainting/mask_image.jpg")
rotated_obj = Image.open("./outputs/zero123_output/rotated_obj.png")
segmentation_mask = Image.open("./outputs/grounding_sam_output_2/mask_image.jpg")




# mask image to a boolean mask where the chair is present
mask_array = np.array(mask_image)
mask = mask_array > 0  # Non-white pixels in the mask

# Extract the bounding box from the mask image
y_coords, x_coords = np.nonzero(mask)
y1, y2, x1, x2 = y_coords.min(), y_coords.max(), x_coords.min(), x_coords.max()

# Processing the segmentation mask to create an alpha channel
seg_mask_array = np.array(segmentation_mask)
chair_seg_mask = seg_mask_array != 0
# Creating an alpha mask where the chair pixels are 255 and the rest are 0
chair_alpha_mask = chair_seg_mask.astype(np.uint8) * 255

# Apply the chair_alpha_mask to the rotated chair image
rotated_obj_array = np.array(rotated_obj)
chair_only_array_with_alpha = np.dstack((rotated_obj_array[:, :, :3], chair_alpha_mask))

# Extracting bounding box from the alpha mask of the rotated chair
y_coords_rotated, x_coords_rotated = np.nonzero(chair_alpha_mask)
ry1, ry2, rx1, rx2 = y_coords_rotated.min(), y_coords_rotated.max(), x_coords_rotated.min(), x_coords_rotated.max()

# Cropping the rotated chair image to its bounding box
cropped_rotated_chair = chair_only_array_with_alpha[ry1:ry2, rx1:rx2]

# Converting the cropped numpy array to an image
cropped_rotated_chair_image = Image.fromarray(cropped_rotated_chair, 'RGBA')

# the new size for the cropped rotated chair to fit within the original chair's bounding box
original_aspect_ratio = (rx2 - rx1) / (ry2 - ry1)
new_width = x2 - x1
new_height = round(new_width / original_aspect_ratio)
if new_height > (y2 - y1):
    new_height = y2 - y1
    new_width = round(new_height * original_aspect_ratio)

# Resize the cropped chair while preserving aspect ratio
resized_chair = cropped_rotated_chair_image.resize((new_width, new_height), Image.LANCZOS)

# Calculate the position to paste the resized chair based on the center of the bounding box
paste_x = x1 + ((x2 - x1) - new_width) // 2
paste_y = y1 + ((y2 - y1) - new_height) // 2

# Paste the resized chair into the image with the object removed
image_with_obj_removed.paste(resized_chair, (paste_x, paste_y), resized_chair)

# Save the corrected image
corrected_final_image_path = "./final_outpupt_image.jpg"
image_with_obj_removed.save(corrected_final_image_path)
