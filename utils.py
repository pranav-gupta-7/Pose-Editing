import os
import numpy as np
from PIL import Image
import cv2

def maskout_segmented_obj():
    path = './outputs/grounding_sam_output/'

    mask_image = Image.open(path + 'mask_image.jpg')
    raw_image = Image.open(path + 'raw_image.jpg')

    mask_image = np.array(mask_image)
    raw_image = np.array(raw_image)

    mask_image = mask_image.astype(np.uint8)
    # kernel = np.ones((3, 3), np.uint8) 
    # mask_image = cv2.dilate(mask_image, kernel, iterations=1) 

    mask_image[np.where(mask_image!=0)] = 1


    masked_rgb = mask_image[..., None] * raw_image


    masked_rgb = Image.fromarray(masked_rgb)
    masked_rgb.save(path + 'masked_rgb.jpg')