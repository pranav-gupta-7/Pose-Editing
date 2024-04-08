from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torch
import numpy as np
import cv2


image = Image.open("./outputs/grounding_sam_output/raw_image.jpg")
mask_image = Image.open('./outputs/grounding_sam_output/mask_image.jpg')

#get size of mask_image
width, height = mask_image.size

mask_image = np.array(mask_image)
mask_image = mask_image.astype(np.uint8)
kernel = np.ones((5, 5), np.uint8) 
mask_image = cv2.dilate(mask_image, kernel, iterations=5) 
mask_image[np.where(mask_image != 0)] = 255
mask_image = Image.fromarray(mask_image.astype(np.uint8))
mask_image.save('./outputs/inpainting/mask_image.jpg')

prompt = ""



pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float32,
)
pipe.to("cuda")
#image and mask_image should be PIL images.
#The mask structure is white for inpainting and black for keeping as is
image = pipe(prompt='', image=image, mask_image=mask_image).images[0]

image = image.resize((width, height))

image.save("./outputs/inpainting/rbg_image_obj_removed.jpg")
