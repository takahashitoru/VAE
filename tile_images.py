import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import nlopt
import math

import myvae

###############################################################
# Directory where VAE model is saved
###############################################################
#dir_ptfile = "./save_caribou_bs200_ep200_lr-3"
#dir_ptfile = "./save_nessie_bs200_ep200_lr-3"
dir_ptfile = "./save_cat_bs200_ep200_lr-3"

# Use double precision
torch.set_default_dtype(torch.float64) # torch.float32 by default

# Load the VAE model
state = torch.load(os.path.join(dir_ptfile, "model.pt"))
image_size = state['image_size']
z_dim = state['z_dim']
print("image_size=", image_size)
print("z_dim=", z_dim)

if image_size == 64:
    modelA = myvae.VAE64(z_dim)
else:
    raise ValueError(f"Unregistered image_size={image_size}. Exit.")

modelA.load_state_dict(state['model_state_dict'])
print("loaded")

modelA.eval() # Necessary for Dropout and Batch Normalization

# Set grids

z1_low = -2
z1_high = 2
z2_low = -1
z2_high = 3

z1_low = -3
z1_high = 5
z2_low = -3
z2_high = 2

z1_low = -5
z1_high = 4
z2_low = -4
z2_high = 3

"""
z1_low = -20
z1_high = 20
z2_low = -20
z2_high = 20
"""

"""
z1_low = -10
z1_high = 10
z2_low = -10
z2_high = 10
"""

# Output filename
output_file = os.path.join(dir_ptfile, f"tile_{z1_low}_{z1_high}_{z2_low}_{z2_high}.eps")
output_png_file = os.path.join(dir_ptfile, f"tile_{z1_low}_{z1_high}_{z2_low}_{z2_high}.png")

image_list = []
num_cols = z1_high - z1_low + 1
num_rows = z2_high - z2_low + 1

for z2 in range(z2_high, z2_low - 1, -1):
    row_images = []
    for z1 in range(z1_low, z1_high + 1):
        z_np = np.array([z1, z2], dtype=np.double)
        z_tensor = torch.tensor(z_np, requires_grad=False)
        png_path = os.path.join(dir_ptfile, f"tile_tmp_{z1}_{z2}.png")
        with torch.no_grad():
            modelA.z2png(z_tensor, png_path)
            row_images.append(Image.open(png_path).convert("RGB"))
    image_list.append(row_images)

total_width = image_size * num_cols
total_height = image_size * num_rows
padding_left = 64
padding_bottom = 64
grid_image = Image.new('RGB', (total_width + padding_left, total_height + padding_bottom), color='white')
draw = ImageDraw.Draw(grid_image)

font_size = 32
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
except IOError:
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
        print(f"DejaVuSans.ttf and LiberationSans-Regular.ttf were not found. The default font will be used (impossible to change the font size)")

        
for i, row in enumerate(image_list):
    for j, img in enumerate(row):
        grid_image.paste(img, (j * image_size + padding_left, i * image_size))

for i, z2_val in enumerate(range(z2_high, z2_low - 1, -1)):
    z2_text = str(z2_val)
    z2_text_width = draw.textlength(z2_text, font=font)
    text_x = padding_left // 2 - z2_text_width // 2
    text_y = i * image_size + image_size // 2
    draw.text((text_x, text_y), z2_text, fill=(0, 0, 0), font=font, anchor="mm")

for j, z1 in enumerate(range(z1_low, z1_high + 1)):
    z1_text = str(z1)
    z1_text_width = draw.textlength(z1_text, font=font)
    text_x = j * image_size + image_size // 2 + padding_left
    text_y = total_height + padding_bottom // 2
    draw.text((text_x, text_y), z1_text, fill=(0, 0, 0), font=font, anchor="mm")

grid_image.save(output_png_file)
print(f"Saved the grid image to {output_png_file}.")

try:
    grid_image.save(output_file, "eps")
    print(f"Saved the grid image to {output_file}.")
except ValueError as e:
    print(f"Failed to save the grid imagen in EPS format: {e}")

for z2 in range(z2_low, z2_high + 1):
    for z1 in range(z1_low, z1_high + 1):
        png_path = os.path.join(dir_ptfile, f"tile_tmp_{z1}_{z2}.png")
        if os.path.exists(png_path):
            os.remove(png_path)

print("Done.")
