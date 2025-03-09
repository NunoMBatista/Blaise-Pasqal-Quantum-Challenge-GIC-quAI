import os
from tqdm import tqdm
import random
import numpy as np
from PIL import Image, ImageDraw
import pandas as pd

# Constants
IMAGE_SIZE = 4  # 8x8 pixels
NUM_IMAGES = 512  # Number of images to generate
OUTPUT_DIR = "dataset"  # Base output directory
ELLIPSE_DIR = os.path.join(OUTPUT_DIR, "with_ellipse")  # Subdirectory for images with ellipses
NO_ELLIPSE_DIR = os.path.join(OUTPUT_DIR, "without_ellipse")  # Subdirectory for images without ellipses
LABELS_FILE = os.path.join(OUTPUT_DIR, "labels.csv")  # Labels file path
GRAYSCALE_LEVELS = 8  # Number of grayscale levels

# Create output directories if they don't exist
os.makedirs(ELLIPSE_DIR, exist_ok=True)
os.makedirs(NO_ELLIPSE_DIR, exist_ok=True)

def create_smooth_gradient_background(size):
    """Create a smooth random gradient background."""
    corners = np.random.randint(0, GRAYSCALE_LEVELS, size=4) * (255 // (GRAYSCALE_LEVELS - 1))
    y, x = np.mgrid[0:size, 0:size] / (size - 1)
    top = corners[0] * (1 - x) + corners[1] * x
    bottom = corners[2] * (1 - x) + corners[3] * x
    gradient = top * (1 - y) + bottom * y
    return np.round(gradient).astype(np.uint8)

def add_ellipse(img_array, size):
    """Add an ellipse to the image."""
    temp_img = Image.fromarray(img_array, 'L')
    draw = ImageDraw.Draw(temp_img)
    x0, y0 = random.randint(0, size-3), random.randint(0, size-3)
    x1, y1 = x0 + random.randint(2, size//2), y0 + random.randint(2, size//2)
    fill_value = random.randint(0, GRAYSCALE_LEVELS-1) * (255 // (GRAYSCALE_LEVELS - 1))
    draw.ellipse([x0, y0, x1, y1], fill=fill_value)
    return np.array(temp_img)

def create_image(has_ellipse):
    img_array = create_smooth_gradient_background(IMAGE_SIZE)
    if has_ellipse:
        img_array = add_ellipse(img_array, IMAGE_SIZE)
    return img_array

# List to store labels
dataset = []

for i in tqdm(range(NUM_IMAGES)):
    has_ellipse = random.random() < 0.5  # 50% chance
    img_array = create_image(has_ellipse)
    img = Image.fromarray(img_array, 'L')
    
    # Decide which directory to save the image in
    subdir = ELLIPSE_DIR if has_ellipse else NO_ELLIPSE_DIR
    img_filename = f"{i}.png"
    img.save(os.path.join(subdir, img_filename))
    
    # Append label and filename to dataset
    dataset.append({
        'filename': os.path.join(subdir, img_filename),
        'label': has_ellipse,
    })

# Save labels to CSV
df_labels = pd.DataFrame(dataset)
df_labels.to_csv(LABELS_FILE, index=False)
