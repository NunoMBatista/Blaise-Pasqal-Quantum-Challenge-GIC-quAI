# README

## Overview

This script generates a dataset of small grayscale images, some containing random ellipses and others without. The dataset is divided into two classes (images with ellipses and images without), and the images are saved to separate directories. Additionally, a CSV file is created, recording the labels and file paths for all the generated images.

## Features

- **Gradient Backgrounds:**  
  Each image has a unique smooth gradient background created using random grayscale levels.  
- **Random Ellipses:**  
  50% of the images have an ellipse added at a random location, size, and grayscale value.  
- **Organized Output:**  
  Images are saved in dedicated directories (`with_ellipse` and `without_ellipse`) under a specified output folder. A CSV file (`labels.csv`) lists the file paths and corresponding labels (true/false for ellipse presence).  
- **Scalable Parameters:**  
  Key parameters, such as image size, number of images, and grayscale levels, can be easily adjusted by changing constants in the script.  

## How It Works

1. **Gradient Background Generation:**  
   The `create_smooth_gradient_background()` function creates a smooth gradient image by interpolating random grayscale values at the corners.  
   
2. **Ellipse Drawing:**  
   The `add_ellipse()` function randomly places an ellipse on the image if required, filling it with a random grayscale intensity.  

3. **Image Generation:**  
   Images are generated in a loop, and for each image, a random decision is made to include an ellipse or not. The generated images are then saved into the corresponding output subdirectory.

4. **CSV Labels File:**  
   Each image’s file path and label (ellipse present or not) are stored in a DataFrame, which is then saved as `labels.csv`. This file can be used for training machine learning models or other analyses.

## Output Structure

- **Dataset Directory (default: `dataset`):**  
  - `with_ellipse/` - Contains images with ellipses.  
  - `without_ellipse/` - Contains images without ellipses.  
  - `labels.csv` - A CSV file with two columns: `filename` (file path) and `label` (boolean indicating ellipse presence).  

## Adjusting Parameters

- **IMAGE_SIZE:**  
  Defines the dimensions of each image (e.g., `8` for an 4x4 image).  
- **NUM_IMAGES:**  
  Sets the total number of images to generate (default: `512`).  
- **OUTPUT_DIR:**  
  The directory where the dataset and labels will be stored (default: `dataset`).  
- **GRAYSCALE_LEVELS:**  
  Controls the number of grayscale levels used in both gradients and ellipses (default: `8`).  

## Requirements

To install external dependencies, run:  
```bash
pip install tqdm numpy pillow pandas
```

## Running the Code

1. Ensure the required libraries are installed.  
2. Adjust constants at the top of the script as needed.  
3. Run the script:  
   ```bash
   python3 dataset_generation/colon_sim.py 
   ```
4. The generated dataset will be found in the specified `dataset` folder.