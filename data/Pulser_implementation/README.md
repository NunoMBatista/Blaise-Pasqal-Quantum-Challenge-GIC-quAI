# Colon Polyp Detection - Synthetic Data Generation and ML Model

## Overview
This project generates synthetic images of the colon with and without polyps to train machine learning models for polyp detection. It provides methods to create synthetic colon backgrounds, add folds for texture, and generate synthetic polyps with variations in shape, color, and texture. The dataset is stored in a structured format, facilitating training for deep learning models.

## Features
- **Synthetic Data Generation**:
  - Colon background creation with a realistic gradient effect.
  - Addition of colon folds to mimic real anatomical structures.
  - Generation of polyps with varied shapes, colors, and textures.
  - Soft-edge blending for more realistic integration of polyps into the background.
- **Machine Learning Compatibility**:
  - The dataset is stored in `dataset/polyp/` and `dataset/no_polyp/`.
  - Images can be used for training deep learning models for polyp detection.

## Requirements
Ensure you have the following dependencies installed:
```bash
pip install numpy matplotlib pillow torch torchvision scikit-learn tqdm
```

## Usage
### Running the Synthetic Data Generation
To generate synthetic images, run:
```python
python3 dataset_generation/enhanced_colon_sim.py
```
This will generate 500 images and save them into the dataset folders.

### Changing the Number of Generated Images
To modify the number of images created, change the `num_images` parameter in the function call and update line 357 of `enhanced_colon_sim.py` accordingly:
```python
images = generate_dataset(num_images=1000)  # Line 357 Generates 1000 images
```
Adjust the number according to your dataset requirements.

## Dataset Structure
- **With Polyps**: Stored in `dataset/polyp/`
- **Without Polyps**: Stored in `dataset/no_polyp/`

Each image is labeled accordingly, by being placed in each folder.