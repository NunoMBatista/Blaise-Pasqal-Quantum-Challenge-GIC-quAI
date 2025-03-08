import os
import sys
from pathlib import Path
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageFilter
import random
import shutil

def load_image(image_path):
    """Load an image from path"""
    return Image.open(image_path)

def save_image(image, output_path):
    """Save image to specified path"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)

def rotate_image(image, angle):
    """Rotate image by specified angle"""
    return image.rotate(angle)

def flip_image(image, flip_type):
    """Flip image horizontally or vertically
    flip_type: 0 for horizontal, 1 for vertical
    """
    if flip_type == 0:
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        return image.transpose(Image.FLIP_TOP_BOTTOM)

def adjust_brightness(image, factor):
    """Adjust brightness of image"""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def adjust_contrast(image, factor):
    """Adjust contrast of image"""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def zoom_image(image, factor):
    """Zoom in on image by cropping edges and resizing back to original size"""
    width, height = image.size
    crop_width = int(width * factor)
    crop_height = int(height * factor)
    
    # Calculate coordinates for cropping
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height
    
    # Crop and resize
    cropped = image.crop((left, top, right, bottom))
    return cropped.resize((width, height))

def add_noise(image, intensity=0.05):
    """Add random noise to image"""
    img_array = np.array(image).astype(np.float32)
    noise = np.random.normal(0, intensity * 255, img_array.shape)
    noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

def apply_blur(image, radius=2):
    """Apply Gaussian blur to image"""
    return image.filter(ImageFilter.GaussianBlur(radius=radius))

def apply_random_augmentation(image_path, output_dir, file_prefix="aug", label=None):
    """Apply a random combination of augmentations to an image"""
    try:
        img = load_image(image_path)
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        
        # Apply 2-4 random augmentations
        num_augmentations = random.randint(2, 4)
        augmentations = random.sample([
            lambda i: rotate_image(i, random.choice([90, 180, 270])),
            lambda i: flip_image(i, random.choice([0, 1])),
            lambda i: adjust_brightness(i, random.uniform(0.7, 1.3)),
            lambda i: adjust_contrast(i, random.uniform(0.7, 1.3)),
            lambda i: zoom_image(i, random.uniform(0.8, 0.95)),
            lambda i: add_noise(i, random.uniform(0.01, 0.05)),
            lambda i: apply_blur(i, random.uniform(0.5, 1.5))
        ], k=num_augmentations)
        
        # Apply selected augmentations
        augmented_img = img
        for aug_func in augmentations:
            augmented_img = aug_func(augmented_img)
        
        # Create output path in appropriate subdirectory based on label
        if label:
            output_subdir = os.path.join(output_dir, label)
        else:
            # Try to infer label from directory structure
            parent_dir = os.path.basename(os.path.dirname(image_path))
            if parent_dir in ["polyp", "no_polyp"]:
                output_subdir = os.path.join(output_dir, parent_dir)
            else:
                output_subdir = output_dir
                
        os.makedirs(output_subdir, exist_ok=True)
        
        # Save augmented image with new name
        output_path = os.path.join(output_subdir, f"{file_prefix}_{name}{ext}")
        save_image(augmented_img, output_path)
        return output_path
        
    except Exception as e:
        print(f"Error augmenting {image_path}: {str(e)}")
        return None

def augment_dataset(input_dir, output_dir, augmentation_factor=5, preserve_structure=True):
    """
    Augment all images in a directory
    
    Args:
        input_dir: Directory containing original images
        output_dir: Directory to save augmented images
        augmentation_factor: Number of augmented versions to create per image
        preserve_structure: Whether to preserve directory structure
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # If preserving structure, copy directory structure first
    if preserve_structure:
        for root, dirs, _ in os.walk(input_dir):
            for dir_name in dirs:
                rel_dir = os.path.relpath(os.path.join(root, dir_name), input_dir)
                os.makedirs(os.path.join(output_dir, rel_dir), exist_ok=True)
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    all_images = []
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                all_images.append(os.path.join(root, file))
    
    print(f"Found {len(all_images)} images for augmentation")
    
    # Apply augmentations
    augmented_files = []
    for i, image_path in enumerate(tqdm(all_images, desc="Augmenting images")):
        for j in range(augmentation_factor):
            # Determine output path based on structure preservation
            if preserve_structure:
                rel_path = os.path.relpath(image_path, input_dir)
                parent_dir = os.path.dirname(rel_path)
                output_path = os.path.join(output_dir, parent_dir)
            else:
                output_path = output_dir
                
            # Apply random augmentation
            aug_file = apply_random_augmentation(
                image_path, 
                output_path,
                file_prefix=f"aug{j+1}"
            )
            
            if aug_file:
                augmented_files.append(aug_file)
    
    print(f"Created {len(augmented_files)} augmented images")
    return augmented_files

def copy_originals_to_output(input_dir, output_dir):
    """Copy original images to output directory to include them in the dataset"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    copied_files = []
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, input_dir)
                
                # Determine output subdirectory (preserving structure)
                dst_path = os.path.join(output_dir, rel_path)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                
                # Copy file
                shutil.copy2(src_path, dst_path)
                copied_files.append(dst_path)
    
    print(f"Copied {len(copied_files)} original images to output directory")
    return copied_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Augment medical image dataset for polyp detection')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing original images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save augmented images')
    parser.add_argument('--augmentation_factor', type=int, default=5, help='Number of augmented versions per image')
    parser.add_argument('--include_originals', action='store_true', help='Include original images in output')
    parser.add_argument('--preserve_structure', action='store_true', help='Preserve directory structure')
    
    args = parser.parse_args()
    
    print(f"Starting dataset augmentation...")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Augmentation factor: {args.augmentation_factor}")
    
    # Create augmented dataset
    augmented_files = augment_dataset(
        args.input_dir, 
        args.output_dir,
        args.augmentation_factor,
        args.preserve_structure
    )
    
    # Optionally include originals
    if args.include_originals:
        copied_files = copy_originals_to_output(args.input_dir, args.output_dir)
        print(f"Total dataset size: {len(augmented_files) + len(copied_files)} images")
    else:
        print(f"Total dataset size: {len(augmented_files)} images")
    
    print("Dataset augmentation complete!")
