
# Image Directory

This directory should contain image files that will be processed by the image_to_graph functions to create graph datasets with texture features.

## Supported Image Formats

- JPG/JPEG
- PNG
- BMP

## How Images Are Processed

1. Images are loaded and resized to a consistent size
2. For each image, a graph is created using one of two methods:
   - **Superpixel segmentation** (default): The image is segmented into regions with similar properties
   - **Pixel-based**: Each pixel becomes a node in the graph (only used for very small images)
3. Texture features are extracted from each region/pixel:
   - Local Binary Patterns (LBP)
   - Gray Level Co-occurrence Matrix (GLCM) features:
     - Contrast
     - Energy
     - Homogeneity
     - Correlation
4. These texture features become node attributes in the resulting graph

## Adding Your Own Images

Simply place any images you want to process in this directory. The script will automatically detect and process them.

## Tips

- For better results, use images with clear textures and patterns
- Very large images will be automatically resized before processing
- Using the superpixel method is recommended for most cases as it produces more manageable graphs
