import torch
import numpy as np
from PIL import Image
import torch_geometric
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.measure import regionprops
from skimage.color import rgb2gray
import networkx as nx
import torch.nn.functional as F
from scipy.spatial import Voronoi
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.util import img_as_ubyte
from skimage.transform import resize
import matplotlib.cm as cm
from matplotlib.colors import Normalize


"""
    behaviour: 
        load and resize an image to a given size
    input: 
        path - path to the image file
        size - optional tuple for desired dimensions
    output: 
        numpy array representing the loaded image
"""
def load_image(path, size=None):
    img = Image.open(path)
    if size:
        img = img.resize(size)
    return np.array(img)


"""
    behaviour: 
        extract local binary pattern texture features from an image
    input: 
        image - numpy array representing grayscale image
        radius - radius for LBP computation
        n_points - number of circularly symmetric neighbor points
    output: 
        LBP features as numpy array
"""
def extract_lbp_features(image, radius=3, n_points=24):
    if len(image.shape) > 2:
        # Convert to grayscale if image is RGB
        gray = rgb2gray(image)
    else:
        gray = image
        
    # Ensure correct data type for LBP
    gray = img_as_ubyte(gray)
    
    # Compute LBP
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    
    # Return LBP image
    return lbp


"""
    behaviour: 
        extract GLCM texture features from an image
    input: 
        image - numpy array representing grayscale image
        distances - list of pixel pair distances
        angles - list of pixel pair angles in radians
    output: 
        dictionary of GLCM features
"""
def extract_glcm_features(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    if len(image.shape) > 2:
        # Convert to grayscale if image is RGB
        gray = rgb2gray(image)
    else:
        gray = image
    
    # Scale down if image is too large for GLCM computation
    if gray.shape[0] > 100 or gray.shape[1] > 100:
        gray = resize(gray, (min(100, gray.shape[0]), min(100, gray.shape[1])))
    
    # Ensure correct data type for GLCM
    gray = img_as_ubyte(gray)
    
    # Compute GLCM
    glcm = graycomatrix(gray, distances=distances, angles=angles, symmetric=True, normed=True)
    
    # Extract features
    features = {
        'contrast': graycoprops(glcm, 'contrast').mean(),
        'dissimilarity': graycoprops(glcm, 'dissimilarity').mean(),
        'homogeneity': graycoprops(glcm, 'homogeneity').mean(),
        'energy': graycoprops(glcm, 'energy').mean(),
        'correlation': graycoprops(glcm, 'correlation').mean(),
        'ASM': graycoprops(glcm, 'ASM').mean()
    }
    
    return features


"""
    behaviour: 
        convert the image to a graph with each pixel as a node with texture features
    input: 
        image - numpy array of shape (h, w) or (h, w, c)
    output: 
        torch_geometric.data.data object representing the image graph
"""
def pixel_to_graph(image):
    if len(image.shape) == 3:  # rgb image
        H, W, C = image.shape
        x = torch.tensor(image.reshape(-1, C), dtype=torch.float)
    else:  # already grayscale image
        H, W = image.shape
        C = 1
        x = torch.tensor(image.reshape(-1, 1), dtype=torch.float)
    
    # Compute texture features
    lbp = extract_lbp_features(image)
    lbp_features = torch.tensor(lbp.reshape(-1, 1), dtype=torch.float)
    
    # Combine color and texture features
    x = torch.cat([x, lbp_features], dim=1)
    
    pos = torch.zeros([H * W, 2], dtype=torch.float)
    for i in range(H):
        for j in range(W):
            pos[i * W + j, 0] = j
            pos[i * W + j, 1] = i
    
    # Create edges
    edge_index = []
    for i in range(H):
        for j in range(W):
            node_idx = i * W + j
            if j < W - 1:
                edge_index.append([node_idx, node_idx + 1])
                edge_index.append([node_idx + 1, node_idx])
            if i < H - 1:
                edge_index.append([node_idx, node_idx + W])
                edge_index.append([node_idx + W, node_idx])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Store texture metadata
    texture_info = {'has_lbp': True, 'feature_dims': {'color': C, 'lbp': 1}}
    
    return Data(x=x, edge_index=edge_index, pos=pos, texture_info=texture_info)


"""
    behaviour: 
        convert the image to a graph based on superpixel segmentation with texture features
    input: 
        image - numpy array of shape (h, w, c); n_segments - number of superpixels; compactness - slic segmentation compactness
    output:     
        torch_geometric.data.data object representing the superpixel graph
"""
def superpixel_to_graph(image, n_segments=10, compactness=10):
    segments = slic(image, n_segments=n_segments, compactness=compactness, channel_axis=-1)
    regions = regionprops(segments + 1)
    centroids = []
    color_features = []
    texture_features = []
    
    # Compute LBP for the whole image
    lbp_full = extract_lbp_features(image)
    
    # Get GLCM features for the whole image to normalize region features
    if len(image.shape) == 3:
        gray_full = rgb2gray(image)
    else:
        gray_full = image
    
    for region in regions:
        centroid = region.centroid
        centroids.append((centroid[1], centroid[0]))  # x, y order
        
        # Extract region from image for color features
        coords = region.coords
        mask = np.zeros(segments.shape, dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        
        # Color features
        if len(image.shape) == 3:
            mean_color = np.mean(image[mask], axis=0)
        else:
            mean_color = np.mean(image[mask])
            mean_color = np.array([mean_color])
        color_features.append(mean_color)
        
        # Texture features - extract LBP values for this region
        lbp_values = lbp_full[mask]
        lbp_hist, _ = np.histogram(lbp_values, bins=10, range=(0, 26))  # 10 bins for uniform LBP
        lbp_hist = lbp_hist / (np.sum(lbp_hist) + 1e-10)  # normalize
        
        # Create a small image patch for GLCM if region is large enough
        if np.sum(mask) > 25:  # Minimum size for reasonable GLCM
            if len(image.shape) == 3:
                region_patch = gray_full[mask].reshape(-1)
            else:
                region_patch = image[mask].reshape(-1)
                
            # Reshape to square for GLCM
            patch_size = int(np.sqrt(len(region_patch)))
            region_patch = region_patch[:patch_size*patch_size].reshape(patch_size, patch_size)
            
            try:
                glcm_features = extract_glcm_features(region_patch)
                glcm_vector = np.array([glcm_features[k] for k in 
                                        ['contrast', 'homogeneity', 'energy', 'correlation']])
            except:
                # Fallback if GLCM fails
                glcm_vector = np.zeros(4)
        else:
            glcm_vector = np.zeros(4)
        
        # Combine texture features
        region_texture = np.concatenate([lbp_hist, glcm_vector])
        texture_features.append(region_texture)
    
    centroids_np = np.array(centroids)
    color_features_np = np.array(color_features)
    texture_features_np = np.array(texture_features)
    
    # Create edges using Voronoi adjacency
    vor = Voronoi(centroids_np)
    edge_set = set()
    for ridge in vor.ridge_points:
        i, j = int(ridge[0]), int(ridge[1])
        edge_set.add(tuple(sorted((i, j))))
    
    edges = []
    for u, v in edge_set:
        edges.append([u, v])
        edges.append([v, u])
    
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    # Combine features
    x = torch.cat([
        torch.tensor(color_features_np, dtype=torch.float),
        torch.tensor(texture_features_np, dtype=torch.float)
    ], dim=1)
    
    pos = torch.tensor(centroids_np, dtype=torch.float)
    
    # Store texture metadata
    texture_info = {
        'has_texture': True, 
        'feature_dims': {
            'color': color_features_np.shape[1], 
            'lbp_hist': 10, 
            'glcm': 4
        }
    }
    
    return Data(x=x, edge_index=edge_index, pos=pos, texture_info=texture_info)


