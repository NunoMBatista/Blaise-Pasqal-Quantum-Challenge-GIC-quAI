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
def superpixel_to_graph(image, n_segments=100, compactness=10):
    segments = slic(image, n_segments=n_segments, compactness=compactness)
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


"""
    behaviour: 
        visualize the graph using networkx
    input:  
        graph - a torch_geometric.data.data object
        ax - optional matplotlib axis
    output: 
        matplotlib axis with the drawn graph
"""
def visualize_graph(graph, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    G = nx.Graph()
    for i in range(graph.num_nodes):
        G.add_node(i, pos=(graph.pos[i, 0].item(), graph.pos[i, 1].item()))
    if hasattr(graph, 'edge_index') and graph.edge_index.dim() > 0 and graph.edge_index.shape[0] > 0:
        for i in range(graph.edge_index.shape[1]):
            src = graph.edge_index[0, i].item()
            dst = graph.edge_index[1, i].item()
            if src >= graph.num_nodes or dst >= graph.num_nodes:
                continue
            G.add_edge(src, dst)
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, ax=ax, node_size=20, node_color='blue')
    return ax


"""
    behaviour: 
        visualize the graph with nodes colored by texture features
    input:  
        graph - a torch_geometric.data.data object
        feature_name - which feature to use for coloring (e.g., 'lbp', 'energy', 'combined')
        ax - optional matplotlib axis
        cmap - colormap to use
    output: 
        matplotlib axis with the drawn graph colored by texture
"""
def visualize_graph_with_texture(graph, feature_name='texture', ax=None, cmap='viridis'):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create a networkx graph for visualization
    G = nx.Graph()
    for i in range(graph.num_nodes):
        G.add_node(i, pos=(graph.pos[i, 0].item(), graph.pos[i, 1].item()))
    
    # Add edges
    if hasattr(graph, 'edge_index') and graph.edge_index.dim() > 0 and graph.edge_index.shape[0] > 0:
        for i in range(graph.edge_index.shape[1]):
            src = graph.edge_index[0, i].item()
            dst = graph.edge_index[1, i].item()
            if src >= graph.num_nodes or dst >= graph.num_nodes:
                continue
            G.add_edge(src, dst)
    
    # Get positions for drawing
    pos = nx.get_node_attributes(G, 'pos')
    
    # Extract texture features from graph
    node_colors = None
    feature_label = 'Feature Value'
    
    if hasattr(graph, 'texture_info') and graph.texture_info is not None:
        feature_dims = graph.texture_info.get('feature_dims', {})
        color_dims = feature_dims.get('color', 0)
        
        if graph.x is not None:
            # Use PCA for combined features
            if feature_name.lower() in ['combined', 'pca', 'mean']:
                # Import from graph_to_register to avoid circular imports
                from graph_to_register import extract_combined_texture_features
                node_colors = extract_combined_texture_features(graph)
                feature_label = 'Combined Texture (PCA)'
            elif feature_name.lower() == 'lbp':
                # Use LBP feature
                feature_idx = color_dims
                feature_label = 'LBP Value'
                node_colors = graph.x[:, feature_idx].numpy()
            elif feature_name.lower() == 'energy' and 'glcm' in feature_dims:
                # Use energy from GLCM
                feature_idx = color_dims + feature_dims.get('lbp_hist', 10) + 2
                feature_label = 'Energy (GLCM)'
                node_colors = graph.x[:, feature_idx].numpy()
            elif feature_name.lower() == 'homogeneity' and 'glcm' in feature_dims:
                # Use homogeneity from GLCM
                feature_idx = color_dims + feature_dims.get('lbp_hist', 10) + 1
                feature_label = 'Homogeneity (GLCM)'
                node_colors = graph.x[:, feature_idx].numpy()
            elif feature_name.lower() == 'contrast' and 'glcm' in feature_dims:
                # Use contrast from GLCM
                feature_idx = color_dims + feature_dims.get('lbp_hist', 10)
                feature_label = 'Contrast (GLCM)'
                node_colors = graph.x[:, feature_idx].numpy()
            else:
                # Default to mean of texture features
                node_colors = np.mean(graph.x[:, color_dims:].numpy(), axis=1)
                feature_label = 'Mean Texture'
            
            # Normalize node colors if not None
            if node_colors is not None:
                vmin, vmax = node_colors.min(), node_colors.max()
                if vmax > vmin:
                    node_colors = (node_colors - vmin) / (vmax - vmin)
    
    # Draw the graph
    if node_colors is not None:
        nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                      node_size=100, alpha=0.8, 
                                      cmap=plt.get_cmap(cmap), ax=ax)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap), 
                                   norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(feature_label)
    else:
        nx.draw_networkx_nodes(G, pos, node_size=100, node_color='blue', ax=ax)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5, ax=ax)
    
    ax.set_title(f'Graph with {feature_label} Visualization')
    ax.axis('equal')
    
    return ax

