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
        convert the image to a graph with each pixel as a node
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
    pos = torch.zeros([H * W, 2], dtype=torch.float)
    for i in range(H):
        for j in range(W):
            pos[i * W + j, 0] = j
            pos[i * W + j, 1] = i
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
    return Data(x=x, edge_index=edge_index, pos=pos)


"""
    behaviour: 
        convert the image to a graph based on superpixel segmentation
    input: 
        image - numpy array of shape (h, w, c); n_segments - number of superpixels; compactness - slic segmentation compactness
    output:     
        torch_geometric.data.data object representing the superpixel graph
"""
def superpixel_to_graph(image, n_segments=100, compactness=10):
    segments = slic(image, n_segments=n_segments, compactness=compactness)
    regions = regionprops(segments + 1)
    centroids = []
    features = []
    for region in regions:
        centroid = region.centroid
        centroids.append((centroid[1], centroid[0]))
        coords = region.coords
        mask = np.zeros(segments.shape, dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        if len(image.shape) == 3:
            mean_color = np.mean(image[mask], axis=0)
        else:
            mean_color = np.mean(image[mask])
            mean_color = np.array([mean_color])
        features.append(mean_color)
    centroids_np = np.array(centroids)
    features_np = np.array(features)
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
    x = torch.tensor(features_np, dtype=torch.float)
    pos = torch.tensor(centroids_np, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, pos=pos)


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


def adaptive_superpixel_to_graph(image, n_segments=100, compactness=10, uniformity_threshold=0.01, grid_density=3):
    """
    Convert an image to a graph where:
    - Uniform color regions get regular grid structures
    - Non-uniform regions get adaptive node placement reflecting color variations
    
    Args:
        image: numpy array of shape (h, w, c)
        n_segments: number of superpixels
        compactness: SLIC segmentation compactness
        uniformity_threshold: threshold below which a region is considered uniform
        grid_density: number of grid points along each dimension for uniform regions
        
    Returns:
        torch_geometric.data.Data object representing the graph
    """
    # Get segmentation
    segments = slic(image, n_segments=n_segments, compactness=compactness)
    
    # Initialize lists for graph construction
    all_nodes_pos = []
    all_nodes_features = []
    all_edges = []
    node_offset = 0
    
    # Process each segment
    for segment_id in range(np.max(segments) + 1):
        # Get mask for current segment
        mask = (segments == segment_id)
        if not np.any(mask):
            continue
            
        # Get coordinates of pixels in this segment
        coords = np.argwhere(mask)
        
        # Calculate bounding box of segment
        y_min, x_min = np.min(coords, axis=0)
        y_max, x_max = np.max(coords, axis=0)
        
        # Get color values in the segment
        if len(image.shape) == 3:
            segment_colors = image[mask]
        else:
            segment_colors = image[mask].reshape(-1, 1)
        
        # Calculate color variance to determine uniformity
        color_variance = np.mean(np.var(segment_colors, axis=0))
        is_uniform = color_variance < uniformity_threshold
        
        segment_nodes_pos = []
        segment_nodes_features = []
        segment_edges = []
        
        if is_uniform:
            # For uniform regions, create a regular grid
            mean_color = np.mean(segment_colors, axis=0)
            
            # Create grid points
            y_points = np.linspace(y_min, y_max, grid_density)
            x_points = np.linspace(x_min, x_max, grid_density)
            
            # Create nodes at grid points that fall within the segment
            node_idx = 0
            grid_nodes = {}
            
            for i, y in enumerate(y_points):
                for j, x in enumerate(x_points):
                    y_int, x_int = int(y), int(x)
                    if y_int < image.shape[0] and x_int < image.shape[1] and mask[y_int, x_int]:
                        segment_nodes_pos.append((x, y))
                        segment_nodes_features.append(mean_color)
                        grid_nodes[(i, j)] = node_idx
                        node_idx += 1
            
            # Connect grid points to form a regular grid
            for (i1, j1), idx1 in grid_nodes.items():
                for di, dj in [(0, 1), (1, 0)]:  # Right and down neighbors
                    if (i1+di, j1+dj) in grid_nodes:
                        idx2 = grid_nodes[(i1+di, j1+dj)]
                        segment_edges.append([idx1 + node_offset, idx2 + node_offset])
                        segment_edges.append([idx2 + node_offset, idx1 + node_offset])
                        
        else:
            # For non-uniform regions, place nodes to reflect color variations
            from sklearn.cluster import KMeans
            
            # Combine position and color for clustering
            pixel_data = []
            for y, x in coords:
                if len(image.shape) == 3:
                    color = image[y, x]
                else:
                    color = np.array([image[y, x]])
                pixel_data.append(np.concatenate(([x, y], color)))
            pixel_data = np.array(pixel_data)
            
            # Normalize features for clustering
            pos_scale = max(1, np.max([x_max - x_min, y_max - y_min]))
            color_scale = max(1, np.max(segment_colors) - np.min(segment_colors))
            
            normalized_data = np.zeros_like(pixel_data, dtype=float)
            normalized_data[:, :2] = pixel_data[:, :2] / pos_scale
            normalized_data[:, 2:] = pixel_data[:, 2:] / color_scale
            
            # Determine number of nodes based on color variance and area
            area_ratio = len(coords) / (image.shape[0] * image.shape[1])
            n_nodes = max(3, min(20, int(n_segments * area_ratio * (1 + color_variance * 10))))
            
            # Use k-means to find representative points
            if len(normalized_data) > n_nodes:
                kmeans = KMeans(n_clusters=n_nodes, random_state=0).fit(normalized_data)
                centers = kmeans.cluster_centers_
                
                # Denormalize
                centers[:, :2] *= pos_scale
                centers[:, 2:] *= color_scale
                
                # Extract positions and colors
                for i in range(centers.shape[0]):
                    segment_nodes_pos.append((centers[i, 0], centers[i, 1]))
                    segment_nodes_features.append(centers[i, 2:])
                    
                # Connect nodes based on Delaunay triangulation
                from scipy.spatial import Delaunay
                if len(segment_nodes_pos) >= 3:  # Need at least 3 points for triangulation
                    tri = Delaunay(np.array(segment_nodes_pos))
                    for simplex in tri.simplices:
                        for i in range(3):
                            for j in range(i+1, 3):
                                segment_edges.append([simplex[i] + node_offset, simplex[j] + node_offset])
                                segment_edges.append([simplex[j] + node_offset, simplex[i] + node_offset])
            else:
                # If we have fewer pixels than desired nodes, just use all pixels
                for i, (y, x) in enumerate(coords):
                    segment_nodes_pos.append((x, y))
                    if len(image.shape) == 3:
                        segment_nodes_features.append(image[y, x])
                    else:
                        segment_nodes_features.append([image[y, x]])
                
                # Connect nearest neighbors
                from scipy.spatial import KDTree
                if len(segment_nodes_pos) > 1:
                    kdtree = KDTree(segment_nodes_pos)
                    for i, pos in enumerate(segment_nodes_pos):
                        _, indices = kdtree.query(pos, k=min(4, len(segment_nodes_pos)))
                        for j in indices[1:]:  # Skip self
                            segment_edges.append([i + node_offset, j + node_offset])
                            segment_edges.append([j + node_offset, i + node_offset])
        
        # Add segment nodes and edges to the overall graph
        all_nodes_pos.extend(segment_nodes_pos)
        all_nodes_features.extend(segment_nodes_features)
        all_edges.extend(segment_edges)
        
        # Update node offset for next segment
        node_offset += len(segment_nodes_pos)
    
    # Convert to PyTorch Geometric Data object
    if not all_nodes_pos:
        raise ValueError("No nodes were created. Try adjusting parameters.")
    
    x = torch.tensor(all_nodes_features, dtype=torch.float)
    pos = torch.tensor(all_nodes_pos, dtype=torch.float)
    
    if all_edges:
        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, pos=pos)


