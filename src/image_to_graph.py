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
    segments = slic(image, n_segments=n_segments, compactness=compactness, channel_axis=None)
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


