from torch_geometric.data import Data, Dataset
from qek.shared.error import CompilationError
from image_to_graph import load_image, superpixel_to_graph, pixel_to_graph, visualize_graph_with_texture
from graph_to_register import graph_to_quantum_register, visualize_register_with_connections
import os
import glob
from tqdm import tqdm 
import torch
from pathlib import Path
import numpy as np


class ImageGraphDataset(Dataset):
    def __init__(self, img_dir, transform=None, pre_transform=None, max_samples=100, 
                 n_segments=20, use_superpixels=True):
        """
        Creates a dataset of graphs from images.

        Args:
            img_dir (str): Directory containing image files.
            transform (callable, optional): Transform to apply to the data.
            pre_transform (callable, optional): Pre-transform to apply to the data.
            max_samples (int): Maximum number of samples to include.
            n_segments (int): Number of superpixels for segmentation.
            use_superpixels (bool): Whether to use superpixel segmentation or pixel-based graphs.
        """
        super().__init__(transform, pre_transform)
        self.img_dir = img_dir
        self.max_samples = max_samples
        self.n_segments = n_segments
        self.use_superpixels = use_superpixels
        
        # Find all image files in the directory
        img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        self.img_paths = []
        for ext in img_extensions:
            self.img_paths.extend(glob.glob(os.path.join(img_dir, '**', ext), recursive=True))
        
        # Limit the number of samples
        self.img_paths = self.img_paths[:max_samples]
        
        print(f"Found {len(self.img_paths)} images in {img_dir}")
        
        # Generate graphs from images
        self.graphs = []
        for path in tqdm(self.img_paths, desc="Converting images to graphs"):
            try:
                # Load and process image
                img = load_image(path, size=(64, 64))  # Resize for consistency
                
                # Convert to graph
                if self.use_superpixels:
                    graph = superpixel_to_graph(img, n_segments=n_segments)
                else:
                    # For pixel graphs, use a small patch to keep graph size reasonable
                    small_img = load_image(path, size=(16, 16))  # Smaller for pixel graphs
                    graph = pixel_to_graph(small_img)
                
                # Add filename as metadata
                graph.filename = Path(path).name
                
                # Add a random binary label (can be changed for real classification tasks)
                graph.y = torch.tensor([1 if np.random.rand() > 0.5 else 0], dtype=torch.long)
                
                self.graphs.append(graph)
            except Exception as e:
                print(f"Error processing {path}: {str(e)}")
                
        print(f"Successfully created {len(self.graphs)} graphs")

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]

