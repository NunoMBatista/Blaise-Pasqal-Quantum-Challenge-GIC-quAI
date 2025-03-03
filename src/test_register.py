import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import pulser
from pulser import Register
from pulser.devices import AnalogDevice
import os

# Import our custom modules
from image_to_graph import load_image, pixel_to_graph, superpixel_to_graph, visualize_graph
from graph_to_register import create_register_from_graph, graph_to_quantum_register, visualize_register_with_connections



if __name__ == "__main__":
    # Test with a sample image
    try:
        image_path = os.path.join("..", "images", "B.png")
        image_path = os.path.join("..", "images", "test_edit.png")

        x, y = 64, 64
        image = load_image(image_path, size=(x, y))
        #if(image.shape != (x, y)):
        #    image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        
    except FileNotFoundError:
        print("Sample image not found, creating random image...")
        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    
    
    # Define scaling factors for different graph types
    pixel_scale = 10.0  # Needs to be larger because pixel graphs are dense
    superpixel_scale = 2.0
    patch_scale = 3.0
    
    figures = []  # Keep track of all figures to prevent garbage collection
    
    # Test with superpixel-based graph
    print("\nConverting image to superpixel-based graph...")
    superpixel_graph = superpixel_to_graph(image, n_segments=15)
    
    print("Converting superpixel-based graph to quantum register...")
    try:
        superpixel_register = graph_to_quantum_register(superpixel_graph, scale_factor=superpixel_scale)
        print(f"Superpixel-based register created with {len(superpixel_register.qubits)} atoms")
        fig2 = visualize_register_with_connections(superpixel_register, superpixel_graph, "Superpixel-based Register")
        figures.append(fig2)
        plt.figure(fig2.number)
        plt.show()
    except Exception as e:
        print(f"Error creating superpixel-based register: {e}")
    

    # Show a comparison with all visualizations
    fig4, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    
    # Plot graph representation (superpixel)
    if 'superpixel_graph' in locals():
        visualize_graph(superpixel_graph, axes[1])
    axes[1].set_title("Graph Representation")
    
    # Plot quantum register with connections (superpixel)
    if 'superpixel_register' in locals() and 'superpixel_graph' in locals():
        plt.sca(axes[2])
        positions = np.array([superpixel_register.qubits[q] for q in superpixel_register.qubits])
        axes[2].scatter(positions[:, 0], positions[:, 1], s=100, color='blue')
        
        # Map nodes to atoms
        node_to_atom = {}
        for i, atom_name in enumerate(superpixel_register.qubits):
            node_id = int(atom_name.split('_')[1])
            node_to_atom[node_id] = i
        
        # Draw edges
        if hasattr(superpixel_graph, 'edge_index') and superpixel_graph.edge_index.size(1) > 0:
            edge_index = superpixel_graph.edge_index.numpy()
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0, i], edge_index[1, i]
                if src in node_to_atom and dst in node_to_atom:
                    src_idx, dst_idx = node_to_atom[src], node_to_atom[dst]
                    axes[2].plot([positions[src_idx][0], positions[dst_idx][0]],
                                [positions[src_idx][1], positions[dst_idx][1]],
                                'k-', alpha=0.5, linewidth=0.5)
        axes[2].set_title("Quantum Register with Connections")
    
    plt.tight_layout()
    plt.show()
    
    # Show atom counts comparison
    if all(var in locals() for var in ['pixel_register', 'superpixel_register', 'patch_register']):
        register_types = ['Pixel', 'Superpixel', 'Patch']
        atom_counts = [len(superpixel_register.qubits)]
        
        fig5, ax = plt.subplots(figsize=(8, 5))
        ax.bar(register_types, atom_counts)
        ax.set_title('Number of Atoms in Different Register Types')
        ax.set_ylabel('Number of Atoms')
        plt.tight_layout()
        plt.show()
