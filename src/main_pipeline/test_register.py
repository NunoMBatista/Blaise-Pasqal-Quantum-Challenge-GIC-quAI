import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import pulser
from pulser import Register
from pulser.devices import AnalogDevice
import os

# Import our custom modules
from image_to_graph import load_image, pixel_to_graph, superpixel_to_graph, visualize_graph, visualize_graph_with_texture
from graph_to_register import create_register_from_graph, graph_to_quantum_register, visualize_register_with_connections

if __name__ == "__main__":
    # Test with a sample image
    try:
        image_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "test_edit.png")
        x, y = 64, 64
        image = load_image(image_path, size=(x, y))
    except FileNotFoundError:
        print("Sample image not found, creating random image...")
        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)

    # Define scaling factors for different graph types
    superpixel_scale = 2.0
    
    figures = []  # Keep track of all figures to prevent garbage collection
    
    # Test with superpixel-based graph
    print("\nConverting image to superpixel-based graph...")
    superpixel_graph = superpixel_to_graph(image, n_segments=20)
    
    # # Visualize graph with PCA representation of texture features
    # visualize_graph_with_texture(superpixel_graph, feature_name='PCA')

    # Show regular register and combined texture register
    print("\nCreating register with combined texture features...")
    #fig_comparison, axes = plt.subplots(1, 2, figsize=(14, 6))
    

    # Register with combined PCA features
    try:
        register_combined = graph_to_quantum_register(superpixel_graph, scale_factor=superpixel_scale, 
                                                  texture_feature="pca")
        #visualize_register_with_connections(register_combined, superpixel_graph, 
        #                                 title="Register with Combined Texture Features (PCA)")
        #plt.sca(axes[1])
    except Exception as e:
        print(f"Error creating register with combined features: {e}")

    
    # Comprehensive visualization comparing original image, graph with combined texture, and quantum register
    fig_final, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Plot original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    
    # Plot graph representation with combined texture
    visualize_graph_with_texture(superpixel_graph, feature_name='pca', ax=axes[1])
    axes[1].set_title("Graph with Combined Texture Features")
    
    # Plot quantum register with combined texture features
    try:
        if 'register_combined' not in locals():
            register_combined = graph_to_quantum_register(superpixel_graph, scale_factor=superpixel_scale,
                                                      texture_feature='combined')
        plt.sca(axes[0])
        visualize_register_with_connections(register_combined, superpixel_graph, 
                                         "Quantum Register with Combined Texture")
    except Exception as e:
        print(f"Error in final visualization: {e}")
        axes[0].text(0.5, 0.5, "Error creating visualization", ha='center')
    
    plt.tight_layout()
    plt.show()
