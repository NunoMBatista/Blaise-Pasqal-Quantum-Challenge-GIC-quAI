import torch
import os
from tqdm import tqdm
from data_utils.image_graph_dataset import ImageGraphDataset
from utils.compatibility_utils import make_compatible_with_device
from graphs.texture_aware_graph import TextureAwareGraph
import pulser as pl

def load_datasets(no_polyp_dir, polyp_dir, max_samples, n_qubits, use_superpixels=True, compactness=10):
    """Load and combine polyp and non-polyp datasets"""
    
    polyp_dataset = ImageGraphDataset(
        img_dir=polyp_dir,
        max_samples=max_samples,
        n_segments=n_qubits,
        use_superpixels=use_superpixels,
        label=1,  # Label 1 for polyp
        compactness=compactness
    )
    
    # Create datasets for each class (with labels)
    no_polyp_dataset = ImageGraphDataset(
        img_dir=no_polyp_dir,
        max_samples=max_samples,
        n_segments=n_qubits,
        use_superpixels=use_superpixels,
        label=0,  # Label 0 for no polyp
        compactness=compactness
    )

    # Combine datasets
    combined_dataset = no_polyp_dataset + polyp_dataset
    
    print_dataset_info(no_polyp_dataset, polyp_dataset, combined_dataset)
    
    return combined_dataset

def prepare_graphs_for_compilation(dataset, device=pl.DigitalAnalogDevice, global_duration_coef=1):
    """Prepare graphs for quantum compilation"""
    graphs_to_compile = []
    original_data = []  # Store the original data separately for later reference

    for i, data in enumerate(tqdm(dataset)):
        try:
            # Make the graph compatible with AnalogDevice
            compatible_data = make_compatible_with_device(data)
            original_data.append(compatible_data)
            
            # Create TextureAwareGraph
            graph = TextureAwareGraph(
                id=i,
                data=compatible_data,
                device=device,
                global_duration_coef=global_duration_coef   
            )
            
            graph.target = compatible_data.y.item()  # Preserve the class label
            graphs_to_compile.append(graph)
        except ValueError as e:
            print(f"Graph {i} could not be made compatible: {str(e)}")
        except Exception as e:
            print(f"Unexpected error with graph {i}: {str(e)}")
    
    return graphs_to_compile, original_data

def print_dataset_info(no_polyp_dataset, polyp_dataset, combined_dataset):
    """Print dataset statistics"""
    print(f"""
------------- Dataset created -------------

    - No Polyp Graphs: {len(no_polyp_dataset)}
    - Polyp Graphs: {len(polyp_dataset)}
    - Total Graphs: {len(combined_dataset)}

-------------------------------------------
    """)