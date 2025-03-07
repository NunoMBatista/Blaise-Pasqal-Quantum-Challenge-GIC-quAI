import numpy as np
import pulser
from pulser import Register
from pulser.devices import AnalogDevice
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import torch
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


"""
    behaviour: 
        convert a pytorch geometric graph to a pulser quantum register with texture attributes
    input: 
        graph_data - a torch_geometric.data.data object representing a graph
        device - pulser device (default: analogdevice)
        scale_factor - factor to scale graph positions
        min_distance - minimal distance between atoms
        texture_feature - which texture feature to use ("lbp", "contrast", "homogeneity", "energy")
    output: 
        a pulser register with atoms positioned according to graph nodes and texture metadata
"""
def create_register_from_graph(graph_data, device=AnalogDevice, scale_factor=5.0, min_distance=4.0, texture_feature="pca"):
    # Extract node positions from the graph
    if not hasattr(graph_data, 'pos') or graph_data.pos is None:
        raise ValueError("Graph data must have position information (pos attribute)")
    
    # Get device constraints
    max_atoms = getattr(device, 'max_atom_num', 25)  # Default to 25 if not specified
    max_radius = getattr(device, 'max_distance_from_center', 35.0)  # Default to 35μm
    min_atom_distance = device.min_atom_distance
    
    # Convert positions to numpy and scale to appropriate distances for atoms
    pos_np = graph_data.pos.numpy() * scale_factor
    
    # Center the positions around (0, 0) to maximize available space
    center = np.mean(pos_np, axis=0)
    pos_np = pos_np - center
    
    # Create dictionary of positions for pulser Register
    atom_positions = {}
    valid_positions = []
    valid_indices = []  # Track which indices are valid for texture mapping
    
    # Extract texture features if available
    has_texture = hasattr(graph_data, 'texture_info') and graph_data.texture_info is not None
    texture_features = {}
    texture_feature_name = texture_feature  # Keep track of which feature we're using
    
    if has_texture:
        # Determine which features to use for visualization
        feature_dims = graph_data.texture_info.get('feature_dims', {})
        color_dims = feature_dims.get('color', 0)
        
        # Extract relevant texture features from node attributes
        if graph_data.x is not None:
            # Select texture feature based on input parameter
            if texture_feature.lower() == 'lbp' and color_dims > 0:
                # Use LBP feature (first texture feature after color)
                texture_col = color_dims
                texture_feature_name = 'LBP'
            elif texture_feature.lower() == 'contrast' and 'glcm' in feature_dims:
                # Use contrast from GLCM (typically the 1st GLCM feature)
                texture_col = color_dims + feature_dims.get('lbp_hist', 10)
                texture_feature_name = 'Contrast'
            elif texture_feature.lower() == 'homogeneity' and 'glcm' in feature_dims:
                # Use homogeneity from GLCM (typically the 2nd GLCM feature)
                texture_col = color_dims + feature_dims.get('lbp_hist', 10) + 1
                texture_feature_name = 'Homogeneity'
            elif texture_feature.lower() == 'energy' and 'glcm' in feature_dims:
                # Use energy from GLCM (typically the 3rd GLCM feature)
                texture_col = color_dims + feature_dims.get('lbp_hist', 10) + 2
                texture_feature_name = 'Energy'
            elif texture_feature.lower() == 'correlation' and 'glcm' in feature_dims:
                # Use correlation from GLCM (typically the 4th GLCM feature)
                texture_col = color_dims + feature_dims.get('lbp_hist', 10) + 3
                texture_feature_name = 'Correlation'
            else:
                # Default to LBP histogram mean as a general texture measure
                if 'lbp_hist' in feature_dims and feature_dims['lbp_hist'] > 0:
                    # Use mean of LBP histogram
                    lbp_start = color_dims
                    lbp_end = color_dims + feature_dims['lbp_hist']
                    texture_values = np.mean(graph_data.x[:, lbp_start:lbp_end].numpy(), axis=1)
                    texture_feature_name = 'LBP Mean'
                else:
                    # Fallback to first non-color feature
                    texture_col = color_dims
                    texture_feature_name = 'Texture'
            
            # Get texture values based on column if we haven't computed them yet
            if 'texture_values' not in locals():
                try:
                    texture_values = graph_data.x[:, texture_col].numpy()
                except (IndexError, ValueError):
                    print(f"Warning: Could not extract {texture_feature} feature, using default")
                    texture_values = np.zeros(graph_data.num_nodes)
                
            # Normalize texture values
            if len(texture_values) > 0:
                min_val = texture_values.min()
                max_val = texture_values.max()
                if max_val > min_val:
                    texture_values = (texture_values - min_val) / (max_val - min_val)
                else:
                    texture_values = np.zeros_like(texture_values)
    
    # For each node in the graph, create an atom in the register
    for i, pos in enumerate(pos_np):
        # Check if we've reached the maximum number of atoms
        if len(valid_positions) >= max_atoms:
            break
            
        # Check if this position is within the maximum radius
        distance_from_center = np.linalg.norm(pos)
        if distance_from_center > max_radius:
            continue
        
        # Check if this position is valid (not too close to other atoms)
        is_valid = True
        for valid_pos in valid_positions:
            distance = np.linalg.norm(pos - valid_pos)
            if distance < min_atom_distance:
                is_valid = False
                break
                
        if is_valid:
            atom_positions[f"atom_{i}"] = tuple(pos)
            valid_positions.append(pos)
            valid_indices.append(i)
            
            # Store texture attributes if available
            if has_texture and 'texture_values' in locals():
                texture_features[f"atom_{i}"] = texture_values[i]
    
    # Create pulser Register
    if not atom_positions:
        raise ValueError("No valid atom positions found. Try adjusting scale_factor.")
    
    print(f"Creating register with {len(atom_positions)} atoms (max allowed: {max_atoms})")
    register = Register(atom_positions)
    
    # Attach texture features as metadata
    if texture_features:
        register.metadata = {
            "texture_features": texture_features,
            "texture_feature_name": texture_feature_name
        }
    
    return register


"""
    behaviour: 
        convert a pytorch geometric graph to a quantum register
    input: 
        graph_data - a torch_geometric.data.data object
        scale_factor - factor to scale positions
        device - pulser device
        texture_feature - which texture feature to use for visualization
        register_dim - desired dimension of the register (square area in μm)
    output: 
        a pulser register object
"""
def graph_to_quantum_register(graph_data, scale_factor=5.0, device=AnalogDevice, texture_feature="pca", register_dim=None):
    # Get device constraints
    max_radius = getattr(device, 'max_distance_from_center', 35.0)
    
    # Check if register_dim is specified to override scale_factor
    if register_dim is not None:
        if hasattr(graph_data, 'pos') and graph_data.pos is not None:
            pos = graph_data.pos.numpy()
            # Calculate the current spread of positions
            min_pos = np.min(pos, axis=0)
            max_pos = np.max(pos, axis=0)
            current_width = max_pos[0] - min_pos[0]
            current_height = max_pos[1] - min_pos[1]
            
            # Calculate the larger dimension to preserve aspect ratio
            max_dimension = max(current_width, current_height)
            if max_dimension > 0:
                # Adjust scale factor to fit within register_dim
                scale_factor = register_dim / max_dimension
                print(f"Calculated scale factor to fit {register_dim}×{register_dim} μm area: {scale_factor:.4f}")
            else:
                print(f"Warning: Graph positions have zero spread, using default scale factor: {scale_factor}")
    
    # Adjust scale factor based on graph size to fit device constraints
    num_nodes = graph_data.num_nodes
    
    # Auto-adjust scale factor if there are too many nodes or they're too spread out
    if num_nodes > 25:
        print(f"Warning: Graph has {num_nodes} nodes, but device supports max 25 atoms.")
        print("Only the first 25 valid positions will be used.")
    
    # Estimate if positions might be too far from center
    if hasattr(graph_data, 'pos') and graph_data.pos is not None:
        pos = graph_data.pos.numpy()
        center = np.mean(pos, axis=0)
        max_dist = np.max(np.linalg.norm(pos - center, axis=1))
        
        # If positions would be outside max radius, reduce scale factor
        if max_dist * scale_factor > max_radius:
            adjusted_scale = max_radius / max_dist * 0.9  # 10% safety margin
            print(f"Reducing scale factor from {scale_factor:.4f} to {adjusted_scale:.4f} to fit device constraints")
            scale_factor = adjusted_scale
    
    # Create the register from graph positions
    register = create_register_from_graph(graph_data, device=device, scale_factor=scale_factor, 
                                         texture_feature=texture_feature)
    
    return register


"""
    behaviour: 
        visualize the quantum register with connections based on graph edges and texture information
    input: 
        register - a pulser register object
        graph_data - optional graph data with edge_index
        title - title for the plot
    output: 
        a matplotlib figure showing the register
"""
def visualize_register_with_connections(register, graph_data=None, title="atom register"):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Check if texture features are available
    has_texture = hasattr(register, 'metadata') and register.metadata and 'texture_features' in register.metadata
    
    # Get atom positions
    positions = np.array([register.qubits[q] for q in register.qubits])
    
    # Create mapping from atom names to indices
    atom_to_idx = {atom: i for i, atom in enumerate(register.qubits)}
    
    # Draw the register
    register.draw(custom_ax=ax, blockade_radius=AnalogDevice.min_atom_distance, show=False)
    
    # Add connections between atoms based on graph edge_index if available
    if graph_data is not None and hasattr(graph_data, 'edge_index') and graph_data.edge_index.size(1) > 0:
        # Create mapping from node indices to atom names
        node_to_atom = {}
        for atom_name in register.qubits:
            node_id = int(atom_name.split('_')[1])
            node_to_atom[node_id] = atom_name
        
        # Draw edges
        edge_index = graph_data.edge_index.numpy()
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            # Only draw edges if both nodes are in the register
            if src in node_to_atom and dst in node_to_atom:
                src_atom, dst_atom = node_to_atom[src], node_to_atom[dst]
                src_idx, dst_idx = atom_to_idx[src_atom], atom_to_idx[dst_atom]
                ax.plot([positions[src_idx][0], positions[dst_idx][0]],
                       [positions[src_idx][1], positions[dst_idx][1]],
                       'k-', alpha=0.3, linewidth=0.5)
    
    # Visualize texture information if available
    if has_texture:
        texture_features = register.metadata['texture_features']
        texture_feature_name = register.metadata.get('texture_feature_name', 'Texture')
        
        # Extract texture values and create color map
        texture_values = np.array([texture_features.get(atom, 0) for atom in register.qubits])
        norm = Normalize(vmin=0, vmax=1)
        cmap = cm.viridis
        
        # Scatter plot with texture-based coloring
        scatter = ax.scatter(positions[:, 0], positions[:, 1], c=texture_values, 
                           cmap=cmap, norm=norm, s=40, zorder=5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(f'{texture_feature_name} Intensity')
    
    plt.title(title)
    plt.tight_layout()
    return fig


"""
    behaviour:
        extract and combine texture features using PCA
    input:
        graph_data - a torch_geometric.data.data object with texture features
    output:
        combined texture values as numpy array
"""
def extract_combined_texture_features(graph_data, verbose=False):
    if not hasattr(graph_data, 'texture_info') or graph_data.texture_info is None:
        return None
    
    # Get feature dimensions
    feature_dims = graph_data.texture_info.get('feature_dims', {})
    color_dims = feature_dims.get('color', 0)
    
    # Check if we have necessary features
    if graph_data.x is None or graph_data.x.shape[1] <= color_dims:
        return None
    
    # Extract all texture features (everything after color features)
    texture_features = graph_data.x[:, color_dims:].numpy()
    
    # Check if we have enough data for PCA
    if texture_features.shape[0] < 2 or texture_features.shape[1] < 2:
        # Not enough data for PCA, use mean of features
        return np.mean(texture_features, axis=1)
    
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(texture_features)
    
    # Apply PCA to reduce to 1 dimension
    pca = PCA(n_components=1)
    principal_components = pca.fit_transform(scaled_features).flatten()
    
    # Normalize to [0, 1] range
    min_val = np.min(principal_components)
    max_val = np.max(principal_components)
    if max_val > min_val:
        normalized_pca = (principal_components - min_val) / (max_val - min_val)
    else:
        normalized_pca = np.zeros_like(principal_components)
    
    # Get explained variance to report
    explained_variance = pca.explained_variance_ratio_[0]
    if(verbose == True):
        print(f"PCA first component explains {explained_variance:.2%} of texture variation")
    
    return normalized_pca


"""
    behaviour: 
        convert a pytorch geometric graph to a pulser quantum register with texture attributes
    input: 
        graph_data - a torch_geometric.data.data object representing a graph
        device - pulser device (default: analogdevice)
        scale_factor - factor to scale graph positions
        min_distance - minimal distance between atoms
        texture_feature - which texture feature to use or "combined" for PCA-based combination
    output: 
        a pulser register with atoms positioned according to graph nodes and texture metadata
"""
def create_register_from_graph(graph_data, device=AnalogDevice, scale_factor=5.0, min_distance=4.0, texture_feature="pca"):
    # Extract node positions from the graph
    if not hasattr(graph_data, 'pos') or graph_data.pos is None:
        raise ValueError("Graph data must have position information (pos attribute)")
    
    # Get device constraints
    max_atoms = getattr(device, 'max_atom_num', 25)  # Default to 25 if not specified
    max_radius = getattr(device, 'max_distance_from_center', 35.0)  # Default to 35μm
    min_atom_distance = device.min_atom_distance
    
    # Convert positions to numpy and scale to appropriate distances for atoms
    pos_np = graph_data.pos.numpy() * scale_factor
    
    # Center the positions around (0, 0) to maximize available space
    center = np.mean(pos_np, axis=0)
    pos_np = pos_np - center
    
    # Create dictionary of positions for pulser Register
    atom_positions = {}
    valid_positions = []
    valid_indices = []  # Track which indices are valid for texture mapping
    
    # Extract texture features if available
    has_texture = hasattr(graph_data, 'texture_info') and graph_data.texture_info is not None
    texture_features = {}
    texture_feature_name = texture_feature  # Keep track of which feature we're using
    
    if has_texture:
        # Determine which features to use for visualization
        feature_dims = graph_data.texture_info.get('feature_dims', {})
        color_dims = feature_dims.get('color', 0)
        
        # Extract relevant texture features from node attributes
        if graph_data.x is not None:
            # Use PCA or specified texture feature
            if texture_feature.lower() in ['combined', 'pca', 'mean']:
                texture_values = extract_combined_texture_features(graph_data)
                texture_feature_name = 'Combined Texture (PCA)'
            elif texture_feature.lower() == 'lbp' and color_dims > 0:
                # Use LBP feature (first texture feature after color)
                texture_col = color_dims
                texture_feature_name = 'LBP'
                texture_values = graph_data.x[:, texture_col].numpy()
            elif texture_feature.lower() == 'contrast' and 'glcm' in feature_dims:
                # Use contrast from GLCM
                texture_col = color_dims + feature_dims.get('lbp_hist', 10)
                texture_feature_name = 'Contrast'
                texture_values = graph_data.x[:, texture_col].numpy()
            elif texture_feature.lower() == 'homogeneity' and 'glcm' in feature_dims:
                # Use homogeneity from GLCM
                texture_col = color_dims + feature_dims.get('lbp_hist', 10) + 1
                texture_feature_name = 'Homogeneity'
                texture_values = graph_data.x[:, texture_col].numpy()
            elif texture_feature.lower() == 'energy' and 'glcm' in feature_dims:
                # Use energy from GLCM
                texture_col = color_dims + feature_dims.get('lbp_hist', 10) + 2
                texture_feature_name = 'Energy'
                texture_values = graph_data.x[:, texture_col].numpy()
            elif texture_feature.lower() == 'correlation' and 'glcm' in feature_dims:
                # Use correlation from GLCM
                texture_col = color_dims + feature_dims.get('lbp_hist', 10) + 3
                texture_feature_name = 'Correlation'
                texture_values = graph_data.x[:, texture_col].numpy()
            else:
                # Default to mean of all texture features
                texture_values = np.mean(graph_data.x[:, color_dims:].numpy(), axis=1)
                texture_feature_name = 'Mean Texture'
                
            # Normalize texture values if not already done by PCA
            if texture_feature.lower() not in ['combined', 'pca', 'mean']:
                if len(texture_values) > 0:
                    min_val = texture_values.min()
                    max_val = texture_values.max()
                    if max_val > min_val:
                        texture_values = (texture_values - min_val) / (max_val - min_val)
                    else:
                        texture_values = np.zeros_like(texture_values)
    
    # For each node in the graph, create an atom in the register
    for i, pos in enumerate(pos_np):
        # Check if we've reached the maximum number of atoms
        if len(valid_positions) >= max_atoms:
            break
            
        # Check if this position is within the maximum radius
        distance_from_center = np.linalg.norm(pos)
        if distance_from_center > max_radius:
            continue
        
        # Check if this position is valid (not too close to other atoms)
        is_valid = True
        for valid_pos in valid_positions:
            distance = np.linalg.norm(pos - valid_pos)
            if distance < min_atom_distance:
                is_valid = False
                break
                
        if is_valid:
            atom_positions[f"atom_{i}"] = tuple(pos)
            valid_positions.append(pos)
            valid_indices.append(i)
            
            # Store texture attributes if available
            if has_texture and 'texture_values' in locals():
                texture_features[f"atom_{i}"] = texture_values[i]
    
    # Create pulser Register
    if not atom_positions:
        raise ValueError("No valid atom positions found. Try adjusting scale_factor.")
    
    print(f"Creating register with {len(atom_positions)} atoms (max allowed: {max_atoms})")
    register = Register(atom_positions)
    
    # Attach texture features as metadata
    if texture_features:
        register.metadata = {
            "texture_features": texture_features,
            "texture_feature_name": texture_feature_name
        }
    
    return register