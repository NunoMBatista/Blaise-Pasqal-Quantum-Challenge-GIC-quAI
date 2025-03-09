import numpy as np
import pulser
from pulser import Register
from pulser.devices import DigitalAnalogDevice
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import torch
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils.register_scaling import scale_positions_to_register_dim


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
def graph_to_quantum_register(graph_data, texture_feature='pca', register_dim=30, global_pulse_coef=1):
    """
    Convert a graph to a quantum register with texture information.
    
    Args:
        graph_data: PyTorch Geometric data object with positions
        texture_feature: Feature to extract ('pca', 'lbp', 'energy', etc.)
        register_dim: Target dimension for the register in microns
    
    Returns:
        Pulser Register with texture information in metadata
    """
    # Check for positions
    if not hasattr(graph_data, 'pos') or graph_data.pos is None:
        raise ValueError("Graph requires position data (pos attribute)")
    
    # Scale positions according to register_dim
    scaled_positions = scale_positions_to_register_dim(graph_data.pos, register_dim)
    
    # Extract texture features
    texture_features = extract_texture_features(graph_data, texture_feature)
    
    # Create register
    atom_positions = {}
    texture_dict = {}
    
    for i in range(scaled_positions.shape[0]):
        atom_name = f"atom_{i}"
        # Convert position to tuple for Register
        position = tuple(scaled_positions[i].tolist()) if isinstance(scaled_positions, torch.Tensor) else tuple(scaled_positions[i])
        atom_positions[atom_name] = position
        
        # Store texture value
        if texture_features is not None and i < len(texture_features):
            texture_dict[atom_name] = float(texture_features[i])
    
    # Create register
    register = Register(atom_positions)
    
    # Add texture metadata
    register.metadata = {
        "texture_features": texture_dict,
        "texture_feature_name": texture_feature.capitalize(),
        "register_dim": register_dim
    }
    
    return register


"""
    behaviour:
        extract texture feature values based on the specified feature type
    input:
        graph_data - a torch_geometric.data.data object
        texture_feature - which texture feature to use ("pca", "lbp", etc.)
    output:
        tuple of (texture_values, texture_feature_name)
"""
def extract_texture_features(graph_data, feature_name='pca'):
    """
    Extract texture features from graph data.
    
    Args:
        graph_data: PyTorch Geometric data object
        feature_name: Type of texture feature to extract
    
    Returns:
        Normalized feature values for each node
    """
    if feature_name.lower() in ['combined', 'pca', 'mean']:
        # Extract combined features using PCA
        return extract_texture_pca(graph_data)
        
    # Get texture features from node attributes
    if hasattr(graph_data, 'texture_info') and graph_data.texture_info is not None:
        feature_dims = graph_data.texture_info.get('feature_dims', {})
        
        if feature_name.lower() == 'lbp' and 'color' in feature_dims:
            feature_idx = feature_dims.get('color', 0)
            features = graph_data.x[:, feature_idx].numpy()
        elif feature_name.lower() == 'energy' and 'glcm' in feature_dims:
            feature_idx = feature_dims.get('color', 0) + feature_dims.get('lbp_hist', 10) + 2
            features = graph_data.x[:, feature_idx].numpy()
        elif feature_name.lower() == 'homogeneity' and 'glcm' in feature_dims:
            feature_idx = feature_dims.get('color', 0) + feature_dims.get('lbp_hist', 10) + 1
            features = graph_data.x[:, feature_idx].numpy()
        elif feature_name.lower() == 'contrast' and 'glcm' in feature_dims:
            feature_idx = feature_dims.get('color', 0) + feature_dims.get('lbp_hist', 10)
            features = graph_data.x[:, feature_idx].numpy()
        else:
            # Default to mean of texture features
            color_dims = feature_dims.get('color', 0)
            features = np.mean(graph_data.x[:, color_dims:].numpy(), axis=1)
    else:
        # No texture info, use the first few node features as proxy
        if graph_data.x is not None and graph_data.x.shape[1] > 0:
            features = graph_data.x[:, 0].numpy()
        else:
            # No features available
            features = np.zeros(graph_data.num_nodes)
    
    # Normalize features to [0,1] range
    features_min = np.min(features)
    features_max = np.max(features)
    if features_max > features_min:
        normalized_features = (features - features_min) / (features_max - features_min)
    else:
        normalized_features = np.zeros_like(features)
        
    return normalized_features


"""
    behaviour:
        extract combined texture features using PCA from graph data
    input:
        graph_data - a torch_geometric.data.data object
    output:
        normalized PCA-based texture values
"""
def extract_texture_pca(graph_data):
    """
    Extract combined texture features using PCA from graph data.
    
    Args:
        graph_data: PyTorch Geometric data object
    
    Returns:
        Normalized texture values based on PCA of node features
    """
    # Check if we have node features
    if not hasattr(graph_data, 'x') or graph_data.x is None or graph_data.x.shape[1] == 0:
        return np.zeros(graph_data.num_nodes)
    
    # Get node features
    if isinstance(graph_data.x, torch.Tensor):
        features = graph_data.x.numpy()
    else:
        features = graph_data.x
    
    # Use texture features if we know where they are
    if hasattr(graph_data, 'texture_info') and graph_data.texture_info is not None:
        feature_dims = graph_data.texture_info.get('feature_dims', {})
        if 'color' in feature_dims:
            # Use only texture-related features
            start_idx = feature_dims.get('color', 0)
            features = features[:, start_idx:]
    
    # Standardize features (important for PCA)
    scaler = StandardScaler()
    try:
        scaled_features = scaler.fit_transform(features)
    except:
        # If scaling fails, just use raw features
        scaled_features = features
    
    # Apply PCA to reduce to 1D
    try:
        if scaled_features.shape[1] > 1:
            pca = PCA(n_components=1)
            combined_features = pca.fit_transform(scaled_features).flatten()
        else:
            # Already 1D
            combined_features = scaled_features.flatten()
    except:
        # If PCA fails, use mean across features
        combined_features = np.mean(features, axis=1)
    
    # Normalize to [0, 1]
    features_min = np.min(combined_features)
    features_max = np.max(combined_features)
    
    if features_max > features_min:
        normalized_features = (combined_features - features_min) / (features_max - features_min)
    else:
        normalized_features = np.zeros_like(combined_features)
        
    return normalized_features