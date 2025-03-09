import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from pulser import DigitalAnalogDevice, AnalogDevice, MockDevice
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import seaborn as sns
import datetime
from sklearn.metrics import confusion_matrix
from pipeline.config import RESULTS_DIR
import os

"""
    behaviour:
        visualize the texture features of the register
    input:
        register - a pulser register object
        feature_name - which feature to use for coloring (e.g., 'lbp', 'energy', 'combined')
        cmap - colormap to use
    output:
        a matplotlib figure showing the register with texture features
"""
def visualize_texture_pulse_effects(graph, pulse_or_sequence, original_data):
    if not hasattr(graph.register, 'metadata') or 'texture_features' not in graph.register.metadata:
        print("No texture information available in register")
        return plt.figure(figsize=(8, 3))
    
    # Extract texture features
    texture_features = graph.register.metadata['texture_features']
    texture_name = graph.register.metadata.get('texture_feature_name', 'Texture')
    
    # Extract pulse parameters
    atom_names = list(graph.register.qubits)
    texture_values = [texture_features.get(atom, 0) for atom in atom_names]
    
    # Extract durations and amplitudes based on object type
    durations = []
    amplitudes = []
    
    # Track if Rydberg pulse is present
    has_rydberg_pulse = False
    rydberg_amplitude = None
    rydberg_duration = None
    
    # Case 1: Direct Pulse object
    if hasattr(pulse_or_sequence, 'duration') and hasattr(pulse_or_sequence, 'amplitude'):
        durations = [pulse_or_sequence.duration] * len(texture_values)
        amplitudes = [pulse_or_sequence.amplitude] * len(texture_values)
    
    # Case 2: Sequence object with channels
    elif hasattr(pulse_or_sequence, 'channels'):
        channels = pulse_or_sequence.channels
        
        # Try to extract pulses from each channel
        for channel_name, channel in channels.items():
            if hasattr(channel, 'pulses') and channel.pulses:
                # Check if this is a Rydberg channel
                if 'rydberg' in channel_name.lower():
                    has_rydberg_pulse = True
                    if channel.pulses:
                        first_pulse = channel.pulses[0]
                        rydberg_amplitude = getattr(first_pulse, 'amplitude', 0)
                        rydberg_duration = getattr(first_pulse, 'duration', 0)
                    continue
                
                # Found pulses in regular channel
                for i, atom in enumerate(atom_names):
                    # Look for pulses targeting this atom
                    matching_pulse = None
                    if hasattr(channel, 'targets') and atom in channel.targets:
                        matching_pulse = channel.pulses[0]  # Use first pulse as example
                    
                    if matching_pulse is not None:
                        durations.append(getattr(matching_pulse, 'duration', 660))
                        amplitudes.append(getattr(matching_pulse, 'amplitude', 2*np.pi))
                
                # If we found pulses in this channel, we're done
                if durations:
                    break
        
        # If still no durations, try a different approach for sequence
        if not durations and hasattr(pulse_or_sequence, 'get_pulses'):
            try:
                # This extracts all pulses from the sequence as a dictionary
                all_pulses = pulse_or_sequence.get_pulses()
                if all_pulses:
                    # Just use the first pulse we find as a reference
                    first_pulse = list(all_pulses.values())[0]
                    if hasattr(first_pulse, 'duration') and hasattr(first_pulse, 'amplitude'):
                        durations = [first_pulse.duration] * len(texture_values)
                        amplitudes = [first_pulse.amplitude] * len(texture_values)
            except Exception as e:
                print(f"Could not extract pulses using get_pulses(): {e}")
    
    # Case 3: Using our TextureAwareGraph's custom sequence creation
    if not durations and hasattr(graph, 'base_duration') and hasattr(graph, 'base_amplitude'):
        # Use the graph's base pulse parameters to estimate
        base_duration = graph.base_duration
        base_amplitude = graph.base_amplitude
        
        # Create pulse parameters that vary with texture value
        durations = [int(base_duration * (0.5 + val)) for val in texture_values]
        amplitudes = [base_amplitude * (0.8 + 0.4 * val) for val in texture_values]
    
    # If we still couldn't extract durations, use placeholder values
    if not durations:
        print("Warning: Could not extract pulse parameters directly. Using TextureAwareGraph formula.")
        # Use the formula from TextureAwareGraph.create_texture_sequence
        base_duration = 660
        base_amplitude = 2 * np.pi
        durations = [int(base_duration * (0.5 + val)) for val in texture_values]
        amplitudes = [base_amplitude * (0.8 + 0.4 * val) for val in texture_values]
    
    # Create visualization
    if has_rydberg_pulse:
        # If we have Rydberg pulse, create 4 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    else:
        # Original 3 subplots if no Rydberg pulse
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot texture distribution
    ax1.hist(texture_values, bins=10)
    ax1.set_title(f'{texture_name} Distribution')
    ax1.set_xlabel('Texture Value')
    ax1.set_ylabel('Count')
    
    # Only plot scatter plots if we have data
    if len(durations) == len(texture_values) and len(durations) > 0:
        # Plot texture vs duration
        ax2.scatter(texture_values, durations)
        ax2.set_title('Texture vs Pulse Duration')
        ax2.set_xlabel('Texture Value')
        ax2.set_ylabel('Duration (ns)')
        
        # Plot texture vs amplitude
        ax3.scatter(texture_values, amplitudes)
        ax3.set_title('Texture vs Pulse Amplitude')
        ax3.set_xlabel('Texture Value')
        ax3.set_ylabel('Amplitude (rad/μs)')
    else:
        ax2.text(0.5, 0.5, 'No pulse parameter data available', 
                horizontalalignment='center', verticalalignment='center')
        ax3.text(0.5, 0.5, 'No pulse parameter data available', 
                horizontalalignment='center', verticalalignment='center')
    
    # Add Rydberg pulse information if available
    if has_rydberg_pulse:
        ax4.axis('off')  # No plot needed, just text
        info_text = "Rydberg Global Pulse:\n"
        info_text += f"Amplitude: {rydberg_amplitude:.2f} rad/μs\n"
        info_text += f"Duration: {rydberg_duration} ns\n"
        ax4.text(0.1, 0.5, info_text, fontsize=14, 
                 horizontalalignment='left', verticalalignment='center')
        ax4.set_title('Rydberg Pulse Parameters')
    
    plt.tight_layout()
    return fig


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
                from graphs.graph_to_register import extract_combined_texture_features
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
    register.draw(custom_ax=ax, blockade_radius=DigitalAnalogDevice.min_atom_distance, show=False)
    
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
        plot a confusion matrix using matplotlib and seaborn
    input:
        y_true - ground truth labels
        y_pred - predicted labels
        class_names - names of the classes for the axis labels
    output:
        a matplotlib figure showing the confusion matrix
"""
def plot_confusion_matrix(y_true, y_pred, class_names=['No Polyp', 'Polyp'], save_results=False, result_dir="./results"):   
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages for annotations
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum * 100
    annot = np.empty_like(cm).astype(str)
    
    # Format annotations to show both count and percentage
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_perc[i, j]:.1f}%)"
    
    # Plot using seaborn for a nicer appearance
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=annot, 
        fmt='', 
        cmap='Blues',
        cbar=True,
        square=True,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    # Labels and title
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    
    # Display the plot
    plt.tight_layout()
    plt.show()

    global RESULTS_DIR
    img_dir = os.path.join(RESULTS_DIR, 'quantum_confusion_matrix.png')
    if save_results:
        fig.savefig(img_dir)
        
    
    return fig