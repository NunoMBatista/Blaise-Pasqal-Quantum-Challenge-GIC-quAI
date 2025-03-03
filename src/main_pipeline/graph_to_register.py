import numpy as np
import pulser
from pulser import Register
from pulser.devices import AnalogDevice
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import torch


"""
    behaviour: 
        convert a pytorch geometric graph to a pulser quantum register
    input: 
        graph_data - a torch_geometric.data.data object representing a graph
        device - pulser device (default: analogdevice)
        scale_factor - factor to scale graph positions
        min_distance - minimal distance between atoms
    output: 
        a pulser register with atoms positioned according to graph nodes
"""
def create_register_from_graph(graph_data, device=AnalogDevice, scale_factor=5.0, min_distance=4.0):
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
    
    # Create pulser Register
    if not atom_positions:
        raise ValueError("No valid atom positions found. Try adjusting scale_factor.")
    
    print(f"Creating register with {len(atom_positions)} atoms (max allowed: {max_atoms})")
    register = Register(atom_positions)
    return register


"""
    behaviour: 
        convert a pytorch geometric graph to a quantum register
    input: 
        graph_data - a torch_geometric.data.data object
        scale_factor - factor to scale positions
        device - pulser device
    output: 
        a pulser register object
"""
def graph_to_quantum_register(graph_data, scale_factor=5.0, device=AnalogDevice):
    # Adjust scale factor based on graph size to fit device constraints
    num_nodes = graph_data.num_nodes
    max_radius = getattr(device, 'max_distance_from_center', 35.0)
    
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
            print(f"Reducing scale factor from {scale_factor} to {adjusted_scale:.2f} to fit device constraints")
            scale_factor = adjusted_scale
    
    # Create the register from graph positions
    register = create_register_from_graph(graph_data, device=device, scale_factor=scale_factor)
    
    return register


"""
    behaviour: 
        visualize the quantum register with connections based on graph edges
    input: 
        register - a pulser register object
        graph_data - optional graph data with edge_index
        title - title for the plot
    output: 
        a matplotlib figure showing the register
"""
def visualize_register_with_connections(register, graph_data=None, title="atom register"):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the register
    register.draw(custom_ax=ax, blockade_radius=AnalogDevice.min_atom_distance, show=False)
    
    # Add connections between atoms based on graph edge_index if available
    if graph_data is not None and hasattr(graph_data, 'edge_index') and graph_data.edge_index.size(1) > 0:
        # Create mapping from node indices to atom names
        node_to_atom = {}
        for i, atom_name in enumerate(register.qubits):
            node_id = int(atom_name.split('_')[1])
            node_to_atom[node_id] = i
        
        # Get atom positions - use the correct attribute
        # The Register class stores positions in qubits dictionary
        positions = np.array([register.qubits[q] for q in register.qubits])
        
        # Draw edges
        edge_index = graph_data.edge_index.numpy()
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            # Only draw edges if both nodes are in the register
            if src in node_to_atom and dst in node_to_atom:
                src_idx, dst_idx = node_to_atom[src], node_to_atom[dst]
                ax.plot([positions[src_idx][0], positions[dst_idx][0]],
                       [positions[src_idx][1], positions[dst_idx][1]],
                       'k-', alpha=0.3, linewidth=0.5)
    
    plt.title(title)
    plt.tight_layout()
    return fig