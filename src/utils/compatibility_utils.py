import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
import pulser as pl
from pulser.register import Register
from pulser.devices import DigitalAnalogDevice


"""
    behaviour: 
        Check if a graph is compatible with DigitalAnalogDevice and print issues
    input: 
        graph_data - a torch_geometric.data.data object
    output: 
        tuple (bool, list) - compatibility status and issues list
"""
def check_analog_device_compatibility(graph_data):
    issues = []
    
    # Check for edge_index
    if not hasattr(graph_data, 'edge_index') or graph_data.edge_index is None:
        issues.append("Missing edge_index attribute")
    
    # Check for positions
    if not hasattr(graph_data, 'pos') or graph_data.pos is None:
        issues.append("Missing pos attribute")
    
    # Check node features
    if not hasattr(graph_data, 'x') or graph_data.x is None:
        issues.append("Missing node features (x attribute)")
    
    # Check for positions within device constraints
    if hasattr(graph_data, 'pos') and graph_data.pos is not None:
        max_radius = getattr(DigitalAnalogDevice, 'max_distance_from_center', 35.0)
        pos = graph_data.pos.numpy()
        
        # Calculate distances from center
        if len(pos) > 0:  # Make sure there are positions
            distances = np.linalg.norm(pos, axis=1)
            max_dist = np.max(distances)
            
            if max_dist > max_radius:
                issues.append(f"Graph positions exceed maximum radius ({max_dist} > {max_radius})")
    
    # Check bidirectionality (undirected)
    if hasattr(graph_data, 'edge_index') and graph_data.edge_index is not None and graph_data.edge_index.shape[1] > 0:
        edge_index = graph_data.edge_index.numpy()
        edges = set()
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i], edge_index[1, i]
            edges.add((min(u, v), max(u, v)))
        
        for u, v in edges:
            if not edge_exists(edge_index, u, v) or not edge_exists(edge_index, v, u):
                issues.append(f"Graph is directed: missing reciprocal edge for ({u},{v})")
                break
    
    # Check minimum node distance when converted to register
    if hasattr(graph_data, 'pos') and graph_data.pos is not None and graph_data.pos.shape[0] > 1:
        pos = graph_data.pos.numpy()
        min_dist = DigitalAnalogDevice.min_atom_distance
        
        # Calculate pairwise distances
        for i in range(len(pos)):
            for j in range(i+1, len(pos)):
                dist = np.linalg.norm(pos[i] - pos[j])
                if dist < min_dist:
                    issues.append(f"Nodes too close: distance between nodes {i} and {j} is {dist} (min: {min_dist})")
                    break
    
    return len(issues) == 0, issues


"""
    Helper function to check if an edge exists in edge_index
"""
def edge_exists(edge_index, src, dst):
    for i in range(edge_index.shape[1]):
        if edge_index[0, i] == src and edge_index[1, i] == dst:
            return True
    return False


"""
    behaviour: 
        Create a minimal working graph compatible with DigitalAnalogDevice
    input: 
        num_nodes - number of nodes in the graph
    output: 
        torch_geometric.data.data object
"""
def create_minimal_compatible_graph(num_nodes=3):
    # Create positions in a circle pattern with proper spacing
    pos = []
    radius = 15.0  # μm
    for i in range(num_nodes):
        angle = 2 * np.pi * i / num_nodes
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        pos.append([x, y])
    
    # Create edge index for a cycle graph
    edge_index = []
    for i in range(num_nodes):
        j = (i + 1) % num_nodes
        edge_index.append([i, j])
        edge_index.append([j, i])  # Add the reverse edge
    
    # Create node features (simple one-hot encoding)
    x = torch.eye(num_nodes)
    
    # Create the graph
    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        pos=torch.tensor(pos, dtype=torch.float),
        y=torch.tensor([0], dtype=torch.long)
    )
    
    return data


"""
    behaviour: 
        Debug and print detailed information about why a graph might not be compatible
    input: 
        graph - a qek_graphs.BaseGraph object
        device - pulser device
    output: 
        None (prints debug info)
"""
def debug_graph_compatibility(graph, device=DigitalAnalogDevice):
    print(f"\n--- Debugging Graph Compatibility for Graph {graph.id} ---")
    
    # Check base attributes
    print(f"Graph has {graph.num_nodes} nodes and {graph.num_edges} edges")
    
    if not hasattr(graph, 'data'):
        print("ERROR: Graph does not have 'data' attribute")
        return
    
    # Check graph.data attributes
    data = graph.data
    print(f"Node features shape: {data.x.shape if hasattr(data, 'x') and data.x is not None else 'None'}")
    print(f"Edge index shape: {data.edge_index.shape if hasattr(data, 'edge_index') and data.edge_index is not None else 'None'}")
    print(f"Position shape: {data.pos.shape if hasattr(data, 'pos') and data.pos is not None else 'None'}")
    
    # Run compatibility check
    is_compatible, issues = check_analog_device_compatibility(data)
    
    if is_compatible:
        print("Graph appears to be compatible with DigitalAnalogDevice")
    else:
        print("Graph is NOT compatible with DigitalAnalogDevice. Issues found:")
        for issue in issues:
            print(f" - {issue}")
    
    # Additional checks for device-specific constraints
    if hasattr(data, 'pos') and data.pos is not None:
        pos = data.pos.numpy()
        
        # Check if positions would create a valid Register
        try:
            # Create a test register
            atom_positions = {f"atom_{i}": tuple(pos[i]) for i in range(len(pos))}
            test_register = Register(atom_positions)
            print("Successfully created a test Register")
        except Exception as e:
            print(f"ERROR creating Register: {str(e)}")
    
    print("--- End Debug Info ---\n")



"""
    behaviour: 
        make a graph compatible with the DigitalAnalogDevice
    input: 
        graph_data - a torch_geometric.data.data object
    output: 
        modified graph_data compatible with DigitalAnalogDevice
"""
def make_compatible_with_device(graph_data, device=pl.DigitalAnalogDevice):
    # 1. Check if the graph has necessary attributes
    if not hasattr(graph_data, 'edge_index') or graph_data.edge_index is None:
        raise ValueError("Graph must have edge_index")
    
    if not hasattr(graph_data, 'pos') or graph_data.pos is None:
        raise ValueError("Graph must have position information (pos attribute)")
    
    # 2. DigitalAnalogDevice requires undirected graphs - ensure bidirectional edges
    edge_index = graph_data.edge_index.numpy()
    edge_set = set()
    
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i], edge_index[1, i]
        # Add both directions to ensure undirected
        edge_set.add((min(u, v), max(u, v)))
    
    # Rebuild edge_index with undirected edges
    new_edges = []
    for u, v in edge_set:
        new_edges.append([u, v])
        new_edges.append([v, u])  # Add the reverse edge
    
    # If there are no edges, create an empty edge_index with the right shape
    if len(new_edges) == 0:
        graph_data.edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        graph_data.edge_index = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
    
    # 3. Scale positions to fit within device constraints
    max_radius = getattr(device, 'max_distance_from_center', 35.0)  # Default to 35μm
    pos = graph_data.pos.numpy()
    
    # Center around origin
    center = np.mean(pos, axis=0)
    pos = pos - center
    
    # Find maximum distance from center
    max_dist = np.max(np.linalg.norm(pos, axis=1))
    
    # Scale if necessary to fit within max_radius (with 10% margin)
    if max_dist > 0 and max_dist > max_radius * 0.9:
        scale_factor = max_radius * 0.9 / max_dist
        pos = pos * scale_factor
    
    # Update positions
    graph_data.pos = torch.tensor(pos, dtype=torch.float)
    
    # 4. Make sure node features are properly normalized
    if hasattr(graph_data, 'x') and graph_data.x is not None:
        # Normalize each feature column to have zero mean and unit variance
        x = graph_data.x.numpy()
        for i in range(x.shape[1]):
            col = x[:, i]
            mean = np.mean(col)
            std = np.std(col)
            
            if std > 0:
                x[:, i] = (col - mean) / std
            else:
                x[:, i] = np.zeros_like(col)
        
        graph_data.x = torch.tensor(x, dtype=torch.float)
    
    # 5. Make sure there's at least one node
    if not hasattr(graph_data, 'num_nodes') or graph_data.num_nodes == 0:
        raise ValueError("Graph must have at least one node")
    
    # 6. Ensure the graph has an attribute for labels/target
    if not hasattr(graph_data, 'y'):
        graph_data.y = torch.tensor([0], dtype=torch.long)
    
    return graph_data