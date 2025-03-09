from tqdm import tqdm
import pulser as pl
from graphs.graph_to_register import graph_to_quantum_register

def compile_graphs(graphs_to_compile, original_data, register_dim=30, texture_feature='pca', global_duration_coef=1):
    """Compile graphs to quantum registers and pulses"""
    compiled = []
    
    print(f"\nUsing device: {pl.DigitalAnalogDevice.name}")
    print(f"Register dimension: {register_dim} μm")
    print(f"Minimum atom distance: {pl.DigitalAnalogDevice.min_atom_distance} μm")
    
    if hasattr(pl.DigitalAnalogDevice, 'channel_objects'):
        print(f"Available channels: {pl.DigitalAnalogDevice.channel_objects}")
    
    for i, graph in enumerate(tqdm(graphs_to_compile)):
        try:
            # Access the original graph data with texture info
            original_graph_data = original_data[i]

            # Create custom register with properly scaled positions
            custom_register = graph_to_quantum_register(
                original_graph_data, 
                texture_feature=texture_feature,
                register_dim=register_dim,
                global_pulse_coef=global_duration_coef
            )
            
            # Assign register to graph
            graph.register = custom_register
            
            try:
                # Compile pulse sequence
                sequence = graph.compile_pulse(use_texture=True)
                compiled.append((graph, original_graph_data, sequence))
            except Exception as e:
                print(f"Compilation error for graph {graph.id}: {str(e)}")
                print(f"Register has {len(graph.register.qubits)} atoms")
                
        except Exception as e:
            print(f"Unexpected error during compilation for graph {graph.id}: {str(e)}")
    
    print(f"Compiled {len(compiled)} graphs out of {len(graphs_to_compile)}.")
    return compiled