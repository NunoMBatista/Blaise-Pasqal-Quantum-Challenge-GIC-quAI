import matplotlib.pyplot as plt
from visualization.visualization import visualize_texture_pulse_effects, visualize_register_with_connections

def visualize_example(compiled_graphs, example_index=0):
    """Visualize an example graph and its quantum representation"""
    if not compiled_graphs or example_index >= len(compiled_graphs):
        print("No compiled graphs available to visualize")
        return None
        
    try:
        example_graph, example_data, example_pulse = compiled_graphs[example_index]
        
        # Create a sequence for visualization
        example_sequence = example_graph.create_texture_sequence(use_texture=True)
        
        # Visualize sequence
        sequence_fig = example_sequence.draw()
        
        # Visualize texture effects
        texture_fig = visualize_texture_pulse_effects(
            example_graph, 
            example_sequence, 
            example_data
        )
        
        return sequence_fig, texture_fig
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        return None

def visualize_processed_data(processed_data):
    """Visualize processed quantum data"""
    if not processed_data:
        print("No processed data to visualize")
        return None
        
    example_data = processed_data[0]
    
    # Draw register and pulse
    register_fig = example_data.draw_register()
    pulse_fig = example_data.draw_pulse()
    
    # Draw excitation if available
    if hasattr(example_data, 'draw_excitation'):
        excitation_fig = example_data.draw_excitation()
        return register_fig, pulse_fig, excitation_fig
    
    return register_fig, pulse_fig