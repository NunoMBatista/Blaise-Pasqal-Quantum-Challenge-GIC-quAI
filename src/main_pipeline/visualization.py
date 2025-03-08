import matplotlib.pyplot as plt
import numpy as np

def visualize_texture_pulse_effects(graph, pulse_or_sequence, original_data):
    """
    Visualize how texture affects pulse parameters.
    
    Args:
        graph: TextureAwareGraph object with register and texture metadata
        pulse_or_sequence: Either a pulser Pulse object or a pulser Sequence object
        original_data: Original graph data object
        
    Returns:
        Matplotlib figure with visualizations
    """
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