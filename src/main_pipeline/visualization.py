import matplotlib.pyplot as plt
import numpy as np

def visualize_texture_pulse_effects(graph, pulse, original_data):
    """Visualize how texture affects pulse parameters"""
    if not hasattr(graph.register, 'metadata') or 'texture_features' not in graph.register.metadata:
        print("No texture information available in register")
        return
    
    # Extract texture features
    texture_features = graph.register.metadata['texture_features']
    texture_name = graph.register.metadata.get('texture_feature_name', 'Texture')
    
    # Extract pulse parameters
    atom_names = list(graph.register.qubits)
    texture_values = [texture_features.get(atom, 0) for atom in atom_names]
    
    # Get pulse parameters (this depends on your pulse structure)
    durations = []
    amplitudes = []
    
    # For sequences with multiple channels - handle potential missing attributes
    if hasattr(pulse, 'declared_channels'):
        # In newer versions, use declared_channels
        channel_names = pulse.declared_channels
        
        # Assuming the channel_map contains mapping from target (atom name) to channel
        if hasattr(pulse, 'channel_map') and pulse.channel_map:
            for atom in atom_names:
                # Try to find channel for this atom
                for channel_name, channel in pulse.channel_map.items():
                    if atom in channel.targets:
                        if hasattr(channel, 'pulses') and channel.pulses:
                            durations.append(channel.pulses[0].duration)
                            amplitudes.append(channel.pulses[0].amplitude)
        
        # If we couldn't get durations from the channel_map, use a different approach
        if not durations and hasattr(pulse, 'channels'):
            # Try direct access to channels
            for atom in atom_names:
                channel_name = f"raman_{atom}"
                if channel_name in pulse.channels:
                    channel = pulse.channels[channel_name]
                    if hasattr(channel, 'pulses') and channel.pulses:
                        durations.append(channel.pulses[0].duration)
                        amplitudes.append(channel.pulses[0].amplitude)
    
    # If we still couldn't extract durations, use placeholder values
    if not durations:
        print("Warning: Could not extract pulse parameters. Using placeholder values.")
        durations = [660] * len(texture_values)  # Default duration
        amplitudes = [2 * np.pi * (0.8 + 0.4 * val) for val in texture_values]  # Scaled by texture
    
    # Create visualization
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
    
    plt.tight_layout()
    return fig