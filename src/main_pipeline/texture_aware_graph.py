import numpy as np
import pulser
from pulser import Register, Sequence, Pulse
from pulser.devices import MockDevice
import torch
from qek.data.graphs import BaseGraph
from pulser.waveforms import RampWaveform, BlackmanWaveform

class TextureAwareGraph(BaseGraph):
    """
    Extension of BaseGraph that encodes texture information into quantum pulses.
    """
    
    def __init__(self, id, data, device, target=None):
        """Initialize with the same parameters as BaseGraph."""
        super().__init__(id=id, data=data, device=device, target=target)
        
        # Default pulse parameters
        self.base_amplitude = 1.0 * 2 * np.pi  # rad/μs
        self.base_duration = 660  # nanoseconds
            
            
    def compile_pulse(self, use_texture=True):
        """
        Create a sequence of pulses that encodes texture information for each node.
        
        Args:
            use_texture: Whether to use texture information to modify pulses
            
        Returns:
            A Sequence object with node-specific texture-dependent pulses
        """
        # Generate the full sequence with node-specific pulses
        sequence = self.create_texture_sequence(use_texture=use_texture)
        return sequence  # Note: This returns a Sequence, not a Pulse
        
    def create_texture_sequence(self, use_texture=True):
        """
        Create a full sequence that encodes texture information with node-specific pulses.
        """
        if not hasattr(self, 'register'):
            self.register = self.compile_register()
            
        # Start building a sequence
        seq = Sequence(self.register, self.device)
        
        # Check if register has texture metadata
        has_texture = (hasattr(self.register, 'metadata') and 
                      self.register.metadata is not None and 
                      'texture_features' in self.register.metadata)
        
        # Get available channels from device
        available_channels = ['raman_local']
        # if hasattr(self.device, 'channels'):
        #     available_channels = self.device.channels
        # else:
        #     # For MockDevice or devices where channels are defined differently
        #     # This is a more general approach that should work with most devices
        #     device_attrs = dir(self.device)
        #     channel_attrs = [attr for attr in device_attrs if 'channel' in attr.lower() and not attr.startswith('__')]
            
        #     # Try different ways of accessing channels
        #     for attr in channel_attrs:
        #         channels_obj = getattr(self.device, attr, None)
        #         if isinstance(channels_obj, (list, tuple, set)):
        #             available_channels = list(channels_obj)
        #             break
        #         elif isinstance(channels_obj, dict):
        #             available_channels = list(channels_obj.keys())
        #             break
        
        print(available_channels)
        
        # Find an appropriate channel
        channel_name = None
        
        # First try for raman_local channel
        if 'raman_local' in available_channels:
            channel_name = 'raman_local'
        # Then look for any local channel
        elif any('_local' in ch for ch in available_channels):
            local_channels = [ch for ch in available_channels if '_local' in ch]
            channel_name = local_channels[0]
        # Then just use raman_global if available
        elif 'raman_global' in available_channels:
            channel_name = 'raman_global'
        # Last resort: use the first available channel
        elif available_channels:
            channel_name = available_channels[0]
        else:
            # If we still can't find a channel, use fixed default channels that should exist
            # Default channels that typically exist in Pulser devices
            default_channels = ['rydberg_global', 'ground_rydberg', 'digital']
            for ch in default_channels:
                try:
                    # Try declaring and see if it works
                    test_seq = Sequence(self.register, self.device)
                    test_seq.declare_channel('ch', ch)
                    channel_name = ch
                    break
                except Exception:
                    continue
            
            # If still no channel found, raise a descriptive error
            if channel_name is None:
                raise ValueError(f"Could not determine valid channel for device {self.device.id}. "
                                f"Available channels detection failed.")
        
        # Now declare the channel with proper error handling
        try:
            seq.declare_channel('pulse_channel', channel_name)
        except Exception as e:
            raise ValueError(f"Error declaring channel '{channel_name}': {str(e)}. "
                            f"Available channels: {available_channels}")
            
        # Get all atoms in the register
        atoms = list(self.register.qubits.keys())
        
        if has_texture and use_texture:
            # Extract texture features for each node
            texture_features = self.register.metadata.get('texture_features', {})
            
            # Default texture value if not available
            default_texture = 0.5
            
            # Target all atoms at once for the pulse_channel
            seq.target(atoms, 'pulse_channel')
            
            for atom in atoms:
                # Get this node's texture value or use default
                node_texture = texture_features.get(atom, default_texture)
                
                # Scale duration based on texture value
                duration = int(self.base_duration * (0.5 + node_texture))
                
                # Create node-specific pulse with duration proportional to texture
                pulse = Pulse.ConstantAmplitude(
                    amplitude=self.base_amplitude,
                    detuning=RampWaveform(duration, 0, 0),
                    phase=0.0
                )

                                
                # Add pulse to the properly targeted channel
                seq.add(pulse, 'pulse_channel')
                #seq.delay('pulse_channel', 10)  # Small delay between consecutive pulses
        else:
            # If no texture info, use default pulses for all atoms
            # Target all atoms at once
            seq.target(atoms, 'pulse_channel')
            
            pulse = Pulse.ConstantAmplitude(
                amplitude=self.base_amplitude,
                detuning=RampWaveform(self.base_duration, 0, 0),
                phase=0.0
            )
            seq.add(pulse, 'pulse_channel')
        
        # Add measurement if supported
        try:
            #seq.measure("ground-rydberg")
            seq.measure("digital") # FOR RAMAN
        except Exception as e:
            # Just log the error but don't fail if measurement isn't supported
            print(f"Warning: Could not add measurement: {str(e)}")
        
        return seq