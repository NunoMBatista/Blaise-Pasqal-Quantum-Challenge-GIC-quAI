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
        Create a pulse that encodes texture information.
        
        Args:
            use_texture: Whether to use texture information to modify pulses
            
        Returns:
            A Pulse object (not a Sequence) with texture-dependent parameters
        """
        if not hasattr(self, 'register'):
            self.register = self.compile_register()
            
        # Check if register has texture metadata
        has_texture = (hasattr(self.register, 'metadata') and 
                    self.register.metadata is not None and 
                    'texture_features' in self.register.metadata)
        
        # Extract texture features and determine pulse parameters
        if has_texture and use_texture:
            texture_features = self.register.metadata['texture_features']
            avg_texture = np.mean([val for val in texture_features.values()])
            
            # Scale parameters based on average texture
            duration = int(self.base_duration * (0.75 + 0.5 * avg_texture))
            amplitude = self.base_amplitude * (0.8 + 0.4 * avg_texture)
        else:
            # Default parameters
            duration = self.base_duration
            amplitude = self.base_amplitude
        
        # Looking at the error, ConstantAmplitude doesn't take duration directly
        # Instead, it likely takes amplitude, phase, and detuning waveform
        
        # Create detuning ramp waveform with the duration
        detuning = RampWaveform(duration, 0, 0)
        
        # Create and return a single Pulse object with proper parameters
        pulse = Pulse.ConstantAmplitude(
            amplitude=amplitude,
            detuning=detuning,
            phase=0.0
        )
        
        return pulse
        
    def create_texture_sequence(self, use_texture=True):
        """
        Create a full sequence that encodes texture information.
        This is separate from compile_pulse and can be used for visualization.
        """
        if not hasattr(self, 'register'):
            self.register = self.compile_register()
            
        # Start building a sequence
        seq = Sequence(self.register, self.device)
        
        # Check available channels
        available_channels = self.device.channels
        #print(f"Available channels: {available_channels}")
        
        # Check if register has texture metadata
        has_texture = (hasattr(self.register, 'metadata') and 
                      self.register.metadata is not None and 
                      'texture_features' in self.register.metadata)
        
        if has_texture and use_texture and 'rydberg_global' in available_channels:
            # Extract texture features
            texture_features = self.register.metadata['texture_features']
            avg_texture = np.mean([val for val in texture_features.values()])
            
            # Declare global channel
            seq.declare_channel('global', 'rydberg_global')
            
            # Create pulse with texture-modulated parameters
            duration = int(self.base_duration * (0.75 + 0.5 * avg_texture))
            amplitude = self.base_amplitude * (0.8 + 0.4 * avg_texture)
            
            pulse = Pulse.ConstantAmplitude(
                amplitude=amplitude,
                detuning=RampWaveform(duration, 0, 0),
                phase=0.0
            )
            
            seq.add(pulse, 'global')
        else:
            # Fallback to default pulse on first available channel
            channel_name = available_channels[0]
            seq.declare_channel('global', channel_name)
            
            pulse = Pulse.ConstantAmplitude(
                amplitude=self.base_amplitude,
                detuning=RampWaveform(self.base_duration, 0, 0),
                phase=0.0
            )
            
            seq.add(pulse, 'global')
        
        # Add measurement if supported
        try:
            seq.measure("ground-rydberg")
        except Exception as e:
            print(f"Warning: Could not add measurement: {str(e)}")
        
        return seq.build()