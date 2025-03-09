import numpy as np
import pulser
from pulser import Register, Sequence, Pulse
from pulser.devices import DigitalAnalogDevice
import torch
from qek.data.graphs import BaseGraph
from pulser.waveforms import RampWaveform, BlackmanWaveform

class TextureAwareGraph(BaseGraph):
    """
        Extension of BaseGraph (from pasqal's QEK implementation) that encodes texture information into quantum pulses.
    """
    
    def __init__(self, id, data, device, target=None, global_duration_coef=1):
        """Initialize with the same parameters as BaseGraph."""
        super().__init__(id=id, data=data, device=device, target=target)
        
        # Default pulse parameters
        self.base_amplitude = 1.0 * 2 * np.pi  # rad/μs
        self.base_duration = 660 # nanoseconds
        self.global_duration_coef = global_duration_coef
            
            
    """
        behaviour:
            Create a sequence of pulses that encodes texture information for each node.
        input:        
            use_texture: Whether to use texture information to modify pulses
        output:
            A Sequence object with node-specific texture-dependent pulses
    """
    def compile_pulse(self, use_texture=True):
        # Generate the full sequence with node-specific pulses
        sequence = self.create_texture_sequence(use_texture=use_texture)
        return sequence  # Note: This returns a Sequence, not a Pulse


    """
        behaviour:
            Create a full sequence that encodes texture information with node-specific pulses.
        input:
            use_texture: Whether to use texture information to modify pulses
        output:
            A Sequence object with node-specific texture-dependent pulses
    """        
    def create_texture_sequence(self, use_texture=True):
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
        
        # Find an appropriate channel
        channel_name = None
            
        seq.declare_channel('local_pulse', 'rydberg_local')
        seq.declare_channel('rydberg_glob', 'rydberg_global')
            
        # Get all atoms in the register
        atoms = list(self.register.qubits.keys())
        
        if has_texture and use_texture:
            # Extract texture features for each node
            texture_features = self.register.metadata.get('texture_features', {})
            
            # Default texture value if not available
            default_texture = 0.5
            
            for atom in atoms:
                # Target each atom individually
                seq.target(atom, 'local_pulse')
             
                # Get this node's texture value or use default
                node_texture = texture_features.get(atom, default_texture)
                
                # Scale duration based on texture value
                #duration = int(self.base_duration * (0.5 + node_texture))
                duration = int(self.base_duration * (0.25 + 1.5 * node_texture))

                # Round duration to a multiple of the device's clock period (4)
                duration = 4 * round(duration / 4) 

                # Also modify the amplitude to encode texture information
                amplitude = self.base_amplitude * (0.5 + node_texture)
                
                # Create node-specific pulse with duration proportional to texture
                pulse = Pulse.ConstantAmplitude(
                    amplitude=amplitude,
                    detuning=RampWaveform(duration, 0, 0),
                    phase=0.0
                )
                                
                # Add pulse to the properly targeted channel
                seq.add(pulse, 'local_pulse')
        else:
            # If no texture info, use default pulses for all atoms
            # Target all atoms at once
            seq.target(atoms, 'local_pulse')
            
            pulse = Pulse.ConstantAmplitude(
                amplitude=self.base_amplitude,
                detuning=RampWaveform(self.base_duration, 0, 0),
                phase=0.0
            )
            seq.add(pulse, 'local_pulse')
        
        # Also add rydberg global pulse, even if there is no texture
        rydberg_duration = 1000*self.global_duration_coef  # Standard duration for Rydberg pulse (in ns)
        rydberg_amplitude = 2.5 * np.pi  # rad/μs
        rydberg_pulse = Pulse.ConstantAmplitude(
            amplitude=rydberg_amplitude,
            detuning=RampWaveform(rydberg_duration, 0, 0),
            phase=0.0
        )
        seq.add(rydberg_pulse, 'rydberg_glob')
        
        
        try:
            seq.measure("ground-rydberg")
            #seq.measure("digital") # FOR RAMAN
        except Exception as e:
            # Just log the error but don't fail if measurement isn't supported
            print(f"Warning: Could not add measurement: {str(e)}")
        
        return seq