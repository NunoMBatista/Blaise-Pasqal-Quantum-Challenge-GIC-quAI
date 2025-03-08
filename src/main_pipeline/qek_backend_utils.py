import numpy as np
import pulser
from pulser import Pulse, Sequence
from pulser.devices import DigitalAnalogDevice
from pulser.waveforms import RampWaveform


def create_compatible_pulse(sequence=None, base_amplitude=2*np.pi, base_duration=660):
    """
    Create a compatible pulse for the QEK backend from a sequence or default parameters.
    
    Args:
        sequence: Optional sequence to extract parameters from
        base_amplitude: Default amplitude if not extracted from sequence
        base_duration: Default duration if not extracted from sequence
        
    Returns:
        A Pulse object compatible with QEK backend
    """
    amplitude = base_amplitude
    duration = base_duration
    
    # Try to extract parameters from the sequence if provided
    if sequence is not None:
        try:
            # Look for pulse parameters in the sequence's channels
            if hasattr(sequence, 'channels'):
                for channel_name, channel in sequence.channels.items():
                    if hasattr(channel, 'pulses') and channel.pulses:
                        # Use parameters from the first pulse we find
                        first_pulse = channel.pulses[0]
                        if hasattr(first_pulse, 'amplitude'):
                            amplitude = first_pulse.amplitude
                        if hasattr(first_pulse, 'duration'):
                            duration = first_pulse.duration
                        break
        except Exception as e:
            print(f"Warning: Could not extract pulse parameters from sequence: {e}")
            print("Using default pulse parameters instead.")
    
    # Create a simple, compatible pulse
    compatible_pulse = Pulse.ConstantAmplitude(
        amplitude=amplitude,
        detuning=RampWaveform(duration, 0, 0),
        phase=0.0
    )
    
    return compatible_pulse


def prepare_for_qek_backend(graph, sequence):
    """
    Prepare compatible objects for QEK backend execution.
    
    Args:
        graph: TextureAwareGraph containing the register
        sequence: Pulser Sequence object
        
    Returns:
        tuple of (register, pulse) suitable for QutipBackend.run()
    """
    register = graph.register
    pulse = create_compatible_pulse(sequence)
    
    return register, pulse


def diagnose_sequence(sequence):
    """
    Output diagnostic information about a pulse sequence
    
    Args:
        sequence: A pulser Sequence object
        
    Returns:
        Dict with diagnostic information
    """
    diagnostics = {}
    
    try:
        diagnostics["type"] = type(sequence).__name__
        
        if hasattr(sequence, "channels"):
            diagnostics["channels"] = list(sequence.channels.keys())
            
            channel_diagnostics = {}
            for name, channel in sequence.channels.items():
                channel_info = {
                    "targets": getattr(channel, "targets", "unknown"),
                    "has_pulses": hasattr(channel, "pulses") and len(channel.pulses) > 0
                }
                if channel_info["has_pulses"]:
                    pulse_info = []
                    for p in channel.pulses:
                        pulse_info.append({
                            "type": type(p).__name__,
                            "amplitude": getattr(p, "amplitude", "unknown"),
                            "duration": getattr(p, "duration", "unknown"),
                        })
                    channel_info["pulse_info"] = pulse_info
                channel_diagnostics[name] = channel_info
            
            diagnostics["channel_details"] = channel_diagnostics
            
        if hasattr(sequence, "register"):
            diagnostics["register_size"] = len(sequence.register.qubits)
            
    except Exception as e:
        diagnostics["error"] = str(e)
        
    return diagnostics


def configure_backend_for_stability(backend, nsteps=50000):
    """
    Configure a QEK backend for improved numerical stability.
    
    Args:
        backend: A QutipBackend or similar backend object
        nsteps: Number of ODE solver steps to allow
        
    Returns:
        The configured backend
    """
    # Check if the backend has solver_kwargs attribute
    if hasattr(backend, 'solver_kwargs'):
        # Update the solver parameters
        backend.solver_kwargs.update({
            'nsteps': nsteps,
            'atol': 1e-10,
            'rtol': 1e-10,
            'method': 'bdf' if hasattr(backend, 'supports_bdf') and backend.supports_bdf else 'zvode'
        })
    
    # Some backends might use options attribute instead
    elif hasattr(backend, 'options'):
        if isinstance(backend.options, dict):
            backend.options['nsteps'] = nsteps
            backend.options['atol'] = 1e-10
            backend.options['rtol'] = 1e-10
    
    return backend
