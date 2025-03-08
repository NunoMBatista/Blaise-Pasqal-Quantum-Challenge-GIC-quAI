"""
Utilities for scaling atom positions in quantum registers.
"""
import numpy as np
import torch
from pulser.devices import DigitalAnalogDevice

def scale_positions_to_register_dim(positions, register_dim=30):
    """
    Scale atom positions to fit within the specified register dimension.
    
    Args:
        positions: numpy array or torch tensor of atom positions
        register_dim: target dimension for the register in microns
    
    Returns:
        Scaled positions that fit within the register_dim while preserving
        the minimum atom distance requirements
    """
    # Convert to numpy if tensor
    is_tensor = isinstance(positions, torch.Tensor)
    if is_tensor:
        positions = positions.detach().numpy()
    
    # Center positions around origin
    center = np.mean(positions, axis=0)
    centered_positions = positions - center
    
    # Calculate current maximum dimension
    max_distance = np.max(np.abs(centered_positions))
    
    # Calculate scaling factor to fit within register_dim (with 10% margin)
    target_max = register_dim * 0.9 / 2  # Half dimension with margin
    if max_distance > 0:
        scale_factor = target_max / max_distance
    else:
        scale_factor = 1.0
    
    # Scale positions
    scaled_positions = centered_positions * scale_factor
    
    # Ensure minimum atom distance is respected
    min_required_distance = DigitalAnalogDevice.min_atom_distance
    adjust_positions_for_min_distance(scaled_positions, min_required_distance)
    
    # Convert back to tensor if needed
    if is_tensor:
        return torch.tensor(scaled_positions, dtype=torch.float)
    
    return scaled_positions

def adjust_positions_for_min_distance(positions, min_distance=5.0, max_attempts=100):
    """
    Adjust positions to ensure minimum distance between atoms.
    Uses a simple repulsion model to move atoms away from each other.
    
    Args:
        positions: numpy array of atom positions to adjust in-place
        min_distance: minimum allowed distance between atoms
        max_attempts: maximum number of adjustment iterations
    """
    n_atoms = positions.shape[0]
    if n_atoms <= 1:
        return
    
    # Iteratively adjust positions
    for _ in range(max_attempts):
        violations = False
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < min_distance:
                    # Calculate repulsion direction
                    direction = positions[i] - positions[j]
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                    else:
                        direction = np.random.randn(2)
                        direction = direction / np.linalg.norm(direction)
                    
                    # Move atoms apart
                    adjustment = direction * (min_distance - dist) * 0.5
                    positions[i] += adjustment
                    positions[j] -= adjustment
                    violations = True
        
        if not violations:
            break
