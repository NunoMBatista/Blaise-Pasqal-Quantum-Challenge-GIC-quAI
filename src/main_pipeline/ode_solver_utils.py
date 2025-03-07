"""
Utilities for handling ODE solver issues in quantum simulations.
"""

import numpy as np
from qek.backends import QutipBackend
import pulser as pl
from qutip import Options
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def configure_backend_with_options(device=None, **kwargs):
    """
    Configure a QutipBackend with optimized solver options.
    
    Args:
        device: The device to use (defaults to MockDevice if None)
        **kwargs: Additional options to pass to the QutipBackend
        
    Returns:
        Configured QutipBackend instance
    """
    device = device or pl.MockDevice
    
    # Default options for stable ODE solving
    default_options = {
        'nsteps': 50000,        # Increase from default 1000
        'atol': 1e-8,           # Absolute tolerance
        'rtol': 1e-6,           # Relative tolerance
        'method': 'bdf',        # Use backward differentiation formula for stiff problems
        'max_step': 0.01        # Maximum step size
    }
    
    # Update with any user-provided options
    options = {**default_options, **kwargs}
    
    # Create QutipBackend with custom Options
    qutip_options = Options(**options)
    backend = QutipBackend(device=device, options=qutip_options)
    
    logger.info(f"Configured backend with options: {options}")
    return backend

def get_adaptive_backend_for_difficult_case(previous_error=None, attempt=1):
    """
    Get a backend with progressively more robust solver settings based on 
    the type of error encountered and the attempt number.
    
    Args:
        previous_error: The exception that was raised in the previous attempt
        attempt: The attempt number (starting from 1)
        
    Returns:
        QutipBackend configured with appropriate settings
    """
    # Base options that get progressively more extreme
    options = {
        'nsteps': min(50000 * attempt, 500000),  # Increase steps with each attempt
        'atol': 1e-8 * (10 ** (attempt - 1)),    # Relax tolerance with each attempt
        'rtol': 1e-6 * (10 ** (attempt - 1)),    # Relax tolerance with each attempt
    }
    
    # For convergence failures, try different methods
    if previous_error and "convergence" in str(previous_error).lower():
        if attempt == 1:
            # First try BDF which is good for stiff equations
            options['method'] = 'bdf'
        elif attempt == 2:
            # Then try Adams method with relaxed tolerances
            options['method'] = 'adams'
            options['nsteps'] = 100000
        elif attempt == 3:
            # Try vode with more internal steps
            options['method'] = 'vode'
            options['nsteps'] = 200000
        elif attempt >= 4:
            # Last resort: zvode with extremely relaxed tolerances
            options['method'] = 'zvode'
            options['nsteps'] = 500000
            options['atol'] = 1e-6
            options['rtol'] = 1e-4
    
    logger.info(f"Attempt {attempt}: Configuring solver with {options}")
    return configure_backend_with_options(**options)

async def safe_quantum_execution(executor, register, pulse, max_attempts=5):
    """
    Execute quantum simulation with progressive fallback options if errors occur.
    
    Args:
        executor: The initial QutipBackend executor
        register: The quantum register to simulate
        pulse: The pulse sequence to apply
        max_attempts: Maximum number of attempts before giving up
        
    Returns:
        states: The resulting quantum states
        
    Raises:
        Exception: If all attempts fail
    """
    last_error = None
    
    for attempt in range(1, max_attempts + 1):
        try:
            if attempt > 1:
                # Get a more robust executor for retry attempts
                executor = get_adaptive_backend_for_difficult_case(last_error, attempt)
            
            # Run the simulation
            states = await executor.run(register=register, pulse=pulse)
            
            if attempt > 1:
                logger.info(f"Succeeded on attempt {attempt} with modified solver settings")
            
            return states
            
        except Exception as e:
            last_error = e
            logger.warning(f"Attempt {attempt} failed: {str(e)}")
            
            # If we're out of attempts, raise the last error
            if attempt == max_attempts:
                raise Exception(f"All {max_attempts} attempts failed. Last error: {str(e)}")
