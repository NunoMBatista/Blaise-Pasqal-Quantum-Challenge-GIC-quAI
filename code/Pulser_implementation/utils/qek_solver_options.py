"""
Utilities for configuring ODE solver options in quantum simulations.
"""
import numpy as np
from scipy.integrate import ode
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ODESolverOptions:
    """
    Options for configuring the ODE solver used in quantum simulations.
    
    Parameters:
        nsteps (int): Maximum number of internal steps to take for each integration step
        method (str): Integration method to use ('zvode' for complex ODEs)
        max_retries (int): Maximum number of retries with increased nsteps on failure
        retry_factor (float): Factor to multiply nsteps by on each retry
    """
    nsteps: int = 200000
    method: str = 'zvode'
    max_retries: int = 3
    retry_factor: float = 5.0
    
    # Additional ODE solver options
    atol: float = 1e-12
    rtol: float = 1e-12
    order: int = 12


"""
    behaviour:
        Configure a scipy ODE solver with the specified options.
    input:
        solver - The scipy.integrate.ode solver to configure
        options - Solver options; if None, defaults will be used
    output:
        The configured solver
"""
def configure_ode_solver(solver: ode, options: Optional[ODESolverOptions] = None) -> ode:
    if options is None:
        options = ODESolverOptions()
        
    # Set integration method
    solver.set_integrator(
        options.method,
        nsteps=options.nsteps,
        atol=options.atol,
        rtol=options.rtol,
        order=options.order
    )
    return solver


"""
    behaviour:
        Run an ODE solver function with automatic retry on solver errors.
    input:
        solve_func - Function that performs the ODE solve operation
        options - Solver configuration options
        **kwargs - Additional arguments to pass to solve_func
    output:
        The result from the solve_func
"""
def solve_with_retry(solve_func, options: Optional[ODESolverOptions] = None, **kwargs) -> Any:
    if options is None:
        options = ODESolverOptions()
        
    current_nsteps = options.nsteps
    
    for attempt in range(options.max_retries + 1):
        try:
            # Don't pass options to the solve_func - it's just for our retry logic
            # Run the function with only the kwargs it expects
            result = solve_func(**kwargs)
            return result
            
        except Exception as e:
            if "Excess work done" in str(e) and attempt < options.max_retries:
                # Increase nsteps for the next attempt
                current_nsteps = int(current_nsteps * options.retry_factor)
                print(f"ODE solver warning: Increasing nsteps to {current_nsteps} (attempt {attempt+1}/{options.max_retries})")
                
                # Try to update ODE solver settings from outside
                # This is a bit of a hack but might work depending on how the solve_func is configured
                if hasattr(solve_func, "solver_kwargs"):
                    solve_func.solver_kwargs['nsteps'] = current_nsteps
                elif hasattr(solve_func, "solver"):
                    if hasattr(solve_func.solver, "nsteps"):
                        solve_func.solver.nsteps = current_nsteps
            else:
                # Re-raise the error if it's not the "excess work" error or we're out of retries
                raise Exception(f"ODE solver error after {attempt+1} attempts: {str(e)}")
