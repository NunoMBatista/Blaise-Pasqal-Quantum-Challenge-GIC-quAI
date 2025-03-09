import asyncio
from tqdm import tqdm
from qek.data.processed_data import ProcessedData
from qek.backends import QutipBackend
import pulser as pl
from utils.qek_backend_utils import prepare_for_qek_backend, configure_backend_for_stability

async def execute_quantum_simulation(compiled_graphs, nsteps=50000, nsteps_high=250000):
    """Execute compiled graphs on quantum simulator"""
    processed_dataset = []
    
    # Configure backend with optimized solver settings
    executor = QutipBackend(device=pl.DigitalAnalogDevice)
    executor = configure_backend_for_stability(executor, nsteps=nsteps)

    for graph, original_data, sequence in tqdm(compiled_graphs):
        try:
            # Create compatible objects for backend
            register, compatible_pulse = prepare_for_qek_backend(graph, sequence)
            
            try:
                # Run simulation
                states = await executor.run(
                    register=register, 
                    pulse=compatible_pulse
                )
                
                # Store results
                processed_dataset.append(ProcessedData.from_register(
                    register=graph.register,
                    pulse=compatible_pulse,
                    device=pl.DigitalAnalogDevice,
                    state_dict=states,
                    target=graph.target
                ))
                
            except Exception as e:
                if "Excess work done" in str(e):
                    print(f"ODE solver error with graph {graph.id}, retrying with higher nsteps...")
                    # Try with higher nsteps
                    temp_executor = configure_backend_for_stability(
                        QutipBackend(device=pl.DigitalAnalogDevice), 
                        nsteps=nsteps_high
                    )
                    states = await temp_executor.run(register=register, pulse=compatible_pulse)
                    
                    processed_dataset.append(ProcessedData.from_register(
                        register=graph.register,
                        pulse=compatible_pulse,
                        device=pl.DigitalAnalogDevice,
                        state_dict=states,
                        target=graph.target
                    ))
                else:
                    raise e
                    
        except Exception as e:
            print(f"Error processing graph {graph.id}: {str(e)}")
    
    return processed_dataset


def run_quantum_execution(compiled_graphs, nsteps=50000, nsteps_high=250000):
    """Wrapper to call async execution function"""
    return asyncio.run(execute_quantum_simulation(compiled_graphs, nsteps, nsteps_high))