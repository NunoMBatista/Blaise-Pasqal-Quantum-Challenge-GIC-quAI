import math 
import torch 
import numpy as np 
import networkx as nx
from torch_geometric.data import Data, Dataset
import os
import matplotlib.pyplot as plt
import glob
from pathlib import Path

from tqdm import tqdm
import pulser as pl
import rdkit
import qek.data.graphs as qek_graphs

from qek.shared.error import CompilationError
from image_to_graph import load_image, superpixel_to_graph, pixel_to_graph, visualize_graph_with_texture
from graph_to_register import graph_to_quantum_register, visualize_register_with_connections

import os

from image_graph_dataset import ImageGraphDataset
from compatibility_utils import make_compatible_with_analog_device

from texture_aware_graph import TextureAwareGraph
from visualization import visualize_texture_pulse_effects

# Specify the directories 
no_polyp_dir = os.path.join(os.getcwd(), 'dataset', 'synthetic_colon_data', 'no_polyp')
polyp_dir = os.path.join(os.getcwd(), 'dataset', 'synthetic_colon_data', 'polyp')
print(f"Loading data from:\n- No polyp: {no_polyp_dir}\n- Polyp: {polyp_dir}")


# COMMENT THIS LINE TO TURN ON EXAMPLE DISPLAYS
plt.ion()

# TODO: ?? DYNAMICALLY CHANGE THE ATOM REGISTER SIZE BASED ON THE IMAGE SIZE ??

N_QUBITS = 10
MAX_SAMPLES = 200
REGISTER_DIM = 20 # X*X μm dimension of qubitsA


# Create datasets for each class (with labels)
no_polyp_dataset = ImageGraphDataset(
    img_dir=no_polyp_dir,
    max_samples=MAX_SAMPLES,
    n_segments=N_QUBITS,
    use_superpixels=True,
    label=0  # Label 0 for no polyp
)

polyp_dataset = ImageGraphDataset(
    img_dir=polyp_dir,
    max_samples=MAX_SAMPLES,
    n_segments=N_QUBITS,
    use_superpixels=True,
    label=1  # Label 1 for polyp
)

# Combine datasets
combined_dataset = no_polyp_dataset + polyp_dataset

print(f"""
------------- Dataset created -------------

    - No Polyp Graphs: {len(no_polyp_dataset)}
    - Polyp Graphs: {len(polyp_dataset)}
    - Total Graphs: {len(combined_dataset)}

-------------------------------------------
    """)

# Prepare graphs for compilation
graphs_to_compile = []
original_data = []  # Store the original data separately for later reference

for i, data in enumerate(tqdm(combined_dataset)):
    try:
        # Make the graph compatible with AnalogDevice
        compatible_data = make_compatible_with_analog_device(data)
        original_data.append(compatible_data)  # Store the compatible version for later
        
        # Create BaseGraph - Note: BaseGraph stores data in pyg attribute, not data
        # TextureAwareGraph is an extension of the BaseGraph that encodes texture info
        
        graph = TextureAwareGraph(
            id=i,
            data=compatible_data,
            device=pl.MockDevice
        )
        
        graph.target = compatible_data.y.item()  # Preserve the class label
        graphs_to_compile.append(graph)
    except ValueError as e:
        print(f"Graph {i} could not be made compatible: {str(e)}")
    except Exception as e:
        print(f"Unexpected error with graph {i}: {str(e)}")


    
# Compile graphs to pulse and register
compiled = []

# Add debug information about device
print(f"\nUsing device: {pl.MockDevice.name}")
if hasattr(pl.MockDevice, 'channel_objects'):
    print(f"Available channels: {pl.MockDevice.channel_objects}")
else:
    print("Note: Device does not expose available_channels attribute")

for i, graph in enumerate(tqdm(graphs_to_compile)):
    try:
        # Access the graph data from original_data which preserves the texture info
        # BaseGraph stores minimal information and doesn't have all attributes
        
        # Custom register compilation using our texture-aware function
        original_graph_data = original_data[i]

        custom_register = graph_to_quantum_register(
            original_graph_data, 
            texture_feature="pca",
            register_dim=REGISTER_DIM  # Pass the REGISTER_DIM to control register size
        )
        
        # Assign the register to the graph and compile pulse
        register = custom_register  # Use our custom register
        graph.register = register   # Assign it to the graph
        
        try:
            pulse = graph.compile_pulse(use_texture=True)
            
            # Store the successful compilation
            compiled.append((graph, original_graph_data, pulse))
        except Exception as e:
            print(f"Compilation error for graph {graph.id}: {str(e)}")
            # Try to provide more diagnostic information
            print(f"Register has {len(register.qubits)} atoms")
            if hasattr(graph.device, 'available_channels'):
                print(f"Device channels: {graph.device.channels}")
            
    except Exception as e:
        print(f"Unexpected error during compilation for graph {graph.id}: {str(e)}")
        
print(f"Compiled {len(compiled)} graphs out of {len(graphs_to_compile)}.")
        

# When you want to visualize the full sequence:
try:
    example_graph, example_data, example_pulse = compiled[2]
    
    # Create a sequence for visualization purposes
    example_sequence = example_graph.create_texture_sequence(use_texture=True)
    
    # Try to visualize the sequence 
    example_sequence.draw()
    
    # Instead of using the pulse, use the sequence for visualization
    # This is better because the sequence contains channel and targeting information
    fig = visualize_texture_pulse_effects(example_graph, example_sequence, example_data)
    fig.show()
except Exception as e:
    print(f"Error during visualization: {str(e)}")


"""
EXECUTING ON AN EMULATOR
"""

from qek.data.processed_data import ProcessedData
from qek.backends import QutipBackend
# Import our compatibility utilities
from qek_backend_utils import prepare_for_qek_backend, create_compatible_pulse, configure_backend_for_stability
from qek_solver_options import ODESolverOptions

processed_dataset = []
# Configure the executor with better ODE solver settings upfront
executor = QutipBackend(device=pl.MockDevice)
# Configure the executor for stability with higher nsteps
executor = configure_backend_for_stability(executor, nsteps=50000)

async def process_graphs():
    for graph, original_data, sequence in tqdm(compiled):
        try:
            # Create a compatible pulse for the backend
            register, compatible_pulse = prepare_for_qek_backend(graph, sequence)
            
            try:
                # Run with the compatible objects
                states = await executor.run(register=register, pulse=compatible_pulse)
                
                processed_dataset.append(ProcessedData.from_register(
                    register=graph.register,
                    pulse=compatible_pulse,
                    device=pl.MockDevice,
                    state_dict=states,
                    target=graph.target
                ))
            except Exception as e:
                if "Excess work done" in str(e):
                    print(f"ODE solver error with graph {graph.id}, retrying with higher nsteps...")
                    # Configure with even higher nsteps for this specific difficult case
                    temp_executor = configure_backend_for_stability(QutipBackend(device=pl.MockDevice), nsteps=250000)
                    states = await temp_executor.run(register=register, pulse=compatible_pulse)
                    
                    processed_dataset.append(ProcessedData.from_register(
                        register=graph.register,
                        pulse=compatible_pulse,
                        device=pl.MockDevice,
                        state_dict=states,
                        target=graph.target
                    ))
                else:
                    raise e
                    
        except Exception as e:
            print(f"Error processing graph {graph.id}: {str(e)}")
            # Continue with other graphs even if one fails

import asyncio
asyncio.run(process_graphs())
    
    
dataset_example: ProcessedData = processed_dataset[0]
print(f"""
    Total number of samples: {len(processed_dataset)}
    Example state_dict {dataset_example.state_dict}    
""")

dataset_example.draw_register()
dataset_example.draw_pulse()


from sklearn.model_selection import train_test_split

# Prepare features (X) and targets (y)
X = [data for data in processed_dataset]
y = [data.target for data in processed_dataset]

print("\nClass distribution in processed dataset:")
class_counts = {}
for data in processed_dataset:
    label = data.target
    class_counts[label] = class_counts.get(label, 0) + 1
print(f"No polyp (0): {class_counts.get(0, 0)}")
print(f"Polyp (1): {class_counts.get(1, 1)}")

# Use stratified split to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y, 
    stratify=y,
    test_size=0.2, 
    random_state=42
)

print(f"Size of the training quantum compatible dataset = {len(X_train)}")
print(f"Size of the testing quantum compatible dataset = {len(X_test)}")
print(f"Class distribution - Training: No polyp: {y_train.count(0)}, Polyp: {y_train.count(1)}")
print(f"Class distribution - Testing: No polyp: {y_test.count(0)}, Polyp: {y_test.count(1)}")

X_train[0].draw_excitation()


from qek.kernel import QuantumEvolutionKernel as QEK 

# The Quantum Evolution Kernel will be used to compute a similarity score between 2 graphs

# Initialize the Quantum Evolution Kernel with a hyperparameter mu
kernel = QEK(mu=0.5)

# Fit the kernel to the training data
# This means that the kernel will learn the optimal parameters for the kernel
kernel.fit(X_train)

# Transform
K_train = kernel.transform(X_train)
K_test = kernel.transform(X_test)

print(f"Training Kernel Matrix Shape: {K_train.shape}")
print(f"Testing Kernel Matrix Shape: {K_test.shape}")

from sklearn.svm import SVC

# Define a Support Vector Machine classifier with the Quantum Evolution Kernel
qek_kernel = QEK(mu=0.5)
model = SVC(
    kernel=qek_kernel, 
    random_state=42,
    class_weight='balanced'
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, classification_report

# Add more diagnostics about the model predictions
print("\nModel Prediction Analysis:")
unique_predictions = np.unique(y_pred)
print(f"Unique predicted classes: {unique_predictions}")
print(f"Number of predictions for each class: {np.bincount(y_pred if isinstance(y_pred, np.ndarray) else np.array(y_pred, dtype=int))}")
print(f"True class distribution: {np.bincount(y_test if isinstance(y_test, np.ndarray) else np.array(y_test, dtype=int))}")

# Add probability estimates if the model supports it
if hasattr(model, 'predict_proba'):
    try:
        proba = model.predict_proba(X_test)
        print("\nPrediction probabilities:")
        print(f"Mean probability for class 0: {np.mean(proba[:, 0]):.4f}")
        print(f"Mean probability for class 1: {np.mean(proba[:, 1]):.4f}")
    except Exception as e:
        print(f"Could not get prediction probabilities: {e}")

print("\nEvaluation Results:")
# Set zero_division to avoid warnings
print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted', zero_division=0)}")
print(f"Balanced Accuracy Score: {balanced_accuracy_score(y_test, y_pred)}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Polyp', 'Polyp'], zero_division=0))

# Add a recommendation for small datasets
print("\nNote: For small datasets, consider using:")
print("1. Cross-validation instead of a single train-test split")
print("2. Regularization parameters in the SVM (C parameter)")
print("3. Different kernel parameters (mu value)")

# Optional: Add a cross-validation version
from sklearn.model_selection import cross_val_score, KFold
if len(X) >= 10:  # Only run cross-validation if we have enough data
    print("\nRunning 5-fold cross-validation to get a more robust assessment:")
    try:
        cv_scores = cross_val_score(
            model, 
            X, y, 
            cv=min(5, len(np.unique(y))), 
            scoring='balanced_accuracy'
        )
        print(f"Cross-validation balanced accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    except Exception as e:
        print(f"Could not run cross-validation: {e}")