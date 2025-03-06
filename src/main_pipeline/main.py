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

# Specify the directories 
no_polyp_dir = os.path.join(os.getcwd(), 'dataset', 'synthetic_colon_data', 'no_polyp')
polyp_dir = os.path.join(os.getcwd(), 'dataset', 'synthetic_colon_data', 'polyp')
print(f"Loading data from:\n- No polyp: {no_polyp_dir}\n- Polyp: {polyp_dir}")

plt.ion()


N_QUBITS = 20
# Create datasets for each class (with labels)
no_polyp_dataset = ImageGraphDataset(
    img_dir=no_polyp_dir,
    max_samples=200,
    n_segments=N_QUBITS,
    use_superpixels=True,
    label=0  # Label 0 for no polyp
)

polyp_dataset = ImageGraphDataset(
    img_dir=polyp_dir,
    max_samples=200,
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
        graph = qek_graphs.BaseGraph(
            id=i,
            data=compatible_data,
            device=pl.AnalogDevice
        )
        graph.target = compatible_data.y.item()  # Preserve the class label
        graphs_to_compile.append(graph)
    except ValueError as e:
        print(f"Graph {i} could not be made compatible: {str(e)}")
    except Exception as e:
        print(f"Unexpected error with graph {i}: {str(e)}")


    
# Compile graphs to pulse and register
compiled = []

for i, graph in enumerate(tqdm(graphs_to_compile)):
    try:
        # Access the graph data from original_data which preserves the texture info
        # BaseGraph stores minimal information and doesn't have all attributes
        
        # Custom register compilation using our texture-aware function
        original_graph_data = original_data[i]
        custom_register = graph_to_quantum_register(
            original_graph_data, 
            texture_feature="pca",
            scale_factor=5
        )
        
        # Assign the register to the graph and compile pulse
        register = custom_register  # Use our custom register
        graph.register = register   # Assign it to the graph
        pulse = graph.compile_pulse()
        
        # Store the successful compilation
        compiled.append((graph, original_graph_data, pulse))
    except CompilationError as e:
        # Skip
        print(f"Graph {graph.id} failed compilation. Error: {e}")
    except Exception as e:
        print(f"Unexpected error during compilation for graph {graph.id}: {str(e)}")
        
print(f"Compiled {len(compiled)} graphs out of {len(graphs_to_compile)}.")
        


print(compiled[2])
example_graph, example_data, example_pulse = compiled[2]
example_register = example_graph.register
example_register.draw(blockade_radius=pl.AnalogDevice.max_radial_distance + 0.01)
example_pulse.draw()


"""
EXECUTING ON AN EMULATOR
"""

from qek.data.processed_data import ProcessedData
from qek.backends import QutipBackend

processed_dataset = []
executor = QutipBackend(device=pl.AnalogDevice)

async def process_graphs():
    for graph, original_data, pulse in tqdm(compiled):
        states = await executor.run(register=graph.register, pulse=pulse)
        processed_dataset.append(ProcessedData.from_register(
            register=graph.register,
            pulse=pulse,
            device=pl.AnalogDevice,
            state_dict=states,
            target=graph.target
        ))

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
print(f"Polyp (1): {class_counts.get(1, 0)}")

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

print("\nEvaluation Results:")
print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted')}")
print(f"Balanced Accuracy Score: {balanced_accuracy_score(y_test, y_pred)}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Polyp', 'Polyp']))