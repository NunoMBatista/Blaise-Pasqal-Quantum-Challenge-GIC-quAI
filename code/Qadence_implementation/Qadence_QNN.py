import os
import itertools as it
import csv
import random
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from tqdm import tqdm

from qadence import (
    chain, kron, QuantumCircuit, run,
    H, X, MCRY, Z, RY, RX, CNOT,
    QNN, VariationalParameter, FeatureParameter, hea
)

from qadence.draw import display

import torch
from torch import tensor
from torch.optim import Adam

from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split

# Constants
IMAGE_SIZE = 4  # 4x4 pixels
# Paths to image directories
WITH_ELLIPSE_DIR = "../../data/Qadence_implementation/dataset/with_ellipse"
WITHOUT_ELLIPSE_DIR = "../../data/Qadence_implementation/dataset/without_ellipse"
GRAYSCALE_LEVELS = 8  # Number of grayscale levels

def load_images_from_directory(directory, label):
    """Load all images from a directory and assign them a label."""
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            filepath = os.path.join(directory, filename)
            
            # Load the image and convert to grayscale
            img = Image.open(filepath).convert('L')
            
            # Resize to IMAGE_SIZE x IMAGE_SIZE if needed
            if img.size != (IMAGE_SIZE, IMAGE_SIZE):
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            
            # Convert to numpy array
            img_array = np.array(img)
            
            # If the image has more grayscale levels than GRAYSCALE_LEVELS, 
            # quantize it to GRAYSCALE_LEVELS
            if np.max(img_array) > 0:  # Avoid division by zero
                img_array = np.round(img_array / 255 * (GRAYSCALE_LEVELS - 1)) * (255 // (GRAYSCALE_LEVELS - 1))
            
            images.append({
                'image': img_array,
                'label': label,
                'filename': filename
            })
    return images

# Load images from both directories
print("Loading images with ellipses...")
with_ellipse_images = load_images_from_directory(WITH_ELLIPSE_DIR, True)

print("Loading images without ellipses...")
without_ellipse_images = load_images_from_directory(WITHOUT_ELLIPSE_DIR, False)

# Combine datasets
dataset = with_ellipse_images + without_ellipse_images

# Shuffle the dataset
random.shuffle(dataset)

# If dataset is very large, you might want to limit it
NUM_IMAGES = min(len(dataset), 512)  # Cap at 512 images (or adjust as needed)
dataset = dataset[:NUM_IMAGES]

print(f"Loaded {len(dataset)} images in total")

# Create DataFrame
df_train = pd.DataFrame(dataset)

# Split into train and test sets (70/30 split)
df_train, df_test = train_test_split(df_train, test_size=0.3, random_state=42)

print(f"Training set: {len(df_train)} images")
print(f"Test set: {len(df_test)} images")

# Parameters to input the image into the circuit
# One variable per pixel, indexed by (row, col)
feature_params = {
    (row, col): FeatureParameter(f'x{row}{col}')
    for row in range(IMAGE_SIZE)
    for col in range(IMAGE_SIZE)
}

feature_params_list = list(feature_params.values())

# Quantum parameters
qdim = math.floor(math.log2(IMAGE_SIZE))  # number of qubits to encode a row or a column
n_qubits = 2*qdim + 1  # 6 for position (3 row + 3 column) + 1 for intensity
control_qubits = list(range(n_qubits-1))

def control_state(row: int, col: int):
    row_binary = f"{row:b}".rjust(qdim, '0')
    col_binary = f"{col:b}".rjust(qdim, '0')
    state = row_binary + col_binary
    return state

ops_feature = [kron(H(i) for i in range(n_qubits - 1))]  # Hadamard for position

for (row, col), parameter in feature_params.items():
    rotation_gate = kron(MCRY(control_qubits, n_qubits - 1, parameter * np.pi))
    
    cstate = control_state(row, col)
    qubits_to_flip = [i for i, x in enumerate(cstate) if x == '1']

    if qubits_to_flip:
        ops_feature.append(kron(X(i) for i in qubits_to_flip))
        
    ops_feature.append(rotation_gate)

    if qubits_to_flip:
        ops_feature.append(kron(X(i) for i in qubits_to_flip))

chain_feature = chain(*ops_feature)
qc_feature = QuantumCircuit(n_qubits, chain_feature)
# display(qc_feature)

depth = 5
qc_ansatz = hea(n_qubits, depth)
# display(qc_ansatz)

obs_parameters = [VariationalParameter(f'z{i}') for i in range(n_qubits)]
observable = sum(obs_parameters[i] * Z(i) for i in range(n_qubits)) / n_qubits

ops_all = chain(*qc_feature, qc_ansatz)
qc = QuantumCircuit(n_qubits, ops_all)
model = QNN(qc, observable, inputs=feature_params_list)

# Prepare training data
img_train = np.stack(df_train.image.values) / 255 * np.pi
y_train = torch.tensor(np.array(df_train.label), dtype=torch.float64) * 2 - 1

input_dict_train = {
    feature.name: torch.tensor([img[i, j] for img in img_train], dtype=torch.float64)
    for (i, j), feature in feature_params.items()
}

criterion = torch.nn.SoftMarginLoss()  # SoftMarginLoss is advised for data classification

def loss_fn(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Loss function encoding the problem to solve."""
    # Equation loss
    model_output = model.expectation(values=input_dict_train)
    
    loss = criterion(model_output.squeeze(), y_train)

    return loss


optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
n_epochs = 3000

for epoch in range(n_epochs):
    optimizer.zero_grad()
    loss = loss_fn(model, input_dict_train)
    loss.backward()
    optimizer.step()

    if epoch % 25 == 0:
        print(f"{epoch=}, Loss: {loss.item():.4f}")

print(f"Final loss: {loss.item():.4f}")

# Prepare test data
img_test = np.stack(df_test.image.values) / 255 * np.pi
y_test = torch.tensor(np.array(df_test.label), dtype=torch.float64) * 2 - 1

input_dict_test = {
    feature.name: torch.tensor([img[i, j] for img in img_test], dtype=torch.float64)
    for (i, j), feature in feature_params.items()
}

y_pred = model.expectation(values=input_dict_test).squeeze()

from collections import defaultdict

@np.vectorize
def classify(circuit_output):
    return int(bool(circuit_output > 0))

test_values = classify(y_test.detach().numpy())
predicted_values = classify(y_pred.detach().numpy())

confusion_matrix = np.zeros((2, 2), dtype=int)
for predicted_value, test_value in zip(predicted_values, test_values):
    confusion_matrix[predicted_value, test_value] += 1

print("Confusion Matrix:")
print(confusion_matrix)

# Calculate accuracy
accuracy = (confusion_matrix[0, 0] + confusion_matrix[1, 1]) / len(df_test)
print(f"Accuracy: {accuracy:.4f}")

# Visualize confusion matrix
fig, ax = plt.subplots()

sns.heatmap(
    confusion_matrix / len(df_test),
    annot=True,
    fmt=".2%",
    cmap="Blues",
    ax=ax,
)

ax.set_xticklabels(["Real\n Negative", "Real\n Positive"])
ax.set_yticklabels(["Predicted\n Negative", "Predicted\n Positive"])

plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()