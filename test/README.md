# README

## Minimum Working Examples (MWE) for Quantum Implementations

This repository contains two **Minimum Working Examples (MWEs)** for implementing quantum machine learning models using **Qadence** and **Pulser** frameworks.

## Directory Structure
```
├── Qadence_implementation
│   ├── MWE_QNN_Qadence.ipynb
│
└── Pulser_implementation
    ├── MWE_Pulser.ipynb
```

---

## 1. Qadence Implementation

**Notebook:** `Qadence_implementation/MWE_QNN_Qadence.ipynb`

### Overview
This notebook demonstrates a **Quantum Neural Network (QNN)** for binary image classification using **Qadence**. The goal is to classify images as containing an **ellipse** or not. The model follows a hybrid **quantum-classical** approach:
1. **Classical images** are encoded into quantum states.
2. A **Quantum circuit** processes the encoded data.
3. Measurement results are used for **classification**.

### Key Components
- **Data Generation**: Creates synthetic 4×4 grayscale images with and without ellipses.
- **Quantum Neural Network (QNN)**:
  - **Feature Circuit:** Encodes image data into quantum states.
  - **Variational Circuit:** Trainable quantum circuit to process encoded data.
  - **Measurement & Classification:** Quantum observables determine the output.
- **Training & Evaluation**:
  - Uses **PyTorch** and **Qadence** to optimize quantum parameters.
  - Employs **confusion matrix analysis** for performance evaluation.

---

## 2. Pulser Implementation

**Notebook:** `Pulser_implementation/MWE_Pulser.ipynb`

### Overview
This notebook implements a **Quantum Evolution Kernel (QEK)** for training a **Support Vector Machine (SVM)** classifier. The task involves **synthetic gastrointestinal polyp classification**, using **Pasqal's quantum hardware simulation via Pulser**.

### Key Components
- **Data Preparation**:
  - Loads images of **polyps and non-polyps**.
  - Converts images into **graph structures**.
- **Quantum Kernel Calculation**:
  - Converts graphs into **quantum registers**.
  - Encodes texture features as **quantum states**.
  - Simulates quantum evolution on **Pasqal’s DigitalAnalogDevice**.
- **Training & Evaluation**:
  - Computes a **Quantum Evolution Kernel (QEK)**.
  - Trains an **SVM classifier** on quantum-processed data.
  - Visualizes results and performance metrics.

---

## Running the Notebooks
1. Open Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
1. Navigate to the respective folder and run the `.ipynb` file.

---
