# GIC-quAI: Quantum AI for Colorectal Cancer Diagnosis

## Introduction

The rapid advancement of classical machine learning (ML) algorithms has significantly enhanced image processing capabilities. However, the energy consumption associated with high-performance computing (HPC) for these tasks is increasingly unsustainable. Classical methods, while effective, face scalability challenges due to the exponential increases in computational demands as data volumes grow.

Gastrointestinal (GI) cancers, particularly colorectal cancer (CRC), represent a significant global health challenge, with over 2 million new cases diagnosed annually. The **G-quAI** project leverages quantum computing to enhance artificial intelligence (AI) models for CRC diagnosis and prevention. By integrating **Quantum Neural Networks (QNNs)** and **Quantum Support Vector Machines (QSVMs)**, the project seeks to accelerate image analysis, improve diagnostic precision, and enhance energy efficiency.

### Key Objectives

- Develop scalable and energy-efficient AI solutions for CRC diagnosis.
- Utilize quantum computing to optimize image classification and feature extraction.
- Investigate the potential of **Quantum Evolutionary Kernels (QEKs)** in AI-driven medical imaging.
- Compare the efficiency of **Quadence-based QNNs** and **Pulser-based Graph ML** approaches.

## Mathematical Formulation & Solution Strategy

### Quadence Quantum Neural Networks

This approach focuses on encoding grayscale images into quantum states using **Flexible Representation of Quantum Images (FRQI)**. The **QNN model** is trained to classify synthetic image datasets generated with gradient backgrounds and geometric shapes. Key features include:

- **Quantum Encoding:** Images are transformed into quantum states for efficient processing.
- **Variational Quantum Circuit (VQC):** Utilized for training and inference.
- **Synthetic Dataset:** 8×8 grayscale images with gradient backgrounds and ellipses.

### Pulser Graph-Based ML

An alternative method leveraging **graph-based quantum state encoding** using **Pasqal's Pulser framework**. This technique represents images as graphs, where:

- Each node in the graph corresponds to a qubit.
- Quantum evolution kernels process image features.
- The dataset includes synthetic **256×256 colonoscopy-like images**, featuring **realistic polyp structures**.

## AI Contribution & Methodology

Two main quantum-enhanced AI techniques are explored:

1. **Quantum Neural Networks (QNNs):**
   - Leverages **Qadence** for FRQI-based image encoding.
   - Uses **quantum circuits** for feature extraction and classification.
2. **Quantum Evolutionary Kernels (QEKs):**
   - Explores **quantum-enhanced evolutionary algorithms**.
   - Utilizes **Pulser-based analog quantum computing** for optimization.

## Image Generation & Dataset

### Synthetic Dataset Generation

To optimize the quantum image processing pipeline, a controlled synthetic dataset is created:

- **Quadence-based QNNs**: 8×8 grayscale images with gradient backgrounds and ellipses.
- **Pulser-based Implementation**: 256×256 colonoscopy-like images with polyp segmentation.

The dataset is stored in **NumPy (.npy) format** and labeled with a CSV file (`labels.csv`).

## Quantum Encoding & Model Implementation

### Quadence-Based QNNs

- **FRQI Encoding**:
  - Encodes pixel intensity into quantum states.
  - Uses **rotation gates** for quantum amplitude representation.
- **Quantum Neural Network Architecture**:
  - **Feature Encoding Layer**: Maps classical image data to quantum states.
  - **Variational Layers**: Alternating **RY rotations and CNOT gates** for entanglement.
  - **Quantum Measurement**: Output obtained via parity measurements.
- **Training Strategy**:
  - Uses **mini-batch gradient descent**.
  - Optimizers: **Adam, AdamW, and SGD**.
  - Loss Function: **Binary Cross-Entropy with Logits (BCEWithLogitsLoss)**.

### Pulser Graph-Based ML

- **Graph-Based Encoding**:
  - Images are converted into **graph representations**.
  - **Quantum Evolution Kernels (QEKs)** process the structured data.
- **Quantum Machine Learning Workflow**:
  - Uses **Pasqal’s neutral atom processors**.
  - Incorporates **Hamiltonian evolutions** for feature extraction.
  - Optimizes **energy-based classification methods**.

## Results, Evaluation & Roadmap

### Performance Evaluation

- **Classification Metrics**: Accuracy, Precision, Recall, F1-score.
- **Confusion Matrix**: Visualizing misclassifications.
- **Energy Efficiency**: Evaluating potential energy savings vs. classical HPC.

### Future Work

- **Hybrid Quantum-Classical Models**: Explore more expressive feature maps.
- **Alternative Encoding Methods**: Investigate NEQR for more compact representation.
- **Optimization Strategies**: Explore gradient-free optimization.
- **Hardware Integration**: Testing models on real quantum processors.

## Repository Structure

```plaintext
📂 GIC-quAI
 ├── 📂 code               # Synthetic datasets
 ├── 📂 data               # Synthetic datasets
 ├── 📂 docs               # Synthetic datasets
 ├── 📂 results            # Synthetic datasets
 ├── 📂 test               # Synthetic datasets
 ├── .gitignore            # Ignored Files
 ├── requirements.txt      # Required packages 
 ├── README.md             # Project documentation
```

## How to Run

1. **Clone the repository**:
   ```sh
   git clone https://github.com/your-repo/GIC-quAI.git
   cd GIC-quAI
   ```
2. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```
3. **Generate synthetic dataset**:
   ```sh
   python data/generate_images.py
   ```
4. **Train QNN model**:
   ```sh
   python models/qadence_qnn.py
   ```
5. **Run Pulser-based ML**:
   ```sh
   python models/pulser_graph_ml.py
   ```

---

**GIC-quAI** aims to push the boundaries of AI-driven medical imaging by integrating quantum computing into cancer diagnostics. 🚀

