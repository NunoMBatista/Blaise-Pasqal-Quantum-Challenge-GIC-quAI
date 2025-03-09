# Pulser Implementation

## Overview
This folder presents both classical and quantum results for the pulser-based implementation. It provides benchmarking data, model performance insights, and comparisons between traditional and quantum approaches to classification tasks.

This repository is structured into two main sections:
- **Classical Machine Learning**: Implements kernel-based Support Vector Machines (SVM) for classification tasks.
- **Quantum Machine Learning**: Explore the quantum evolution kernel (QEK), evaluates their performance, and compares them with classical methods.

## Project Structure
```
Pulser_impl/
│── README.md                          # Project documentation
│
├── classical/                         # Classical machine learning models
│   ├── classical_kernel_svm_benchmarking.ipynb  # Jupyter notebook for classical SVM benchmarking
│   ├── classical_svm_roc_curve.png    # ROC curve visualization for classical SVM
│   ├── Classic_Results.txt        # Final classification results for the classical model
│
├── quantum/                           # Quantum machine learning models
│   ├── model_evaluation_YYYY-MM-DD_HH-MM-SS.json  # Model performance metrics
│   ├── quantum_confusion_matrix.png   # Confusion matrix visualization for the quantum model
```

## Classical Implementation
- The classical approach utilizes **Kernel Support Vector Machines (SVM)** to perform classification tasks on structured data.
- The primary implementation is in `classical_kernel_svm_benchmarking.ipynb`, which contains:
  - Data preprocessing
  - Model training and hyperparameter tuning
  - Performance evaluation through accuracy, precision, recall, and ROC curves
  
- The **ROC curve** (`classical_svm_roc_curve.png`) visually represents the trade-off between sensitivity and specificity.
- The final classification results are stored in `Classic_Results.txt`, summarizing the model’s accuracy and other metrics.

## Quantum Implementation
- The quantum approach aims to evaluate machine learning models utilizing **quantum circuits**.
- Evaluation results are stored in JSON format (`model_evaluation_YYYY-MM-DD_HH-MM-SS.json`) and include:
  - Classification accuracy
  - Confusion matrix data
  - Model-specific quantum metrics (if applicable)
- The **confusion matrix** (`quantum_confusion_matrix.png`) visually represents the distribution of predictions.
- The quantum model's performance is compared to the classical approach to assess potential advantages.

## Results and Comparison
### Classical vs. Quantum Performance
- The classical SVM model achieved an **accuracy of 63.6%**, demonstrating stable and reliable classification performance.
- The quantum model achieved an **accuracy of 62.5%**, showing competitive results but slightly lower than the classical approach.
- The **ROC curve analysis** suggests that the classical model had a slightly better balance between precision and recall.
- The **confusion matrix for the quantum model** indicates a slightly higher number of misclassifications compared to the classical model, likely due to limitations in quantum circuit design and hardware noise.

### Key Observations
- **Computational Efficiency**: The classical SVM model is more efficient in terms of execution time, while the quantum model requires specialized hardware and can be affected by noise.
- **Scalability**: The classical model maintains a smaller higher accuracy even as complexity increases, scaling with an O(n²) complexity for larger images, while the quantum model scales with O(2 log n), allowing for potential performance improvements as quantum hardware advances.
- **Accuracy Comparison**:
  - The classical model outperformed the quantum model by **1.1 percentage points**, which suggests that classical methods remain more reliable under current conditions.
  - However, quantum models hold promise for future advancements, particularly in solving complex, high-dimensional problems where classical models may struggle.


