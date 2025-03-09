# Qadence Implementation

This folder contains the results of the Qadence-based implementation, which includes both classical and quantum-based approaches for classification tasks. The project includes confusion matrices, performance curves, and result files for different configurations.

## Project Structure

```
Qadence_implementation/
│── classical/
│   ├── Classical_confusion_matrix_4x4.png
│   ├── Classical_confusion_matrix_8x8.png
│   ├── Classical_curves_4x4.png
│   ├── Classical_curves_8x8.png
│   ├── Classic_Results_4x4.txt
│   ├── Classic_Results_8x8.txt
│   ├── pytorch_classic.ipynb
│
│── quantum/
│   ├── Confusion_matrix_4x4_8x8_512images_1350ep_0.4loss_0.1lr.png
│   ├── Confusion_matrix_8x8_512images_3000ep_0.38loss_0.1lr.png
│   ├── Quantum_confusion_matrix_4x4.png
│   ├── Quantum_Results_4x4.txt
│   ├── Quantum_Results_8x8.txt
```

## Classical Approach
The `classical` directory contains results obtained using a classical deep learning model implemented in PyTorch.

- **pytorch_classic.ipynb**: Jupyter Notebook implementing the classical model.
- **Classical_confusion_matrix_*.png**: Confusion matrices for diferent image sizes.
- **Classical_curves_*.png**: Performance curves for diferent image sizes.
- **Classic_Results_*.txt**: Text files containing numerical results of the classical approach.

## Quantum Approach
The `quantum` directory contains results from a quantum-based classification model.

- **Quantum_confusion_matrix_*.png**: Confusion matrices for  diferent image sizes.
- **Confusion_matrix_*ep_*.png**: Confusion matrices for diferent image sizes.
- **Quantum_Results_*.txt**: Text files containing numerical results for diferent image sizes of the quantum approach.

## Results and Comparison

### Classical Model:
- **4x4 Images:**
  - Accuracy: **94.87%**
  - Balanced precision and recall across classes.
- **8x8 Images:**
  - Accuracy: **89.74%**
  - Slight drop in performance but still maintains strong results.

### Quantum Model:
- **4x4 Images:**
  - Accuracy: **90.6%**
  - False negatives: **13.8%**, False positives: **2.9%**
- **8x8 Images:**
  - Accuracy: **69.1%**
  - Higher error rates, particularly in false negatives (37.2%).

### Key Insights:
1. **Performance:** The classical model **outperforms** the quantum model in accuracy in both cases.
2. **False Negatives:** The classical model maintains higher accuracy even as complexity increases, scaling with an O(n²) complexity for larger images, while the quantum model scales with O(2 log n), allowing for potential performance improvements as quantum hardware advances.
4. **Potential of Quantum Computing:** While the current quantum model lags behind, improvements in quantum hardware and optimization techniques could enhance its performance.



