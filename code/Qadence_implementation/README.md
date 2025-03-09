# Quantum Image Classification Readme

## Overview
This code provides a framework for performing image classification using quantum circuits. It uses a hybrid quantum-classical approach to process grayscale images, encode them into quantum circuits, and train a quantum neural network (QNN) to classify the images.


## Data
- **Image Directories:**  
  - `../../data/Qadence_implementation/dataset/with_ellipse` (path to images containing ellipses)  
  - `../../data/Qadence_implementation/dataset/with_ellipse` (path to images without ellipses)  
  - `../../data/Qadence_implementation/dataset/labels.csv` (path to labels)  
  Images must be grayscale and will be resized to 4x4 pixels.  
- **Grayscale Levels:**  
  Images are quantized to 8 levels of grayscale intensity.

## Workflow
1. **Load and Preprocess Images:**  
   - Images are loaded from the specified directories.  
   - Grayscale images are resized, normalized, and quantized.

2. **Feature Encoding:**  
   - Each pixel intensity is mapped to a feature parameter.  
   - Quantum states represent the pixel positions and intensities.  
   - Hadamard gates are applied for positional superposition, and multi-controlled rotations encode intensity values.

3. **Quantum Circuit Construction:**  
   - A quantum feature map encodes image data.  
   - A variational ansatz applies multiple layers of parameterized gates.  
   - The observable measures the quantum state to produce a classification output.

4. **Training the QNN:**  
   - The QNN is trained using PyTorch’s optimization routines.  
   - A soft margin loss function is employed for binary classification.  
   - Training adjusts the variational parameters to minimize the classification error.

5. **Evaluation:**  
   - The test set is used to evaluate the trained QNN.  
   - A confusion matrix and accuracy score provide performance metrics.  
   - A confusion matrix heatmap is saved and displayed.

## Running the Code
1. **Prepare the Environment:**  
   - Ensure all required Python libraries and the Qadence framework are installed.
   - run
   ```python
    python3 Qadence_QNN.py
    ```

2. **Adjust Constants:**  
   - Set `IMAGE_SIZE`, `GRAYSCALE_LEVELS`, `WITH_ELLIPSE_DIR`, and `WITHOUT_ELLIPSE_DIR` as needed.

## Output
- Final loss and accuracy are printed to the console.
- A confusion matrix heatmap is saved as `confusion_matrix.png`.

## Notes
- The code is modular and can be extended to larger image sizes (resulting in complex circuits that can take a while to execute), different datasets, or more complex quantum architectures.
- Adjusting the number of training epochs and learning rates may help improve performance.