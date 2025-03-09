# Pulser Implementation

## Overview
This repository provides an implementation of quantum-enhanced kernels (QEK) using the Pulser library. It includes a pipeline for processing images into graphs, applying quantum kernel methods, and executing models with quantum backends. This approach allows leveraging quantum computing for graph-based machine learning tasks, enhancing computational efficiency and exploring quantum advantages.

## Directory Structure

- `data_utils/`: Handles image and graph dataset management, including loading and preprocessing of datasets used in the quantum-enhanced kernel workflow.
  - `image_graph_dataset.py`: Defines dataset structures for transforming images into graphs for further processing.
- `graphs/`: Functions for transforming images into graphs, implementing texture-aware graph processing, and converting graph structures into quantum registers.
  - `image_to_graph.py`: Converts raw images into graph representations.
  - `graph_to_register.py`: Maps graph structures onto quantum registers for quantum processing.
  - `texture_aware_graph.py`: Implements texture-aware graph transformations to improve feature extraction.
- `pipeline/`: The main pipeline that orchestrates the execution, including:
  - `config.py`: Defines configuration settings for different processing stages.
  - `data_loader.py`: Handles data loading and pre-processing, ensuring compatibility with different dataset formats.
  - `graph_compiler.py`: Compiles processed graphs into a format suitable for quantum execution.
  - `main.py`: The entry point for running the entire pipeline.
  - `model_trainer.py`: Implements the training of quantum-enhanced kernel models and comparison with classical methods.
  - `quantum_executor.py`: Manages quantum computations, including backend selection and execution, ensuring seamless integration with quantum hardware or simulators.
  - `visualize_examples.py`: Provides utilities for visualizing quantum-enhanced results and debugging.
- `qek/`: Core quantum-enhanced kernel (QEK) functionality, including:
  - `backends.py`: Manages different quantum computing backends, allowing users to switch between classical simulations and real quantum processors.
  - `data/`: Submodule containing utilities for data extraction, graph processing, and dataset handling.
    - `extractors.py`: Defines various feature extraction methods.
    - `graphs.py`: Provides utilities for graph-based representation of data.
    - `processed_data.py`: Structures and stores processed data for model training and execution.
  - `kernel/`: Defines quantum kernel computations, implementing various quantum kernel methods for comparison.
  - `shared/`: Common utility functions such as error handling and retry mechanisms.
- `utils/`: Various utility functions that assist in backend compatibility, solver options, and register scaling.
  - `compatibility_utils.py`: Ensures compatibility between different quantum computing libraries.
  - `qek_backend_utils.py`: Provides backend configuration options.
  - `register_scaling.py`: Handles scaling of quantum registers to optimize execution performance.
- `visualization/`: Tools for visualizing processed graphs, kernel outputs, and debugging results.
  - `visualization.py`: Contains visualization utilities for graph representations and quantum kernel results.

## Usage

### Running the Pipeline
To execute the full pipeline, which involves data loading, graph transformation, and quantum execution, run:

```bash
python3  pipeline/main.py
```

This script processes input images, converts them into graph structures, applies quantum-enhanced kernels, and produces results for further analysis.
