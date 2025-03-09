import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
print(sys.path)
import matplotlib.pyplot as plt
import pulser as pl

# Import modules
from pipeline.config import *
from pipeline.data_loader import load_datasets, prepare_graphs_for_compilation
from pipeline.graph_compiler import compile_graphs
from pipeline.quantum_executor import run_quantum_execution
from pipeline.model_trainer import prepare_dataset, split_dataset, train_qek_svm_model, run_cross_validation
from pipeline.visualize_examples import visualize_example, visualize_processed_data

def main():
    """Main pipeline execution function"""
    # Configure matplotlib for interactive mode
    # plt.ion()

    # 1. Load and prepare datasets
    print(f"Loading data from:\n- No polyp: {NO_POLYP_DIR}\n- Polyp: {POLYP_DIR}")
    combined_dataset = load_datasets(
        NO_POLYP_DIR, 
        POLYP_DIR,
        MAX_SAMPLES,
        N_QUBITS,
        use_superpixels=True,
        compactness=SLIC_COMPACTNESS
    )
    
    # 2. Prepare graphs for compilation
    graphs_to_compile, original_data = prepare_graphs_for_compilation(
        combined_dataset,
        device=DEVICE
    )
    
    # 3. Compile graphs to quantum registers and pulses
    compiled_graphs = compile_graphs(
        graphs_to_compile,
        original_data,
        register_dim=REGISTER_DIM, 
        texture_feature=TEXTURE_FEATURE,
        global_duration_coef=GLOBAL_PULSE_DURATION_COEF
    )
    
    # 4. Visualize example (optional)
    if compiled_graphs and VISUALIZE_EXAMPLES:
        visualize_example(
            compiled_graphs,
            example_index=2
        )
        visualize_example(
            compiled_graphs,
            example_index=210
        )
        
    # 5. Execute quantum simulation
    processed_dataset = run_quantum_execution(
        compiled_graphs,
        nsteps=ODE_NSTEPS,
        nsteps_high=ODE_NSTEPS_HIGH
    )
    
    # 6. Visualize processed data (optional)
    if processed_dataset and VISUALIZE_EXAMPLES:
        visualize_processed_data(processed_dataset)

    
    # 7. Prepare for model training
    X, y = prepare_dataset(processed_dataset)
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    
    # 8. Train and evaluate model
    model, y_pred = train_qek_svm_model(
        X_train,
        X_test,
        y_train,
        y_test,
        mu=MU_HYPERPARAMETER,
        class_weight=CLASS_WEIGHTS,
        save_results=SAVE_RESULTS,
        result_dir=RESULTS_DIR
    )
    
    # 9. Run cross-validation (optional)
    run_cross_validation(
        model, 
        X, 
        y
    )
    
    return model, processed_dataset

if __name__ == "__main__":
    model, processed_dataset = main()
    print("\nPipeline execution completed.")
    print("\nConfig parameters:")
    print(f"Number of qubits: {N_QUBITS}")
    print(f"Maximum samples: {MAX_SAMPLES}")
    print(f"Register dimension: {REGISTER_DIM}")
    print(f"Device: {DEVICE}")
    print(f"ODE nsteps: {ODE_NSTEPS}")
    print(f"ODE nsteps high: {ODE_NSTEPS_HIGH}")
    print(f"Texture feature: {TEXTURE_FEATURE}")
    print(f"μ hyperparameter: {MU_HYPERPARAMETER}")
    print(f"Class weights: {CLASS_WEIGHTS}")
    print(f"Compactness: {SLIC_COMPACTNESS}")
    print(f"Global pulse duration coef: {GLOBAL_PULSE_DURATION_COEF}")