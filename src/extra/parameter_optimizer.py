import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
from datetime import datetime
from itertools import product

# Import pipeline components
from pipeline.config import *
from pipeline.data_loader import load_datasets, prepare_graphs_for_compilation
from pipeline.graph_compiler import compile_graphs
from pipeline.quantum_executor import run_quantum_execution
from pipeline.model_trainer import prepare_dataset, split_dataset, train_qek_svm_model
from sklearn.metrics import balanced_accuracy_score, f1_score


class ParameterOptimizer:
    def __init__(self, results_dir="results"):
        """Initialize parameter optimizer"""
        self.results_dir = Path(__file__).parent / results_dir
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.results = []
        self.best_params = None
        self.best_score = 0
        
        # Set up logging
        self.log_path = self.results_dir / f"optimization_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
    def log(self, message):
        """Log message to console and file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        print(formatted_message)
        with open(self.log_path, "a") as f:
            f.write(formatted_message + "\n")
            
    def define_parameter_grid(self, quick_mode=False):
        """Define parameter grid for testing"""
        if quick_mode:
            # Smaller grid for quick testing
            return {
                "N_QUBITS": [15, 20],
                "REGISTER_DIM": [20, 30],
                "TEXTURE_FEATURE": ["energy", "pca"],
                "MU_HYPERPARAMETER": [0.8, 1.2],
                "SLIC_COMPACTNESS": [10],
                "CLASS_WEIGHTS": [
                    "balanced", 
                    {0: 1.0, 1: 2.0}
                ],
                "MAX_SAMPLES": [50]  # Small sample for quick runs
            }
        else:
            # Full parameter grid
            return {
                "N_QUBITS": [10, 15, 20],
                "REGISTER_DIM": [20, 30, 40],
                "TEXTURE_FEATURE": ["pca", "energy", "homogeneity", "contrast"],
                "MU_HYPERPARAMETER": [0.5, 0.8, 1.0, 1.2, 1.5],
                "SLIC_COMPACTNESS": [5, 10, 15],
                "CLASS_WEIGHTS": [
                    "balanced", 
                    {0: 1.0, 1: 1.5},
                    {0: 1.0, 1: 2.0},
                    {0: 1.0, 1: 3.0}
                ],
                "MAX_SAMPLES": [200]  # Fixed to control runtime
            }
    
    def run_pipeline(self, params):
        """Run the pipeline with specific parameters"""
        # Set global variables from params
        global N_QUBITS, REGISTER_DIM, TEXTURE_FEATURE, MU_HYPERPARAMETER
        global SLIC_COMPACTNESS, CLASS_WEIGHTS, MAX_SAMPLES
        
        N_QUBITS = params["N_QUBITS"]
        REGISTER_DIM = params["REGISTER_DIM"]
        TEXTURE_FEATURE = params["TEXTURE_FEATURE"]
        MU_HYPERPARAMETER = params["MU_HYPERPARAMETER"]
        SLIC_COMPACTNESS = params["SLIC_COMPACTNESS"]
        CLASS_WEIGHTS = params["CLASS_WEIGHTS"]
        MAX_SAMPLES = params["MAX_SAMPLES"]
        
        # Load and prepare datasets
        combined_dataset = load_datasets(
            NO_POLYP_DIR, 
            POLYP_DIR,
            MAX_SAMPLES,
            N_QUBITS,
            use_superpixels=True,
            compactness=SLIC_COMPACTNESS
        )
        
        # Prepare graphs for compilation
        graphs_to_compile, original_data = prepare_graphs_for_compilation(
            combined_dataset,
            device=DEVICE
        )
        
        # Compile graphs to quantum registers and pulses
        compiled_graphs = compile_graphs(
            graphs_to_compile,
            original_data,
            register_dim=REGISTER_DIM, 
            texture_feature=TEXTURE_FEATURE
        )
        
        # Execute quantum simulation
        processed_dataset = run_quantum_execution(
            compiled_graphs,
            nsteps=ODE_NSTEPS,
            nsteps_high=ODE_NSTEPS_HIGH
        )
        
        # Prepare for model training
        X, y = prepare_dataset(processed_dataset)
        X_train, X_test, y_train, y_test = split_dataset(X, y)
        
        # Train and evaluate model
        model, y_pred = train_qek_svm_model(
            X_train, 
            X_test, 
            y_train, 
            y_test, 
            mu=MU_HYPERPARAMETER,
            class_weight=CLASS_WEIGHTS
        )
        
        # Calculate metrics
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        return {
            "balanced_accuracy": bal_acc,
            "f1_score": f1,
            "model": model,
            "y_pred": y_pred,
            "y_test": y_test,
            "compiled_graphs_count": len(compiled_graphs),
            "processed_dataset_count": len(processed_dataset)
        }
    
    def optimize(self, quick_mode=False, max_combinations=None):
        """Run optimization process"""
        param_grid = self.define_parameter_grid(quick_mode)
        
        # Generate all parameter combinations
        keys = param_grid.keys()
        all_combinations = list(product(*(param_grid[key] for key in keys)))
        param_dicts = [dict(zip(keys, combo)) for combo in all_combinations]
        
        if max_combinations and len(param_dicts) > max_combinations:
            self.log(f"Limiting to {max_combinations} random combinations out of {len(param_dicts)} total")
            import random
            random.shuffle(param_dicts)
            param_dicts = param_dicts[:max_combinations]
        
        self.log(f"Starting parameter optimization with {len(param_dicts)} combinations")
        self.log(f"Parameter space: {param_grid}")
        
        # Run each combination
        for i, params in enumerate(param_dicts):
            self.log(f"\n--- Run {i+1}/{len(param_dicts)} ---")
            self.log(f"Testing parameters: {params}")
            
            start_time = time.time()
            try:
                results = self.run_pipeline(params)
                runtime = time.time() - start_time
                
                # Store results
                run_results = {
                    "params": params,
                    "balanced_accuracy": results["balanced_accuracy"],
                    "f1_score": results["f1_score"],
                    "runtime_seconds": runtime,
                    "compiled_graphs": results["compiled_graphs_count"],
                    "processed_dataset": results["processed_dataset_count"],
                    "success": True
                }
                
                self.log(f"Results: Balanced Accuracy = {results['balanced_accuracy']:.4f}, F1 = {results['f1_score']:.4f}")
                self.log(f"Runtime: {runtime:.1f} seconds")
                
                # Check if this is the best run so far
                if results["balanced_accuracy"] > self.best_score:
                    self.best_score = results["balanced_accuracy"]
                    self.best_params = params
                    self.log(f"New best parameters found!")
                
            except Exception as e:
                import traceback
                runtime = time.time() - start_time
                self.log(f"Error running with parameters {params}: {str(e)}")
                self.log(traceback.format_exc())
                
                run_results = {
                    "params": params,
                    "balanced_accuracy": None,
                    "f1_score": None,
                    "runtime_seconds": runtime,
                    "error": str(e),
                    "success": False
                }
            
            self.results.append(run_results)
            self.save_results()
        
        self.analyze_results()
        return self.best_params
    
    def save_results(self):
        """Save current results to file"""
        results_file = self.results_dir / "optimization_results.json"
        
        # Convert any non-serializable objects to strings
        serializable_results = []
        for result in self.results:
            serializable = {}
            for k, v in result.items():
                if isinstance(v, dict) and any(isinstance(x, np.integer) or 
                                              isinstance(x, np.floating) for x in v.values()):
                    serializable[k] = {key: float(val) if isinstance(val, (np.integer, np.floating)) else val 
                                     for key, val in v.items()}
                elif isinstance(v, (np.integer, np.floating)):
                    serializable[k] = float(v)
                else:
                    serializable[k] = v
            serializable_results.append(serializable)
        
        with open(results_file, "w") as f:
            json.dump({
                "results": serializable_results,
                "best_params": self.best_params,
                "best_score": float(self.best_score) if self.best_score else None
            }, f, indent=2)
    
    def analyze_results(self):
        """Analyze and visualize results"""
        # Filter successful runs
        successful_runs = [r for r in self.results if r["success"] and r["balanced_accuracy"] is not None]
        
        if not successful_runs:
            self.log("No successful runs to analyze")
            return
        
        # Create DataFrame for analysis
        df = pd.DataFrame(successful_runs)
        
        # Expand params dictionary into columns
        param_df = pd.DataFrame([r["params"] for r in successful_runs])
        analysis_df = pd.concat([param_df, df[["balanced_accuracy", "f1_score", "runtime_seconds"]]], axis=1)
        
        # Save to CSV
        analysis_df.to_csv(self.results_dir / "parameter_analysis.csv", index=False)
        
        # Create visualization of parameter effects
        self.create_parameter_impact_plots(analysis_df)
        
        # Print top 5 configurations
        self.log("\n=== TOP 5 PARAMETER CONFIGURATIONS ===")
        top5 = analysis_df.sort_values("balanced_accuracy", ascending=False).head(5)
        for i, (_, row) in enumerate(top5.iterrows()):
            self.log(f"\nRank {i+1}: Balanced Accuracy = {row['balanced_accuracy']:.4f}, F1 = {row['f1_score']:.4f}")
            self.log(f"Parameters:")
            for param, value in row.items():
                if param not in ["balanced_accuracy", "f1_score", "runtime_seconds"]:
                    self.log(f"  {param}: {value}")
        
        # Summary of best configuration
        self.log("\n=== OPTIMAL CONFIGURATION ===")
        best_idx = analysis_df["balanced_accuracy"].idxmax()
        best_row = analysis_df.iloc[best_idx]
        self.log(f"Balanced Accuracy: {best_row['balanced_accuracy']:.4f}")
        self.log(f"F1 Score: {best_row['f1_score']:.4f}")
        self.log("Parameters:")
        for param, value in best_row.items():
            if param not in ["balanced_accuracy", "f1_score", "runtime_seconds"]:
                self.log(f"  {param}: {value}")
    
    def create_parameter_impact_plots(self, df):
        """Create visualization of parameter effects"""
        plt.figure(figsize=(20, 16))
        
        # Get all parameter names
        param_names = [col for col in df.columns if col not in ["balanced_accuracy", "f1_score", "runtime_seconds"]]
        
        for i, param in enumerate(param_names):
            plt.subplot(3, 3, i+1)
            
            # Skip if parameter has only one value
            if len(df[param].unique()) <= 1:
                plt.text(0.5, 0.5, f"{param}: Only one value tested", 
                         ha='center', va='center')
                plt.axis('off')
                continue
                
            # For categorical/discrete parameters
            if df[param].dtype == object or len(df[param].unique()) < 10:
                # Convert to string for consistent categorical plotting
                param_values = df[param].astype(str)
                sns.boxplot(x=param_values, y="balanced_accuracy", data=df)
                plt.title(f"Impact of {param}")
                plt.xticks(rotation=45)
                
            # For numerical parameters
            else:
                sns.scatterplot(x=param, y="balanced_accuracy", data=df)
                plt.title(f"Impact of {param}")
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "parameter_impacts.png")
        
        # Create heatmap of interactions between top parameters
        if len(param_names) >= 2:
            top_params = param_names[:2]
            plt.figure(figsize=(12, 10))
            pivot_table = df.pivot_table(
                values="balanced_accuracy", 
                index=top_params[0], 
                columns=top_params[1],
                aggfunc="mean"
            )
            sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="viridis")
            plt.title(f"Interaction between {top_params[0]} and {top_params[1]}")
            plt.tight_layout()
            plt.savefig(self.results_dir / "parameter_interactions.png")


if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Optimize quantum polyp detection parameters")
    parser.add_argument("--quick", action="store_true", help="Run quick test with reduced parameter space")
    parser.add_argument("--max", type=int, help="Maximum number of parameter combinations to test")
    args = parser.parse_args()
    
    # Initialize and run optimizer
    optimizer = ParameterOptimizer()
    best_params = optimizer.optimize(quick_mode=args.quick, max_combinations=args.max)
    
    # Print final results
    print("\n=== OPTIMIZATION COMPLETED ===")
    print(f"Best parameters found: {best_params}")
    print(f"Best balanced accuracy: {optimizer.best_score:.4f}")