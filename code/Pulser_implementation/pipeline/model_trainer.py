import numpy as np
import os
import json
import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, classification_report
from qek.kernel import QuantumEvolutionKernel as QEK
from visualization.visualization import plot_confusion_matrix
from pipeline.config import RESULTS_DIR

def prepare_dataset(processed_dataset):
    """Prepare dataset for model training"""
    X = [data for data in processed_dataset]
    y = [data.target for data in processed_dataset]

    print("\nClass distribution in processed dataset:")
    class_counts = {}
    for data in processed_dataset:
        label = data.target
        class_counts[label] = class_counts.get(label, 0) + 1
    print(f"No polyp (0): {class_counts.get(0, 0)}")
    print(f"Polyp (1): {class_counts.get(1, 1)}")
    
    return X, y


def split_dataset(X, y, test_size=0.2, random_state=42):
    """Split dataset into training and testing sets"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        stratify=y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    print(f"Size of the training set: {len(X_train)}")
    print(f"Size of the testing set: {len(X_test)}")
    print(f"Class distribution - Training: No polyp: {y_train.count(0)}, Polyp: {y_train.count(1)}")
    print(f"Class distribution - Testing: No polyp: {y_test.count(0)}, Polyp: {y_test.count(1)}")
    
    return X_train, X_test, y_train, y_test


def train_qek_svm_model(X_train, X_test, y_train, y_test, mu=0.5, class_weight='balanced', save_results=False, result_dir="./results"):
    """Train SVM model with Quantum Evolution Kernel"""
    # Initialize kernel
    qek_kernel = QEK(mu=mu)
    
    # Create and train model
    model = SVC(
        kernel=qek_kernel, 
        random_state=42,
        class_weight=class_weight
    )
    print("\nTraining SVM model with Quantum Evolution Kernel...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test, y_pred, save_results=save_results, result_dir=result_dir)
    
    return model, y_pred


def evaluate_model(model, X_test, y_test, y_pred, save_results=False, result_dir=RESULTS_DIR):
    """Evaluate model performance with various metrics"""
    
    global RESULTS_DIR
    print("\nModel Prediction Analysis:")
    unique_predictions = np.unique(y_pred)
    print(f"Unique predicted classes: {unique_predictions}")
    print(f"Number of predictions for each class: {np.bincount(y_pred if isinstance(y_pred, np.ndarray) else np.array(y_pred, dtype=int))}")
    print(f"True class distribution: {np.bincount(y_test if isinstance(y_test, np.ndarray) else np.array(y_test, dtype=int))}")

    # Calculate metrics
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
    
    # Prepare results dictionary
    results = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "metrics": {
            "f1_score": float(f1),
            "balanced_accuracy": float(balanced_acc),
            "confusion_matrix": conf_matrix,
        },
        "predictions": {
            "unique_classes": unique_predictions.tolist(),
            "class_distribution": {
                "predictions": np.bincount(np.array(y_pred, dtype=int)).tolist(),
                "true_labels": np.bincount(np.array(y_test, dtype=int)).tolist()
            }
        }
    }
    
    # Show prediction probabilities if available
    if hasattr(model, 'predict_proba'):
        try:
            proba = model.predict_proba(X_test)
            print("\nPrediction probabilities:")
            print(f"Mean probability for class 0: {np.mean(proba[:, 0]):.4f}")
            print(f"Mean probability for class 1: {np.mean(proba[:, 1]):.4f}")
            
            results["predictions"]["probabilities"] = {
                "mean_class_0": float(np.mean(proba[:, 0])),
                "mean_class_1": float(np.mean(proba[:, 1]))
            }
        except Exception as e:
            print(f"Could not get prediction probabilities: {e}")

    print("\nEvaluation Results:")
    print(f"F1 Score: {f1}")
    print(f"Balanced Accuracy Score: {balanced_acc}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred, ['No Polyp', 'Polyp'], save_results=save_results, result_dir=result_dir)
       
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=['No Polyp', 'Polyp'], zero_division=0, output_dict=True)
    print(classification_report(y_test, y_pred, target_names=['No Polyp', 'Polyp'], zero_division=0))
    results["metrics"]["classification_report"] = report
    
    # Save results to JSON file if requested
    if save_results:
        # Create result directory if it doesn't exist
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
            
        # Create filename with timestamp
        filename = f"model_evaluation_{results['timestamp']}.json"
        filepath = os.path.join(result_dir, filename)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to: {filepath}")
    
    analyze_predictions(model, X_test, y_test, y_pred)
    
    print("\n--- End of Model Analysis ---")

def run_cross_validation(model, X, y, cv=5):
    """Run cross-validation for more robust assessment"""
    if len(X) >= 10:  # Only run if we have enough data
        print("\nRunning cross-validation to get a more robust assessment:")
        try:
            cv_scores = cross_val_score(
                model, 
                X, y, 
                cv=min(cv, len(np.unique(y))), 
                scoring='balanced_accuracy'
            )
            print(f"Cross-validation balanced accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
            return cv_scores
        except Exception as e:
            print(f"Could not run cross-validation: {e}")
    return None


def analyze_predictions(model, X_test, y_test, y_pred):
    """Analyze prediction errors in detail"""
    print("\n=== DETAILED PREDICTION ANALYSIS ===")
    
    # Get indices of different prediction outcomes
    true_pos = [i for i, (y, p) in enumerate(zip(y_test, y_pred)) if y == 1 and p == 1]
    false_neg = [i for i, (y, p) in enumerate(zip(y_test, y_pred)) if y == 1 and p == 0]
    false_pos = [i for i, (y, p) in enumerate(zip(y_test, y_pred)) if y == 0 and p == 1]
    
    print(f"Correctly detected polyps: {len(true_pos)} of {y_test.count(1)}")
    print(f"Missed polyps: {len(false_neg)} of {y_test.count(1)}")
    print(f"False alarms: {len(false_pos)} of {len(y_test) - y_test.count(1)}")
    
    # If probability estimates are available, analyze confidence
    if hasattr(model, 'predict_proba'):
        try:
            proba = model.predict_proba(X_test)
            
            # Calculate average confidence for each category
            if true_pos:
                avg_conf_tp = np.mean([proba[i][1] for i in true_pos])
                print(f"Avg. confidence for correctly detected polyps: {avg_conf_tp:.3f}")
            
            if false_neg:
                avg_conf_fn = np.mean([proba[i][0] for i in false_neg])
                print(f"Avg. confidence for missed polyps: {avg_conf_fn:.3f}")
                
            if false_pos:
                avg_conf_fp = np.mean([proba[i][1] for i in false_pos])
                print(f"Avg. confidence for false alarms: {avg_conf_fp:.3f}")
        except:
            pass
    
    print("=== END ANALYSIS ===\n")