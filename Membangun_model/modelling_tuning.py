import os
import shutil
import sys
import mlflow
import dagshub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split, ParameterGrid
from modelling import train_and_log_model # Import function from modelling.py
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

# --- Data Loading and Splitting ---
try:
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "diabetes_preprocessing.csv")

    # Use the full path to load the data
    df = pd.read_csv(data_path)

    # Convert all integer columns to float64 to handle possible missing values
    int_cols = df.select_dtypes(include="int").columns
    df[int_cols] = df[int_cols].astype("float64")

    # Set the target variable and features
    X = df.drop('Diabetes_binary', axis=1)
    y = df['Diabetes_binary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Data loaded and split successfully.")
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
except FileNotFoundError:
    print("Error: Dataset not found.")
    print("Please ensure the dataset is in the same directory or provide the correct path.")
    sys.exit(1)  # Stop the script with a non-zero exit code
# --- End Data Loading ---


# --- MLflow Configuration ---
# Initialize Dagshub MLflow client
dagshub.init("my-first-repo", "Acrola", mlflow=True)
mlflow.set_experiment("Diabetes_Prediction_Hyperparameter_Tuning")

# Define the hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5]
}
grid_search_combinations = list(ParameterGrid(param_grid)) # Generates all combinations

print(f"Total tuning combinations: {len(grid_search_combinations)}")

# --- Start Parent MLflow Run for Hyperparameter Tuning ---
with mlflow.start_run(run_name="ParameterGrid_Hyperparameter_Tuning_Parent_Run") as parent_run:
    print(f"Parent Run ID: {parent_run.info.run_id}")

    # Log global parameters for the tuning process
    mlflow.log_param("tuning_strategy", "Manual Grid Search")
    mlflow.log_param("param_grid_search_space", str(param_grid))
    mlflow.log_param("num_combinations", len(grid_search_combinations))
    mlflow.set_tag("mlflow.note.content", "Orchestrating Random Forest hyperparameter tuning.")

    best_f1_score = -1.0 # Initialize with a low score
    best_overall_params = {}
    best_overall_model = None
    best_run_id = None
    best_overall_accuracy = -1.0
    best_overall_roc_auc = -1.0

    # Iterate through each hyperparameter combination
    for i, params_combo in enumerate(grid_search_combinations):
        print(f"\n--- Running Trial {i+1}/{len(grid_search_combinations)} ---")
        # Start a nested (child) run for each individual training trial
        with mlflow.start_run(nested=True, run_name=f"Trial_{i+1}_RF") as child_run:
            print(f"Child Run ID: {child_run.info.run_id}")
            mlflow.set_tag("mlflow.note.content", f"Trial {i+1} for RF tuning.")

            # Call the function from modelling.py to train and log the model
            trial_results = train_and_log_model(X_train, y_train, X_test, y_test, params_combo)

            # Manually log additional metrics
            mlflow.log_metric("training_duration_seconds", trial_results["training_duration"])
            mlflow.log_metric("test_specificity", trial_results["test_specificity"])
            mlflow.log_metric("test_accuracy", trial_results["test_accuracy"])
            mlflow.log_metric("test_f1_score", trial_results["test_f1_score"])
            mlflow.log_metric("test_roc_auc", trial_results["test_roc_auc"])

            # --- Compute predictions for artifact logging ---
            model = trial_results["model"]
            y_pred = model.predict(X_test)

            # --- Log confusion matrix as artifact ---
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            classes = np.unique(np.concatenate([y_test, y_pred]))
            ax.set_xticks(np.arange(len(classes)))
            ax.set_yticks(np.arange(len(classes)))
            ax.set_xticklabels(classes)
            ax.set_yticklabels(classes)
            for i in range(len(classes)):
                for j in range(len(classes)):
                    ax.text(j, i, cm[i, j], ha="center", va="center", color="red")
            fig.tight_layout()
            plt.savefig("confusion_matrix.png")
            plt.close(fig)
            mlflow.log_artifact("confusion_matrix.png")

            # --- Log feature importances as artifact ---
            if hasattr(model, "feature_importances_"):
                feature_importances = model.feature_importances_
                feature_names = X_test.columns if hasattr(X_test, "columns") else [f"feature_{i}" for i in range(len(feature_importances))]
                fi_df = pd.DataFrame({"feature": feature_names, "importance": feature_importances})
                fi_df.to_csv("feature_importance.csv", index=False)
                mlflow.log_artifact("feature_importance.csv")

            # Track best model
            if trial_results["test_f1_score"] > best_f1_score:
                best_f1_score = trial_results["test_f1_score"]
                best_overall_params = params_combo
                best_overall_model = trial_results["model"]
                best_run_id = child_run.info.run_id
                best_overall_accuracy = trial_results["test_accuracy"]
                best_overall_roc_auc = trial_results["test_roc_auc"]
                best_training_duration = trial_results["training_duration"]
                best_specificity = trial_results["test_specificity"]

    # --- Log Best Model Details to the Parent Run ---
    print("\n--- Hyperparameter Tuning Summary ---")
    print(f"Best F1 Score: {best_f1_score:.4f}")
    print(f"Best Parameters: {best_overall_params}")
    print(f"Best Model Run ID: {best_run_id}")

    mlflow.log_param("best_model_run_id", best_run_id)
    mlflow.log_params(best_overall_params) # Log the winning parameters to the parent run
    mlflow.log_metric("best_overall_f1_score", best_f1_score)
    mlflow.log_metric("best_overall_accuracy_score", best_overall_accuracy)
    mlflow.log_metric("best_overall_roc_auc", best_overall_roc_auc)
    mlflow.log_metric("best_training_duration_seconds", best_training_duration)
    mlflow.log_metric("best_test_specificity", best_specificity)
    

    # --- Fetch and log all metrics and params from the best child run ---
    if best_run_id:
        client = MlflowClient()
        # Fetch all params and metrics from the best child run
        best_run = client.get_run(best_run_id)
        # Log all params
        for key, value in best_run.data.params.items():
            mlflow.log_param(f"best_{key}", value)
        # Log all metrics
        for key, value in best_run.data.metrics.items():
            mlflow.log_metric(f"best_{key}", value)

    if best_overall_model:
        # Log the best model to MLflow (if you want to log it as a separate run or artifact)
        mlflow.sklearn.log_model(best_overall_model, "best_model")
        print("Best model logged to MLflow.")

        # Save the best model locally for LFS or deployment
        local_model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model"))
        if os.path.exists(local_model_dir):
            shutil.rmtree(local_model_dir)
        mlflow.sklearn.save_model(best_overall_model, local_model_dir)
        print(f"Final best model saved locally to {local_model_dir}")
    else:
        print("No best model found (tuning issue or no trials completed).")

print("\nHyperparameter tuning process complete. Check MLflow UI for detailed results.")