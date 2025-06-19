import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os


parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str, required=True)
args = parser.parse_args()

# Handle relative path
DATA_PATH = os.path.join(os.path.dirname(__file__), args.data_file)
df = pd.read_csv(DATA_PATH)

# Features
feature_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
target_column = 'label'

X = df[feature_columns]
y = df[target_column]

print(f"Dataset shape: {df.shape}")
print(f"Features: {feature_columns}")
print(f"Target: {target_column}")
print(f"Target classes: {y.unique()}")

# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Set name experiment
mlflow.set_experiment("Soil_Classification_CI")

print("Starting MLflow Tracking...")

mlflow.sklearn.autolog(disable=True)

def log_metrics(y_true, y_pred, prefix="val"):
    metrics = {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_precision": precision_score(y_true, y_pred, average='weighted'),
        f"{prefix}_recall": recall_score(y_true, y_pred, average='weighted'),
        f"{prefix}_f1": f1_score(y_true, y_pred, average='weighted')
    }
    mlflow.log_metrics(metrics)
    print(f"Logged metrics for {prefix}: {metrics}")
    return metrics

# Baseline Model - Random Forest
with mlflow.start_run(run_name="RF_Baseline"):
    print("\nTraining Random Forest Baseline")
    rf_baseline = RandomForestClassifier(random_state=42)
    rf_baseline.fit(X_train, y_train)
    
    # Manual logging
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("random_state", 42)
    
    y_pred_rf = rf_baseline.predict(X_val)
    rf_metrics = log_metrics(y_val, y_pred_rf, "val")
    
    mlflow.sklearn.log_model(rf_baseline, "model")

# Gradient Boosting Model
with mlflow.start_run(run_name="GBM_Baseline"):
    print("\nTraining Gradient Boosting Machine")
    gbm = GradientBoostingClassifier(random_state=42)
    gbm.fit(X_train, y_train)
    
    # Manual logging
    mlflow.log_param("model_type", "GradientBoosting")
    mlflow.log_param("random_state", 42)
    
    y_pred_gbm = gbm.predict(X_val)
    gbm_metrics = log_metrics(y_val, y_pred_gbm, "val")
    
    mlflow.sklearn.log_model(gbm, "model")

# Compare models
models = {
    'Random Forest': rf_metrics['val_accuracy'],
    'Gradient Boosting': gbm_metrics['val_accuracy']
}

best_model_name = max(models, key=models.get)
best_accuracy = models[best_model_name]

print("\n" + "="*50)
print(f"Best Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
print("="*50)

# Final training with best model
if best_model_name == 'Random Forest':
    best_model = rf_baseline
else:
    best_model = gbm

with mlflow.start_run(run_name=f"Final_{best_model_name}"):
    print(f"\nTraining Final Model: {best_model_name}")
    
    # Combine train and validation sets
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])
    
    best_model.fit(X_train_full, y_train_full)
    
    mlflow.log_param("model_type", best_model_name)
    mlflow.log_param("full_training", True)
    
    y_pred_test = best_model.predict(X_test)
    test_metrics = log_metrics(y_test, y_pred_test, "test")
    
    mlflow.sklearn.log_model(best_model, "model")
    
    print("\nMLflow execution completed successfully!")