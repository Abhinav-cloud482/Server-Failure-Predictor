import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

# Set random seed for reproducibility
np.random.seed(42)

def generate_server_data(num_samples=5000):
    """
    Generates synthetic server telemetry metrics.
    A server is flagged as 'failed' (1) if combined stress indicators cross a threshold.
    """
    print("1. Generating synthetic server telemetry data...")
    
    # Base telemetry features
    cpu_usage = np.random.uniform(10, 95, num_samples)          # Percentage (10% - 95%)
    memory_usage = np.random.uniform(20, 98, num_samples)       # Percentage (20% - 98%)
    disk_io_wait = np.random.exponential(scale=5.0, size=num_samples) # Latency in ms
    network_errors = np.random.poisson(lam=2.0, size=num_samples) # Count of dropped/error packets
    temperature_c = np.random.normal(loc=55, scale=12, size=num_samples) # Celsius
    
    # Create pandas DataFrame
    df = pd.DataFrame({
        'cpu_usage_pct': cpu_usage,
        'memory_usage_pct': memory_usage,
        'disk_io_wait_ms': disk_io_wait,
        'network_errors': network_errors,
        'temperature_c': temperature_c
    })
    
    # Define a failure condition logic based on combined stress
    # High temp + high CPU, or runaway disk wait + memory pressure trigger failures
    stress_score = (
        (df['cpu_usage_pct'] > 85) * 2.5 +
        (df['memory_usage_pct'] > 90) * 2.0 +
        (df['temperature_c'] > 75) * 3.0 +
        (df['disk_io_wait_ms'] > 15) * 1.5 +
        (df['network_errors'] > 5) * 1.0 +
        np.random.normal(0, 1, num_samples) # Random noise/unobserved factors
    )
    
    # Binary target: 1 = Failure, 0 = Normal
    df['failure_label'] = (stress_score > 6.5).astype(int)
    
    return df


def train_and_evaluate(df):
    """
    Splits data, trains XGBoost classifier, and prints evaluation metrics.
    """
    print("2. Preprocessing data and training XGBoost model...")
    
    X = df.drop(columns=['failure_label'])
    y = df['failure_label']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Compute class ratio for handling potential imbalance
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    
    # Initialize XGBoost Classifier
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Evaluation outputs
    print("\n" + "="*50)
    print("MODEL PERFORMANCE REPORT")
    print("="*50)
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_probs):.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Healthy', 'Failure']))
    
    return model, X_test, y_test, y_pred


def visualize_results(model, df, X_test, y_test, y_pred):
    """
    Generates plots for feature importances and model diagnostic results.
    """
    print("3. Generating diagnostic plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Feature Importances
    importances = model.feature_importances_
    features = X_test.columns
    indices = np.argsort(importances)
    
    axes[0].barh(range(len(indices)), importances[indices], color='skyblue', align='center')
    axes[0].set_yticks(range(len(indices)))
    axes[0].set_yticklabels([features[i] for i in indices])
    axes[0].set_xlabel('Relative Importance')
    axes[0].set_title('XGBoost Feature Importance for Server Failure')
    
    # Plot 2: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Healthy', 'Failure'])
    disp.plot(ax=axes[1], cmap='Blues', values_format='d')
    axes[1].set_title('Confusion Matrix')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Prerequisites verification note
    # Make sure you have installed: pip install pandas xgboost matplotlib scikit-learn
    
    # Run full end-to-end pipeline
    telemetry_df = generate_server_data(num_samples=5000)
    
    print("\nSample telemetry data:")
    print(telemetry_df.head())
    print(f"\nClass Distribution:\n{telemetry_df['failure_label'].value_counts(normalize=True)}")
    
    model, X_test, y_test, y_pred = train_and_evaluate(telemetry_df)
    visualize_results(model, telemetry_df, X_test, y_test, y_pred)
