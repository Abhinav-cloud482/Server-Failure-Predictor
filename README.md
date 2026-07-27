# Server-Failure-Predictor
An end-to-end AIOps pipeline that predicts infrastructure outages using machine learning. It simulates real-time server telemetry (CPU, memory, temperature, network, I/O), trains an XGBoost classification model to detect critical failure risks, and outputs actionable diagnostic visualizations.


# Server Failure Predictor using XGBoost

> Predict potential server failures from telemetry metrics using Machine Learning.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Model-red?style=for-the-badge&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-purple?style=for-the-badge&logo=pandas)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

## Project Overview

This project demonstrates an **end-to-end machine learning pipeline** for predicting server failures using **synthetically generated server telemetry data**.

The model learns patterns from server health metrics such as CPU usage, memory usage, disk I/O latency, network errors, and operating temperature to classify whether a server is likely to fail.

The project serves as a practical example of:

- Predictive Maintenance
- Infrastructure Monitoring
- Machine Learning Classification
- Feature Importance Analysis
- Model Evaluation and Visualization

## Features

- Synthetic server telemetry generation
- XGBoost classification model
- ROC-AUC evaluation
- Classification report
- Confusion matrix visualization
- Feature importance visualization
- Automatic handling of class imbalance
- Reproducible results using a fixed random seed

## Machine Learning Workflow

```text
Generate Synthetic Data
          │
          ▼
Preprocess Dataset
          │
          ▼
Train-Test Split
          │
          ▼
Train XGBoost Classifier
          │
          ▼
Evaluate Performance
          │
          ▼
Visualize Results
```

## Project Structure

```text
Server-Failure-Predictor/
│
├── server_failure_predictor.py
├── requirements.txt
├── README.md
└── images/                 (optional for screenshots)
```

## Dataset

The dataset is generated programmatically and simulates real-world server telemetry.

### Features

| Feature | Description |
|---------|-------------|
| CPU Usage (%) | Processor utilization |
| Memory Usage (%) | RAM utilization |
| Disk I/O Wait (ms) | Disk latency |
| Network Errors | Packet errors/drops |
| Temperature (°C) | Server operating temperature |

### Target

| Label | Meaning |
|------|---------|
| 0 | Healthy Server |
| 1 | Server Failure |

Failures are determined using a stress-based scoring system that combines:

- High CPU usage
- High memory usage
- High temperature
- High disk I/O wait
- High network error count
- Random environmental noise

## Technologies Used

- Python
- NumPy
- Pandas
- XGBoost
- Matplotlib
- Scikit-learn

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/server-failure-predictor.git
```

Move into the project directory:

```bash
cd server-failure-predictor
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

```bash
python server_failure_predictor.py
```

## Model Evaluation

The model reports the following evaluation metrics:

- ROC-AUC Score
- Precision
- Recall
- F1-Score
- Support
- Confusion Matrix

Example output:

```text
==================================================
MODEL PERFORMANCE REPORT
==================================================

ROC-AUC Score: 0.98

Classification Report

              precision    recall    f1-score

Healthy         0.99       0.99      0.99
Failure         0.95       0.94      0.95
```

> **Note:** Actual values may vary slightly because the dataset is synthetically generated.

## Visualizations

### Feature Importance

Displays which telemetry metrics contribute most to predicting server failures.

Typical important features include:

- Temperature
- CPU Usage
- Memory Usage
- Disk I/O Wait

### Confusion Matrix

Displays:

- True Positives
- True Negatives
- False Positives
- False Negatives

to help evaluate classification performance.

## Workflow

1. Generate synthetic server telemetry.
2. Build a structured dataset.
3. Split the data into training and testing sets.
4. Train an XGBoost classifier.
5. Predict server failures.
6. Evaluate model performance.
7. Visualize feature importance.
8. Display the confusion matrix.

## Requirements

```text
numpy==2.2.6
pandas==2.3.1
xgboost==3.0.4
matplotlib==3.10.5
scikit-learn==1.7.1
```

## Future Improvements

- Train on real production telemetry
- Hyperparameter tuning
- Cross-validation
- SHAP explainability
- Model persistence using Pickle or Joblib
- Real-time monitoring dashboard
- Stream telemetry using Kafka
- REST API with FastAPI
- Docker deployment
- Kubernetes integration

## Applications

- Data Centers
- Cloud Infrastructure
- Predictive Maintenance
- Server Health Monitoring
- DevOps Automation
- Infrastructure Reliability Engineering

## Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push the branch.
5. Open a Pull Request.

## License

This project is open source and available under the **MIT License**.

## Author

**Your Name**

If you found this project useful, consider giving it a star on GitHub.
