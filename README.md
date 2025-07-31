# Final Project: End-to-End Diabetes Prediction System with MLOps
## Project Overview
This project delivers a comprehensive and automated system for predicting diabetes based on various health indicators. It encompasses the entire Machine Learning Operations (MLOps) lifecycle, from initial data experimentation and robust model building to continuous integration (CI) and advanced monitoring and logging. The primary goal is to provide an accurate and reliable predictive tool while ensuring the model's efficiency, scalability, and maintainability in a production-like environment.

Key Challenges Addressed:

Developing an accurate and robust machine learning model for early diabetes detection.

Automating the entire ML pipeline, including data preprocessing, model training, and deployment.

Implementing real-time monitoring of model performance and underlying infrastructure.

Ensuring automatic retraining capabilities to adapt to new data and maintain model relevance.

Key Features & MLOps Pipeline (Advanced Tier Implementation)
This project demonstrates expertise across all critical MLOps stages, achieving the highest "Advanced" tier in each criterion:

Automated Data Preprocessing & Experimentation:

In-depth Data Exploration: Performed comprehensive exploratory data analysis (EDA) and manual data preprocessing to understand data characteristics and define optimal transformation steps.

Automated Preprocessing Script: Developed a Python script (automate_preprocessing.py) to automatically clean, transform, and prepare raw health indicator data, ensuring it's always ready for model training.

CI for Preprocessing (GitHub Actions): Implemented a GitHub Actions workflow that automatically triggers the data preprocessing script upon specified events (e.g., new data commits), ensuring a continuously updated and validated dataset.

Robust Machine Learning Model Building:

Model Training & Hyperparameter Tuning: Trained a high-performing Machine Learning classification model (e.g., Scikit-learn based) on the prepared dataset, utilizing advanced hyperparameter tuning techniques for optimal performance.

Online Experiment Tracking (MLflow & DagsHub): Leveraged MLflow Tracking UI for comprehensive experiment management, logging runs, parameters, and metrics. All experiments and model artifacts are stored and accessible online via DagsHub, facilitating collaboration and reproducibility.

Manual & Enhanced Logging: Implemented detailed manual logging within MLflow, capturing not only standard metrics (accuracy, precision, recall, F1-score) but also at least two additional custom metrics crucial for in-depth model performance analysis.

Continuous Integration (CI) Workflow for Model Retraining:

MLflow Project Integration: Structured the model training component as an MLflow Project, encapsulating the model code and its environment dependencies.

Automated Model Retraining (GitHub Actions): Established a robust CI workflow using GitHub Actions that automatically triggers model retraining whenever new processed data is available or code changes occur, ensuring the model remains up-to-date and performs optimally.

Artifact Management & Dockerization (Docker Hub): Model artifacts are automatically stored in a designated repository (e.g., GitHub or Google Drive). Furthermore, the project integrates mlflow build-docker to automatically create and push Docker images of the trained model to Docker Hub, streamlining deployment to various environments.

Real-time Monitoring and Logging System:

Model Serving: The trained and versioned model is served, making it accessible for real-time predictions.

Performance Monitoring (Prometheus): Implemented Prometheus to collect and store crucial metrics related to the system's performance, including prediction latency, request volume, and key model performance indicators over time. A minimum of 10 distinct metrics are tracked.

Visualization & Alerting (Grafana): Developed comprehensive, interactive dashboards in Grafana to visualize the collected metrics in real-time. These dashboards provide immediate insights into the system's health and model performance.

Advanced Alerting System: Configured three distinct alerting rules within Grafana, ensuring automated notifications are dispatched to relevant stakeholders when predefined thresholds are breached (e.g., sudden drop in model accuracy, high prediction latency), enabling proactive issue resolution.

Repository Structure
This repository is organized into distinct directories, each corresponding to a major phase of the MLOps pipeline, ensuring clarity, modularity, and ease of navigation:
```
.
├── README.md                                 # This file
├── .github/workflows/                        # GitHub Actions workflows for CI/CD
│   ├── preprocess_data.yml                   # Workflow for automated data preprocessing
│   └── retrain_model.yml                     # Workflow for automated model retraining
├── data/
│   ├── raw/                                  # Contains the raw, unprocessed dataset
│   └── processed/                            # Stores the preprocessed dataset (output of preprocessing)
├── notebooks/
│   └── experiment_diabetes_prediction.ipynb  # Jupyter Notebook for initial data exploration and manual experimentation
├── src/
│   ├── preprocessing/
│   │   └── automate_preprocessing.py         # Python script for automated data preprocessing
│   ├── model_training/
│   │   ├── modelling.py                      # Script for basic model training with MLflow autologging
│   │   └── modelling_tuning.py               # Script for advanced model training with hyperparameter tuning and manual MLflow logging
│   └── serving/
│       └── inference.py                      # Script for serving the trained model and making predictions
├── mlruns/                                   # Local MLflow tracking server directory (if used)
├── monitoring/
│   ├── prometheus.yml                        # Prometheus configuration file
│   ├── prometheus_exporter.py                # Custom exporter for application-specific metrics (if applicable)
│   └── grafana_dashboards/                   # Directory for Grafana dashboard JSON configurations
├── docs/
│   ├── screenshots/                          # Contains screenshots of dashboards, artifacts, monitoring, and alerting
│   │   ├── mlflow_dashboard.jpg
│   │   ├── mlflow_artifacts.jpg
│   │   ├── prometheus_monitoring/
│   │   └── grafana_monitoring/
│   │   └── grafana_alerting/
│   └── serving_proof/                        # Documentation/screenshots proving model serving functionality
├── MLProject                                 # MLflow Project file for defining the model's entry points and dependencies
├── conda.yaml                                # Conda environment definition for MLflow Project
├── requirements.txt                          # Python package dependencies for the entire project
├── DAGsHub.txt                               # Text file containing the link to the DagsHub repository for online MLflow tracking
├── DOCKER_HUB_LINK.txt                       # Text file containing the link to the Docker Hub repository for model images
└── .gitignore                                # Specifies intentionally untracked files to ignore
```
