# Final Project: End-to-End Diabetes Prediction System with MLOps
## Project Overview
This project presents an automated and comprehensive system for predicting diabetes based on various health indicators. It covers the entire Machine Learning Operations (MLOps) lifecycle, from initial data experimentation and robust model building to continuous integration (CI) and advanced monitoring and logging. The primary goal is to provide an accurate and reliable predictive tool while ensuring the model's efficiency, scalability, and maintainability in a production-ready environment.

Key Challenges Addressed:

- Developing an accurate and robust machine learning model for early diabetes detection.

- Automating the entire ML pipeline, including data preprocessing, model training, and deployment.

- Implementing real-time monitoring of model performance and underlying infrastructure.

- Ensuring automatic retraining capabilities to adapt to new data and maintain model relevance.

Project Structure and MLOps Pipeline
This end-to-end diabetes prediction system is designed with a modular approach. This repository serves as the main hub, integrating various stages of the MLOps pipeline, with some core components managed as sub-directories and others as external components accessed via links. This structure enhances maintainability, scalability, and allows for independent development and deployment of components, while still forming a cohesive system.

The complete MLOps pipeline is distributed as follows:

1. Data Experimentation & Preprocessing Automation
This stage focuses on initial data exploration and preprocessing automation. This component resides in a separate repository.

Key Responsibilities:

- In-depth Exploratory Data Analysis (EDA) and feature engineering.

- Manual and automated data preprocessing routines (automate_preprocessing.py).

- Ensuring data quality and readiness for model training.

- Continuous Integration (CI) using GitHub Actions to automatically process data and generate updated datasets upon trigger.

Repository Name: Eksperimen_SML_Kenny

Access: The link to this repository is provided in the Eksperimen_SML_Nama-siswa.txt file within this main repository.

2. Machine Learning Model Building
This directory is dedicated to core machine learning tasks, including model development, hyperparameter tuning, and comprehensive experiment tracking.

Key Responsibilities:

- Training and evaluating various ML classification models for diabetes prediction.

- Advanced hyperparameter tuning for optimal model performance.

- Online experiment tracking and artifact management using MLflow, integrated with DagsHub, for full experiment reproducibility.

- Detailed manual logging of comprehensive model metrics beyond basic autologging.

Location: The Membangun_model sub-directory within this repository.

3. CI Workflow for Model Retraining
This stage is crucial for automating the model's lifecycle, focusing on continuous integration for retraining and deployment readiness. This component resides in a separate repository.

Key Responsibilities:

- Defining the model training and evaluation process as an MLflow Project.

- Automated model retraining via GitHub Actions, triggered by new data or code changes.

- Centralized storage of versioned model artifacts.

- Automated Docker image creation and pushing to Docker Hub for seamless deployment, leveraging mlflow build-docker.

Repository Name: Workflow-CI

Access: The link to this repository is provided in the Workflow-CI.txt file within this main repository.

4. Monitoring and Logging System
This component ensures the long-term health and performance of the deployed diabetes prediction model by providing real-time monitoring and alerting capabilities.

Key Responsibilities:

- Serving the trained machine learning model for inference.

- Real-time system and model performance monitoring using Prometheus, tracking a minimum of 10 different metrics (e.g., prediction latency, data drift, model drift, request rates).

- Visualizing all collected metrics through interactive dashboards in Grafana.

- Implementing advanced alerting rules in Grafana to notify stakeholders of anomalies or performance degradation, enabling proactive intervention.

Location: The Monitoring dan Logging sub-directory within this repository.

Contents of This Repository:
This repository serves as the central entry point and documentation hub for the entire project. Its primary contents are:
```
.
├── README.md                                 # This file: Project overview, aims, and links to sub-repositories.
├── Eksperimen_SML_Nama-siswa.txt             # Text file containing the URL to the Data Experimentation & Preprocessing Repository.
├── Membangun_model/                          # Directory containing code for ML model building, tuning, and experiment tracking.
│   ├── modelling.py                          # Script for basic model training with MLflow autologging.
│   ├── modelling_tuning.py                   # Script for advanced model training with hyperparameter tuning and manual MLflow logging.
│   ├── namadataset_preprocessing/            # Directory/file for the preprocessed dataset (output from preprocessing).
│   ├── screenshoot_dashboard.jpg             # Screenshot of the MLflow dashboard.
│   ├── screenshoot_artifak.jpg               # Screenshot of MLflow artifacts.
│   ├── requirements.txt                      # Python package dependencies for this module.
│   └── DagsHub.txt                           # Text file containing the link to the DagsHub repository for online MLflow tracking.
├── Workflow-CI.txt                           # Text file containing the URL to the CI Workflow for Model Retraining Repository.
├── Monitoring dan Logging/                   # Directory containing components for model serving, monitoring, and alerting.
│   ├── 1.bukti_serving                       # Proof of model serving (e.g., screenshots or logs).
│   ├── 2.prometheus.yml                      # Prometheus configuration file.
│   ├── 3.prometheus_exporter.py              # Custom exporter for application-specific metrics (if applicable).
│   ├── 4.bukti monitoring Prometheus/        # Directory containing Prometheus monitoring screenshots.
│   │   ├── 1.monitoring_<metrics>
│   │   ├── 2.monitoring_<metrics>
│   │   └── dst (adjust based on points achieved)
│   ├── 5.bukti monitoring Grafana/           # Directory containing Grafana monitoring screenshots.
│   │   ├── 1.monitoring_<metrics>
│   │   ├── 2.monitoring_<metrics>
│   │   └── dst (adjust based on points achieved)
│   ├── 6.bukti alerting Grafana/             # Directory containing Grafana alerting proof screenshots.
│   │   ├── 1.rules_<metrics>
│   │   ├── 2.notifikasi_<metrics>
│   │   ├── 3.rules_<metrics>
│   │   ├── 4.notifikasi_<metrics>
│   │   └── dst (adjust based on points achieved)
│   ├── 7.inference.py                        # Inference script for the served model.
│   └── folder/file tambahan                  # Additional relevant folders/files for monitoring/logging.
└── .gitignore                                # Specifies intentionally untracked files to ignore.

```
