# FactoryGuard AI - IoT Predictive Maintenance Engine

## Project Overview
This project implements a **production-ready, end-to-end machine learning system** to predict component failure in 24 hrs.  
The focus is on **imbalanced classification**, **robust evaluation using PR-AUC**, **threshold-aware decision-making**, and **deployment**.

The project goes beyond model training and covers:
- Correct metric selection for imbalanced data
- Cross-validated feature selection
- Multi-model benchmarking
- Threshold tuning aligned with business objectives
- Flask API deployment

---

## Dataset  
- **Source:** [Kaggle - NASA C-MAPSS-1 Turbofan Engine Degradation](https://www.kaggle.com/datasets/bishals098/nasa-turbofan-engine-degradation-simulation)
- **Size:** (20631, 26)  
- **Target:** feature engineered column `label_24` (0/1) failure column

---
## Project Structure
```
├── data/
    ├──raw/                              # Directory containing datasets
        ├── RUL_FD001.txt
        ├── test_FD001.txt
        ├── train_FD001.txt
├── notebooks/
    ├── 01_eda.ipynb                    # EDA notebook for initial data analysis
    ├── 02_feature_engineering.ipynb    # Week 1: Feature Engineering
    ├── 03_modelling.ipynb              # Week 2: Modeling with Imbalance Handling
    ├── 04_explainability_shap.ipynb    # WeeK 3: Shap Explainability
    ├── 05_testing_model.ipynb          # Testing saved model on data/raw/test_FD001.txt
├── output/            
    ├── features.pkl                    # Selected features list
    ├── final_model.pkl                 # Saved LightGBM model
├── api_app.py                          # Backend API-based version
├── api_request.py                      # Interaction with Backend API
├── ui_app.py                           # UI-based version for Demo
├── requirements.txt                    # Requirments for this project
├── README.md                           # This File
```
---
## Data Preprocessing

This dataset contains columns like engine_id, cycle, operating_setting (3 columns), sensor columns(21 columns)
- Removed columns have near zero variance
- Calculated `max_cycle` per engine
- Computed `RUL (Remaining Usable Life)` from `max_cycle` and `cycle` columns
- Created lag features for all sensors `t-1` & `t-2` w.r.t `engine_id`
- Created Rolling mean, Standard Deviation with window size 1,6,12 and Exponential Moving Average with 12 window size
- Created `label_24` column from `RUL` column where RUL<24 indicating failure in next 24 hrs
- Dropped Null values created by lag features
---
## Modeling
Dropped `cycle`,`max_cycle` and `RUL` columns to prevent memorization as they already reflect `label_24` column.
### Feature Selection
Training the model using all 117 engineered features was sub-optimal due to redundancy and overfitting risk. To address this, a LightGBM model with class weights was trained using StratifiedGroupKFold cross-validation to prevent engine-level data leakage between training and validation sets. Feature importances were extracted from each fold, aggregated across folds, and weighted by the corresponding PR-AUC score to prioritize features that consistently contributed to strong minority-class performance there by selecting top 10% of features with higher importances.

GroupShuffleSplit was performed to split data into both train and validation sets. Class weights is passed to both models to mitigate class imbalance during training.

---

### Baseline Model (RandomForestClassifier) performance evaluated on validation dataset

**Average Precision-Recall (PR) Score:** `0.9339`

#### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|------:|----------:|-------:|---------:|--------:|
| 0 (Healthy) | 0.97 | 0.98 | 0.98 | 5,044 |
| 1 (Failure) | 0.86 | 0.83 | 0.84 | 750 |
| **Accuracy** |  |  | **0.96** | **5,794** |
| **Macro Avg** | 0.92 | 0.90 | 0.91 | 5,794 |
| **Weighted Avg** | 0.96 | 0.96 | 0.96 | 5,794 |

**ROC–AUC Score:** `0.9887`

---

### Production Model (LGBMClassifier) performance evaluated on validation dataset

**Final Precision–Recall (PR) Score:** `0.9356`

#### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|------:|----------:|-------:|---------:|--------:|
| 0 (Healthy) | 0.98 | 0.96 | 0.97 | 5,044 |
| 1 (Failure) | 0.77 | 0.90 | 0.83 | 750 |
| **Accuracy** |  |  | **0.95** | **5,794** |
| **Macro Avg** | 0.88 | 0.93 | 0.90 | 5,794 |
| **Weighted Avg** | 0.96 | 0.95 | 0.95 | 5,794 |

**Final ROC–AUC Score:** `0.9882`

---
#### LGB Model performance across different thresholds.
![Performance](https://github.com/user-attachments/assets/1a610821-069d-4bd4-90b1-584e2a08c2c8)


Based on the graph, **optimal threshold 0.6** is selected for improved precision to reduce false alarms while still providing early detection

##### Model Evaluation Metrics (Threshold = 0.6)

**Precision–Recall (PR) Score:** `0.9356`

###### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|------:|----------:|-------:|---------:|--------:|
| 0 (Healthy) | 0.98 | 0.97 | 0.97 | 5,044 |
| 1 (Failure) | 0.79 (↑2%) | 0.88 (↓2%) | 0.83 | 750 |
| **Accuracy** |  |  | **0.95** | **5,794** |
| **Macro Avg** | 0.89 | 0.92 | 0.90 | 5,794 |
| **Weighted Avg** | 0.96 | 0.95 | 0.96 | 5,794 |

**ROC–AUC Score:** `0.9297`

---
## Deployment
The application is deployed as two independent Flask services: a JSON-based inference API and a web-based UI for interactive testing and demonstrations. Both services rely on shared model artifacts stored on disk.

**Model Artifacts:**
All trained artifacts required for inference are stored in the `output/` directory.These artifacts are loaded at runtime by both the API and UI services to ensure feature consistency and reproducible predictions.

**API Service (Inference Layer)**
The inference API exposes a REST endpoint for programmatic access, benchmarking, and production integration through `api_app.py` & `api_request.py`

Start the API server: `python api_app.py`

**API Testing (Client)**
A lightweight client script to test inference latency and validate risk categorization.
RUN `python api_request.py`

The script sends predefined LOW, MEDIUM, HIGH, and CRITICAL payloads/ User-defined to the API and reports:

    * Failure probability
    * Risk level
    * Client-side latency

**Web UI Service (Demonstration Layer)**\
The UI service provides a lightweight HTML interface for manual and preset-based model evaluation using `ui_app.py` & template `templates/index.html`.

Start the UI server: `python ui_app.py`

---
## Contributors
- **Kalavakuri Abhinay:**  Feature Engineering, Model Developemt , Deployment (API-Based) ,Documentation
- **Prathmesh Yadav:** SHap Analysis, Deployment (UI-Based)

