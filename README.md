# Vehicle Price Prediction Using Supervised Machine Learning
*GitHub | Jul '25 - Sep '25*

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=kunalawasthi029/Used-Vehicle-Price-Predictor-App&branch=main&mainModule=app/streamlit_app.py)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2%2B-orange?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-red)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit&logoColor=white)

An end-to-end Machine Learning application that predicts used car prices from historical vehicle listing data. The project includes data preprocessing, exploratory data analysis (EDA), feature engineering, model comparison, and deployment as an interactive web application.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Model Performance](#model-performance)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Overview

This project builds a regression pipeline to predict the selling price of used vehicles based on features such as manufacturer, year, mileage, condition, fuel type, and more. Three models are trained and compared:

| Model | Description |
|---|---|
| **Linear Regression** | Baseline model for comparison |
| **Random Forest** | Ensemble tree-based model |
| **XGBoost** | Gradient boosting - best performer |

The best-performing **XGBoost** model is deployed as a **Streamlit** web app for real-time price estimation.

---

## Dataset

- **Source**: `cleaned_vehicles1.csv`
- **Records**: 156,789 used car listings
- **Features**: 18 columns (numeric + categorical)
- **Target**: `price` (USD $500 - $100,000)

### Key Features

| Feature | Type | Description |
|---|---|---|
| `year` | Numeric | Vehicle model year (2015-2022) |
| `manufacturer` | Categorical | 38 brands (Ford, Toyota, etc.) |
| `condition` | Categorical | Vehicle condition rating |
| `cylinders` | Categorical | Engine cylinder count |
| `fuel` | Categorical | Fuel type (gas, diesel, hybrid, etc.) |
| `odometer` | Numeric | Mileage in miles |
| `transmission` | Categorical | Transmission type |
| `drive` | Categorical | Drivetrain (4WD, FWD, RWD) |
| `type` | Categorical | Vehicle type (sedan, SUV, pickup, etc.) |
| `car_age` | Numeric | Derived feature (2025 - year) |

---

## Project Structure

```
Vehical Price Prediction/
├── cleaned_vehicles1.csv           # Raw dataset
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation (this file)
│
├── src/                            # Source code
│   ├── __init__.py
│   ├── data_preprocessing.py       # Data cleaning & feature engineering
│   ├── manufacturer_mapping.py     # Real-world manufacturer & feature catalog
│   ├── model_training.py           # Model training, evaluation & plots
│   └── utils.py                    # Utility / helper functions
│
├── app/                            # Web application
│   └── streamlit_app.py            # Streamlit interactive app
│
├── models/                         # Saved model artifacts
│   ├── xgboost_best_model.pkl      # Trained XGBoost model
│   ├── label_encoders.pkl          # Fitted label encoders
│   ├── feature_columns.pkl         # Feature column order
│   └── model_metrics.pkl           # Performance metrics
│
├── plots/                          # Generated visualizations
│   ├── price_distribution.png
│   ├── price_by_manufacturer.png
│   ├── price_by_type.png
│   ├── price_by_fuel.png
│   ├── price_vs_odometer.png
│   ├── correlation_heatmap.png
│   ├── model_comparison.png
│   ├── feature_importance.png
│   └── predicted_vs_actual.png
│
└── notebooks/                      # Jupyter notebooks
    └── EDA_and_Modeling.ipynb       # Interactive EDA & modeling notebook
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/kunalawasthi029/Used-Vehicle-Price-Predictor-App.git
cd Used-Vehicle-Price-Predictor-App

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Train the Models

Run the full training pipeline (preprocessing, EDA, training, evaluation):

```bash
python -m src.model_training
```

This will:
- Generate EDA visualizations in `plots/`
- Train Linear Regression, Random Forest, and XGBoost models
- Save the best model and artifacts to `models/`
- Generate model comparison charts

### 2. Launch the Web App

```bash
streamlit run app/streamlit_app.py
```

Open your browser and navigate to `http://localhost:8501` to use the interactive price predictor.

---

## Methodology

### Data Preprocessing
1. **Dropped columns**: `id`, `region`, `model` (too many categories), `price_per_mile` (data leakage)
2. **Feature extraction**: Converted cylinder strings to numeric values
3. **Encoding**: Label-encoded all categorical features
4. **Split**: 80/20 train-test split with fixed random state for reproducibility

### Feature Engineering
- Extracted numeric cylinder count from strings (e.g., "8 cylinders" -> 8)
- Used `car_age` (pre-computed in dataset) as a feature
- Handled "Unknown" values as a distinct category

### Models Trained
1. **Linear Regression** - Simple baseline for benchmarking
2. **Random Forest** - 200 trees, max depth 15
3. **XGBoost** - 300 estimators, learning rate 0.1, max depth 8

### Evaluation Metrics
- **MAE** (Mean Absolute Error) - Average prediction error in dollars
- **RMSE** (Root Mean Squared Error) - Penalizes large errors more heavily
- **R2 Score** - Proportion of variance explained (1.0 = perfect)

---

## Model Performance

| Model | MAE | RMSE | R2 Score |
|---|---|---|---|
| Linear Regression | $8,417 | $11,636 | 0.3172 |
| Random Forest | $4,082 | $6,871 | 0.7619 |
| **XGBoost (Best)** | **$3,566** | **$5,981** | **0.8196** |

> XGBoost outperforms both baselines with 82% variance explained and the lowest error across all metrics.

---

## Tech Stack

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computing |
| **Scikit-learn** | ML models and preprocessing |
| **XGBoost** | Gradient boosting regressor |
| **Matplotlib / Seaborn** | Data visualization |
| **Streamlit** | Web application deployment |
| **Joblib** | Model serialization |

---


