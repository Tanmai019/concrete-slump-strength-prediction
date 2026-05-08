# Concrete Slump and Strength Prediction

Machine learning project for predicting concrete workability and compressive strength using concrete mix composition data and regression modeling techniques.

This project applies neural networks and ensemble machine learning models to analyze how cement, slag, fly ash, water, aggregates, and admixtures influence concrete performance and slump behavior.

---

# Tech Stack

`Python` `Pandas` `NumPy` `Scikit-learn` `MLPRegressor` `XGBoost` `LightGBM` `Matplotlib` `Seaborn`

---

# Project Overview

Concrete quality and workability depend heavily on material composition and environmental factors. Traditional concrete testing can be time-consuming and resource-intensive, so this project explores machine learning as a faster predictive approach for estimating concrete slump behavior and compressive strength.

The workflow includes:

- data preprocessing
- exploratory data analysis
- correlation analysis
- visualization
- regression modeling
- neural network experimentation
- ensemble model comparison
- model evaluation

---

# Dataset

The dataset contains concrete mix design and performance attributes including:

- Cement
- Slag
- Fly ash
- Water
- Superplasticizer
- Coarse aggregate
- Fine aggregate
- Slump
- Flow
- 28-day compressive strength

Dataset size:

```text
103 rows × 10 columns
```

---

# Repository Structure

```text
concrete-slump-strength-prediction/
├── data/
│   └── cement_slump.csv
├── notebooks/
│   ├── 01_mlp_modeling.py
│   └── 02_model_comparison.py
├── README.md
└── requirements.txt
```

---

# Project Workflows

## 1. MLP Modeling Pipeline

`01_mlp_modeling.py`

This workflow focuses on:

- exploratory data analysis
- correlation heatmaps
- boxplots and distributions
- neural network regression
- model evaluation using regression metrics

The primary model used is:

- Multi-Layer Perceptron Regressor (MLPRegressor)

---

## 2. Model Comparison Pipeline

`02_model_comparison.py`

This workflow expands the experimentation process and includes:

- preprocessing
- regression modeling
- ensemble methods
- XGBoost experimentation
- LightGBM experimentation
- comparative model evaluation

---

# Models Used

- Multi-Layer Perceptron (MLP)
- XGBoost Regressor
- LightGBM Regressor
- Regression-based comparison workflows

---

# Evaluation Metrics

Models were evaluated using:

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

---

# Key Learning Outcomes

This project demonstrates how machine learning can support construction and material engineering workflows through predictive analytics.

Key observations include:

- material composition strongly influences slump behavior
- ensemble methods improve predictive stability
- neural networks effectively capture nonlinear relationships
- preprocessing and feature analysis significantly affect model accuracy

---

# Skills Demonstrated

- Regression modeling
- Neural networks
- Ensemble learning
- Engineering analytics
- Feature engineering
- Data preprocessing
- Exploratory data analysis (EDA)
- Predictive modeling

---

# Installation

Clone the repository:

```bash
git clone https://github.com/Tanmai019/concrete-slump-strength-prediction.git
cd concrete-slump-strength-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running The Project

Run the MLP workflow:

```bash
python notebooks/01_mlp_modeling.py
```

Run the model comparison workflow:

```bash
python notebooks/02_model_comparison.py
```

---

# Notes

- This repository contains cleaned Python workflows adapted from academic machine learning experimentation.
- Large generated artifacts and temporary files were excluded for repository optimization.
- The project is intended for learning, experimentation, and portfolio demonstration purposes.
