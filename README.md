# Concrete Slump and Strength Prediction

Machine learning project for predicting concrete workability and compressive strength using concrete mix composition data.

This project applies regression models and neural network techniques to analyze how cement, slag, fly ash, water, superplasticizer, aggregates, slump, and flow influence concrete performance.

---

# Tech Stack

`Python` `Pandas` `NumPy` `Scikit-learn` `MLPRegressor` `XGBoost` `LightGBM` `Matplotlib` `Seaborn`

---

# Project Overview

Concrete quality depends heavily on mix composition and workability. Traditional concrete testing can be time-consuming, so this project explores machine learning as a faster data-driven method for estimating concrete slump and compressive strength.

The workflow includes:

- data preprocessing
- exploratory data analysis
- correlation analysis
- visualization
- regression modeling
- neural network training
- ensemble model comparison
- model evaluation using RMSE, MAE, and R²

---

# Dataset

The dataset contains concrete mix design and performance attributes, including:

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

Dataset size: `103 rows × 10 columns`

---

# Repository Structure

```text
concrete-slump-strength-prediction/
├── data/
│   └── cement_slump.csv
├── notebooks/
│   ├── 01_mlp_modeling.ipynb
│   └── 02_model_comparison.ipynb
├── README.md
└── requirements.txt
```

---

# Notebooks

## 01_mlp_modeling.ipynb

Focused notebook for:

- exploratory data analysis
- correlation heatmaps
- distribution plots
- boxplots
- scatterplots
- MLP regression modeling
- model evaluation

## 02_model_comparison.ipynb

Broader experimentation notebook covering:

- preprocessing
- EDA
- model training
- ensemble methods
- LightGBM / XGBoost experimentation
- model comparison

---

# Models Used

- MLP Regressor
- XGBoost
- LightGBM
- Regression-based model comparison workflows

---

# Evaluation Metrics

Models were evaluated using:

- RMSE
- MAE
- R² Score

---

# Skills Demonstrated

- Regression modeling
- Neural networks
- Ensemble learning
- Engineering analytics
- Feature analysis
- Data preprocessing
- Exploratory data analysis
- Model evaluation

---

# Key Learning

This project shows how machine learning can support construction material analysis by predicting concrete behavior from mix composition data. Ensemble models and neural networks can help identify important factors influencing workability and strength.

---

# Notes

- This is a lightweight portfolio version of an academic machine learning project.
- The repository contains cleaned notebooks and dataset files for reproducibility.
- The project is intended for learning, experimentation, and portfolio demonstration.
