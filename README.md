# Machine Learning Approach to Renewable Energy Dynamics

## Project Overview
This project analyzes the factors determining the share of renewable energy in a country's final energy consumption. [cite_start]Using global data from 2000-2020, we apply various machine learning models to predict renewable energy adoption dynamics[cite: 194, 247].

[cite_start]**Authors:** Grzegorz Mozdzynski, Hajar Sriri, Chopard Jules[cite: 195].

## Dataset
[cite_start]The data is sourced from the World Bank, IEA, and "Our World in Data"[cite: 245]. It includes:
* **Energy variables:** Access to electricity, clean fuels, renewable capacity, etc.
* **Economic variables:** GDP per capita, financial flows, etc.
* **Geographic variables:** Density, Latitude, Longitude.

## Methodology
The project follows a rigorous data science pipeline:
1.  [cite_start]**Data Cleaning:** KNN Imputation ($k=3$) to handle missing values while preserving local tendencies[cite: 281].
2.  [cite_start]**Feature Selection:** Mixed stepwise selection to identify the most significant predictors (e.g., Renewable generating capacity, Energy intensity, GDP per capita)[cite: 315].
3.  **Modeling:** We compared four models:
    * Linear Regression (OLS)
    * Lasso Regression
    * Ridge Regression
    * [cite_start]Random Forest Regressor [cite: 325-328].

## Key Results
The Random Forest model significantly outperformed linear approaches, suggesting that the relationship between economic/energy factors and renewable adoption is non-linear.

| Model | R² (Test) | RMSE (Test) |
|-------|-----------|-------------|
| Linear Regression | 0.7879 | 13.37 |
| Ridge Regression | 0.7881 | 13.36 |
| Lasso Regression | 0.7886 | 13.35 |
| **Random Forest** | **0.9839** | **3.69** |

[cite_start]*Table Reference: [cite: 370]*

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt