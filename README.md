# DSA4262-BurnoutManager

This repository contains an end-to-end data science pipeline designed to predict employee stress levels and simulate chronic burnout trajectories. By leveraging gradient-boosted models and temporal filtering, the project aims to provide a framework for proactive workplace interventions.

## 📂 Project Structure

1. **`Model Training.ipynb`**: Handles the full machine learning lifecycle, including Exploratory Data Analysis (EDA), feature engineering, and benchmarking various regression models to predict stress levels.
2. **`Burnout Simulation.ipynb`**: Implements a simulation engine that uses Dual Exponential Moving Averages (EMA) to monitor employee health over time, distinguishing between temporary stress spikes and chronic burnout states.

---

## 🧠 Model Training

### Feature Correlation
We began by analyzing the linear relationships between all available features in the dataset to identify multi-collinearity and primary drivers of employee distress.

[Insert Figure: Correlation Matrix]

### Predicting Stress vs. Burnout
A key design choice in this project is the decision to predict `stress_level` instead of `burnout_risk`. We define **burnout** as the result of prolonged and excessive stress rather than a single point-in-time state. By predicting the immediate `stress_level`, we can apply downstream logic (such as moving averages) to identify when high stress transitions into a chronic burnout state.

### Feature Selection & Ablation Study
To ensure model efficiency and interpretability, we performed an ablation test to observe the drop in performance when specific variables were removed.

[Insert Figure: Ablation Test Performance Drop]

We can see that the model’s performance drastically drops when features like `workload_score`, `satisfaction_score`, `career_progression_score`, `performance_score`, and `tenure_months` are removed. Thus, we decided to narrow down our input features to just these 5 variables.

### Model Benchmarking
We compared three state-of-the-art gradient boosting frameworks. The models were evaluated using Root Mean Squared Error (RMSE) on a consistent validation set.

| Model | Validation RMSE |
| :--- | :--- |
| **LightGBM** | **0.1531** |
| XGBoost | 0.1532 |
| CatBoost | 0.1534 |

**Final Performance**
Since LightGBM yielded the lowest error, it was selected as the final production model.

* **Final LightGBM Test RMSE:** 0.1542

---

## 🛠️ Requirements
* Python 3.x
* LightGBM
* XGBoost
* CatBoost
* Scikit-Learn
* Pandas / Numpy
* Matplotlib / Seaborn

