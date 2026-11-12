# Explainable Machine Learning for Electrical Discharge Machining (EDM)

## Project Overview

This project explores the application of Explainable Machine Learning (XML) techniques for modelling and analysing the Electrical Discharge Machining (EDM) process.

The objective is to compare conventional and advanced machine learning models, understand the influence of machining parameters, and investigate parameter optimisation using explainable AI techniques.

The study uses 262 experimental observations involving hardened EN31 steel and a sintered cermet tool electrode (75% Cu–25% TiC).

## Objectives

1. Develop machine learning models to predict EDM performance.
2. Compare model performance using appropriate evaluation metrics and cross-validation.
3. Interpret model behaviour using Explainable AI techniques.
4. Investigate inverse optimisation to identify suitable machining parameter combinations.

## Dataset

- Number of experimental observations: 262
- Workpiece material: Hardened EN31 steel
- Tool electrode: Cu–TiC (75% Cu–25% TiC)
- Process: Electrical Discharge Machining

The dataset is available in the `data/` directory.

## Machine Learning Models

The following models are investigated:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor
- Explainable Boosting Machine (EBM)

## Explainable AI Techniques

This project incorporates:

- Explainable Boosting Machine (EBM)
- SHAP (SHapley Additive exPlanations)
- Feature importance analysis
- Feature effect interpretation

These techniques help understand the relationship between EDM process parameters and machining responses.

## Model Evaluation

Model performance is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)
- Five-fold Cross-Validation

Detailed evaluation outputs are available in the `results/` directory.

## Inverse Optimisation

An inverse optimisation approach is explored to identify EDM process parameter combinations corresponding to desired machining responses.

The implementation and generated outputs are available in the source code and results directories.

## Repository Structure

```text
edm-explainable-machine-learning/
│
├── data/
│   └── EDM_dataset_262_samples.csv
│
├── src/
│   ├── cross_validation.py
│   ├── ebm.py
│   ├── eda.py
│   ├── explainability.py
│   ├── models.py
│   ├── EDM_reproduction_and_EBM_CV_fixed.py
│   └── inverse_optimization.py
│
├── figures/
│   ├── Figure_1.png
│   ├── Figure_2.png
│   ├── Figure_3.png
│   └── Figure_4.png
│
├── results/
│   └── Model evaluation and optimisation outputs
│
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Abhirup845/edm-explainable-machine-learning.git
```

Navigate to the project directory:

```bash
cd edm-explainable-machine-learning
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- InterpretML
- SHAP
- SciPy

## Author

**Abhirup Das**

Production Engineering  
Jadavpur University

GitHub: [Abhirup845](https://github.com/Abhirup845)

---

*This project is an academic research project focused on explainable machine learning, EDM process modelling and parameter optimisation.*
