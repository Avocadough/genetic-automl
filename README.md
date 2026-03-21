# 🧬 Meta-AutoML System <br> *(Solving CASH with Genetic Algorithms)*

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python) 
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white) 
![XGBoost](https://img.shields.io/badge/XGBoost-1296DB?style=for-the-badge&logo=xgboost&logoColor=white)
![Custom GA](https://img.shields.io/badge/Custom_GA-Built_from_Scratch-success?style=for-the-badge)

A powerful, custom-built **Automated Machine Learning (AutoML)** framework designed completely from scratch using Python. 

The system utilizes a custom **Genetic Algorithm (GA)** to solve the *Combined Algorithm Selection and Hyperparameter optimization (CASH)* problem. It seamlessly hunts and evolves the perfect end-to-end `scikit-learn` Pipeline directly tailored to your dataset.

## ✨ Key Features

- **End-to-End Pipeline Evolution**: Searches across Preprocessors (Standard/MinMax), Feature Extractors (PCA/SelectKBest), Models (Linear, SVM, Trees, MLP), and extensive hyperparameter domains.
- **Classification & Regression Support**: Automatically targets `F1-macro` or `Negative-RMSE` depending on your task.
- **100% Reproducibility**: Granular seed propagation guarantees that your experimental findings are mathematically reproducible.
- **Fail-safe Timeout Handlers**: Protected against hanging pipelines natively via daemon threads (rogue SVMs/MLPs are punished, not frozen).
- **Explicit Elitism**: The best traits are guaranteed mathematically never to degrade across GA generations.
- **OOP Architecture**: Highly modular, clean codebase ideal for academics, researchers, and professional ML engineers.

---

## ⚙️ Installation

To use the `MetaAutoML` module locally, simply clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/Meta-AutoML.git
cd Meta-AutoML
pip install -r requirements.txt
```

---

## 🚀 Quick Start & Usage

The top-level `MetaAutoML` class mirrors the standard scikit-learn API, abstracting away all GA complexities behind standard `.fit()` and `.predict()` methods.

```python
from sklearn.datasets import load_breast_cancer
from meta_automl import MetaAutoML

# 1. Load Data
X, y = load_breast_cancer(return_X_y=True)

# 2. Initialize the GA Optimizer
automl = MetaAutoML(
    task_type="classification", # Or "regression"
    pop_size=20,                # Chromosomes per generation
    n_generations=15,           # Evolutionary cycles
    cv_folds=5,                 # K-Fold Evaluation
    timeout_seconds=30,         # Guard against infinite models
    random_state=42             # Total Reproducibility
)

# 3. Fit (Evolve) the Pipeline
automl.fit(X, y, verbose=True)

# 4. View visually engaging Summary
automl.summary()

# 5. Predict or extract pipeline
predictions = automl.predict(X)
best_model = automl.get_best_pipeline() 
```

---

## 📂 Repository Structure

```text
AutoML_Track/
├── meta_automl/
│   ├── __init__.py           # Package exports
│   ├── search_space.py       # Defines Models, Scaling, & HPs
│   ├── chromosome.py         # Pipeline Encoding
│   ├── fitness.py            # Async K-Fold CV Evaluation
│   ├── genetic_operators.py  # Tournament, Crossover, Mutation
│   ├── ga_engine.py          # Master Evolutionary Loop
│   └── meta_automl.py        # Top-level Human API
├── main.py                   # Executable Demo (Breast Cancer/Diabetes)
├── requirements.txt
├── README.md                 # Project Overview
└── project_review_for_LLM.md # Deep context file for Agentic AI
```

---
*Developed as a robust Metaheuristic Optimization solution.* 🧬📈
