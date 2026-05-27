# Classification Models & Model Evaluation – Supervised Learning

Practical work developed for the Artificial Intelligence Engineering program at Universidad de San Andrés. This project extends the supervised learning pipeline into classification tasks, focusing on training, comparing, and evaluating multiple models on real-world datasets.

---

## Overview

This notebook covers the full workflow for classification problems:

1. **Data Preprocessing:** cleaning, encoding categorical variables, and normalizing features.
2. **Feature Engineering:** selecting the most relevant variables for classification.
3. **Model Training:** implementing and comparing multiple classification algorithms.
4. **Validation:** using train/validation/test splits to assess generalization.
5. **Evaluation:** analyzing performance through accuracy, F1-score, precision, recall, and confusion matrices.

---

## Key Concepts

- Binary and multiclass classification
- Model comparison and selection
- Evaluation metrics: accuracy, precision, recall, F1-score
- Confusion matrix analysis
- Overfitting detection and mitigation

## Project Structure

```
├── data/
│   └── raw/                  # Raw input data
├── notebooks/                # Jupyter notebooks
├── src/
│   ├── data_splitting.py     # Train/test split logic
│   ├── metrics.py            # Evaluation metrics
│   ├── models.py             # Model definitions
│   ├── preprocessing.py      # Data preprocessing pipeline
│   └── utils.py              # Helper functions
└── README.md
```

*Part of the Machine Learning and Deep Learning coursework — AI Engineering, Universidad de San Andrés.*

