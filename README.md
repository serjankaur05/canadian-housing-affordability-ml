# Canadian Housing Affordability | Machine Learning & Predictive Analytics

**Using machine learning and explainable AI to understand housing affordability challenges across Canada.**

`Python` `Scikit-Learn` `XGBoost` `K-Means` `SHAP` `Pandas` `NumPy`

## Overview

How can machine learning help us better understand housing affordability across Canada?

This project explores that question by analyzing demographic and household spending data across **56,857 Canadian geographic areas**. We developed a predictive modelling workflow to identify factors associated with housing stress, compare machine learning approaches, and translate complex data into interpretable insights.

The goal wasn't simply to achieve high predictive accuracy. It was to explore how data-driven models could support more informed decisions while remaining transparent about their limitations.

## What We Built

- **Data Processing:** Integrated socioeconomic and household spending datasets, implemented preprocessing pipelines, and removed leakage-prone variables to improve model reliability.
- **Unsupervised Learning:** Applied K-Means clustering to explore patterns in household characteristics and housing affordability.
- **Predictive Modelling:** Trained and evaluated Elastic Net and XGBoost models using cross-validation and held-out testing.
- **Explainable AI:** Applied SHAP analysis to understand which features influenced model predictions.

## Key Results

| Metric | Result |
|---|---|
| Geographic areas analyzed | 56,857 |
| XGBoost Test R² | **0.5807** |
| Elastic Net Test R² | 0.4049 |
| K-Means Silhouette Score | 0.775 |
| Best-performing predictive model | XGBoost |

XGBoost outperformed the linear baseline, while SHAP analysis highlighted the importance of household income, housing tenure, and spending characteristics.

These findings demonstrate how combining predictive modelling with interpretability can produce more useful insights than model accuracy alone.

## What We Learned

Building this project reinforced three important lessons:

1. **Data quality comes first.** Preventing leakage and validating preprocessing decisions were essential for meaningful model evaluation.
2. **Experimentation matters.** Comparing approaches revealed important non-linear relationships that simpler models could not fully capture.
3. **Explainability is essential.** A prediction becomes more useful when stakeholders can understand its drivers, limitations, and appropriate applications.

## Technical Implementation

The repository contains Python modules for data preparation, modelling utilities, clustering, and visualization, alongside a Jupyter notebook.

The complete analytical results are documented in the research paper below. The modular source files currently represent foundational components rather than the full final modelling pipeline.

**Data availability:** The original Environics Analytics datasets are excluded from this public repository due to academic data-use restrictions.


## Collaborators

Developed collaboratively as part of Data Science 3000 at Western University.

- [Serjan Kaur](https://github.com/serjankaur05)
- [Sudrisha Sarkar](https://github.com/SudrishaSarkar)
- Gabriel Hernandez

*Data provided by Environics Analytics for academic research.*
