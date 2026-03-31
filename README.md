# Hotel Booking Demand - Machine Learning Assignment Package

This package contains Jupyter notebooks and report templates for a supervised learning group assignment
using the Hotel Booking Demand dataset.

## Folder structure

- `dataset/`  
  Place the Hotel Booking Demand CSV file here. The notebooks automatically load the first supported file.
- `outputs/`
  - `describing/`
  - `logistic_regression/`
  - `decision_tree/`
  - `random_forest/`
  - `gradient_boosting/`
  - `comparison/`
- `Dataset_Description.ipynb`
- `Logistic_Regression.ipynb`
- `Decision_Tree.ipynb`
- `Random_Forest.ipynb`
- `Gradient_Boosting.ipynb`
- `Model_Comparison.ipynb`
- `hotel_booking_common.py`
- `reports/`
- `members.txt`
- `submission.txt`

## Recommended execution order

1. Put the dataset file inside the `dataset/` folder.
2. Run `Dataset_Description.ipynb`.
3. Run `Logistic_Regression.ipynb`.
4. Run `Decision_Tree.ipynb`.
5. Run `Random_Forest.ipynb`.
6. Run `Gradient_Boosting.ipynb`.
7. Run `Model_Comparison.ipynb`.

## What the notebooks do

- Load and describe the dataset.
- Detect missing values, duplicates, anomalies, outliers, and leakage columns.
- Perform assignment-friendly preprocessing and feature engineering.
- Tune each algorithm using cross-validation.
- Save best models as `.joblib`.
- Save metrics, predictions, feature importance tables, and visualizations as files.
- Compare all four saved models on the same hold-out test split.

## Notes for the final report

The assignment requires the report to compare the four algorithms, describe the dataset, explain preprocessing,
include critical discussion, and add the source code as text in the appendix. The files in `reports/` are
draft templates that should be updated after running the notebooks and filling in team/member details.
