# Hotel Booking Demand — Preprocessing Plan for 4 Supervised Learning Models

## 1. Purpose of this document

This document explains the **full preprocessing strategy** for the Hotel Booking Demand dataset when building and comparing these four supervised learning models for the same binary classification problem:

- **Logistic Regression**
- **Decision Tree**
- **Random Forest**
- **Gradient Boosting**

The classification target is:

- **`is_canceled`**
  - `1` = booking canceled
  - `0` = booking not canceled

The goal is to use **one common cleaned dataset** for fairness, then apply **model-specific final preprocessing** where needed.

---

## 2. Why preprocessing is split into two parts

To compare four algorithms properly, preprocessing should be split into:

### A. Base preprocessing
This is the **common preprocessing** done for all four models.

It ensures:
- fair comparison
- same cleaned data foundation
- same business problem
- same feature engineering logic
- same leakage removal
- same train/test split strategy

### B. Final preprocessing
This is the **model-specific preprocessing** done just before training each individual model.

It changes depending on the algorithm because different models behave differently with:
- feature scaling
- outliers
- skewed data
- correlated variables
- categorical encoding size

---

# 3. Base preprocessing for all 4 models

This section applies to:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

These steps should be common for all models.

---

## 3.1 Load and inspect the dataset

### Step
Load the dataset and perform an initial audit:
- dataset shape
- column names
- data types
- sample rows
- summary statistics
- target distribution
- missing values
- duplicate rows
- cardinality of categorical variables

### Why do this?
Because before preprocessing, we must understand:
- what the dataset contains
- what type of problem it is
- which columns are numerical vs categorical
- which columns have missing values
- whether the target is balanced or imbalanced
- whether there are suspicious patterns or anomalies

### Why this matters for the assignment
This strengthens:
- dataset description
- preprocessing justification
- understanding of the data before modeling

---

## 3.2 Check and clean data types

### Step
Ensure columns have appropriate types:
- numerical columns should be numeric
- categorical columns should be treated as categorical/string
- target should be integer or binary numeric

### Why do this?
Because incorrect data types can:
- break transformations
- cause wrong summary statistics
- cause encoding or scaling errors
- lead to wrong model input handling

### Why not skip this?
If data types are wrong, downstream preprocessing becomes unreliable.

---

## 3.3 Check target variable

### Step
Confirm that:
- `is_canceled` exists
- it is binary
- class distribution is understood

### Why do this?
Because the whole project is a **supervised binary classification task**.

We must confirm:
- the target is valid
- the dataset is usable for classification
- class imbalance is not ignored

### Why not skip this?
If the target is not validated, the entire modeling pipeline may be incorrect.

---

## 3.4 Remove duplicate rows

### Step
Identify and remove exact duplicate rows.

### Why do this?
Duplicates can:
- overstate patterns in the dataset
- make the model appear better than it really is
- bias training and evaluation
- create repeated observations that reduce generalization quality

### Why do this for all models?
Because duplicate removal is a dataset quality issue, not a model-specific issue.

### Why not keep duplicates?
Keeping duplicates can make the learning process unfair and may inflate performance metrics.

---

## 3.5 Identify invalid or suspicious records

### Step
Check for anomalies such as:
- negative `adr`
- bookings with zero total guests (`adults + children + babies == 0`)
- impossible counts
- suspicious extreme values

### Why do this?
Because real-world datasets often contain invalid or noisy records.

These can:
- distort model learning
- distort descriptive statistics
- create meaningless patterns
- reduce trust in results

### Why not remove all extreme values immediately?
Because some high values are real business cases, not errors.
Only clearly invalid values should be removed at the base stage.

---

## 3.6 Drop target leakage columns

### Step
Drop columns that leak information about the final outcome:
- `reservation_status`
- `reservation_status_date`

### Why do this?
These columns are highly related to the final booking outcome and may directly reveal cancellation status.

If they are kept:
- the model may learn information it would not have at prediction time
- performance becomes unrealistically high
- the comparison becomes invalid

### Why must this be done for all models?
Because leakage affects every model and makes all results misleading.

### Why not keep them as "important predictors"?
Because a useful predictor is only valid if it would be available before the cancellation outcome is known.
These columns fail that rule.

---

## 3.7 Handle missing values at the base level

### Step
Handle known missing-value columns consistently:

#### `children`
- fill missing values with `0`

#### `country`
- fill missing values with `"Unknown"`

#### `agent`
- missing usually means no agent involved
- fill with a meaningful placeholder such as `"No_Agent"` or `0`
- create `has_agent`

#### `company`
- missing usually means no company involved
- fill with a meaningful placeholder such as `"No_Company"` or `0`
- create `has_company`

### Why do this?
Because missing values must be handled before modeling.

The reason for each choice:
- `children`: missing can reasonably be treated as zero if no value was recorded
- `country`: unknown country should remain a separate category rather than being dropped
- `agent` and `company`: missing often has business meaning, not just absent data

### Why not drop rows with missing values?
Because dropping rows would:
- unnecessarily reduce dataset size
- remove useful information
- possibly bias the sample

### Why not drop the columns entirely?
Because these columns contain useful business information.
Missingness itself may also be informative.

---

## 3.8 Create missingness indicator features

### Step
Create:
- `has_agent`
- `has_company`

### Why do this?
Because the presence or absence of an agent/company can be predictive of cancellation behavior.

This is useful because:
- missing in `agent` and `company` is not random noise
- it may represent a different booking channel or business type

### Why not rely only on the filled values?
Because a direct binary indicator makes the signal clearer and easier for models to use.

---

## 3.9 Feature engineering

### Step
Create additional meaningful features.

#### A. `total_nights`
`stays_in_weekend_nights + stays_in_week_nights`

**Why do this?**
Because total stay duration is more informative than splitting weekend/weekdays alone.

#### B. `total_guests`
`adults + children + babies`

**Why do this?**
Because total group size is an important booking characteristic.

#### C. `is_family`
`1` if children or babies are present, else `0`

**Why do this?**
Because family bookings may behave differently from solo or couple bookings.

#### D. `room_changed`
`1` if `reserved_room_type != assigned_room_type`, else `0`

**Why do this?**
Because room reassignment may reflect operational patterns connected with cancellations or booking complexity.

#### E. `arrival_month_num`
Convert month name to numerical order.

**Why do this?**
Because month names are categorical text, but seasonality is often better captured in numerical or grouped form.

#### F. `arrival_season`
Group months into seasonal categories.

**Why do this?**
Because seasonality can affect hotel demand and cancellation behavior.

### Why do feature engineering at the base stage?
Because these features are meaningful for all models and should be available fairly to all of them.

### Why not engineer too many features?
Because too many complex features can:
- reduce interpretability
- increase noise
- create overfitting risk
- make the report harder to justify

---

## 3.10 Define final input columns

### Step
Separate features into:
- numerical features
- categorical features
- target variable

### Why do this?
Because each type needs different downstream transformations.

This also helps:
- column-based preprocessing pipelines
- clear documentation
- reproducibility

---

## 3.11 Train/test split

### Step
Split the cleaned data into training and testing sets using:
- stratified split
- fixed random state

### Why do this?
Because the same split must be used for all four models for fair comparison.

Stratification is important because:
- it preserves the class distribution
- it makes evaluation more reliable

### Why not use a different split for each model?
Because then performance differences could come from data differences instead of model differences.

---

## 3.12 Outlier handling at the base stage

### Step
Only remove or correct **clearly invalid** outliers at the common stage.
Examples:
- impossible negative values where they do not make sense
- impossible counts

### Why do only limited outlier handling here?
Because not all extreme values are errors.
Some extreme bookings are genuine real-world observations.

### Why not aggressively trim all outliers for everyone?
Because:
- tree-based models are usually robust to outliers
- aggressive trimming may remove valid business information
- it may unfairly favor some models over others

---

# 4. Final preprocessing for each model

Now the model-specific stage begins.

Each model gets:
- the same base-cleaned dataset
- the same train/test split logic
- the same target

But the final transformations differ.

---

# 5. Logistic Regression — Final preprocessing

## 5.1 Why Logistic Regression needs special preprocessing

Logistic Regression is a **linear model**.
It is more sensitive to:
- different feature scales
- skewed numerical distributions
- multicollinearity
- extreme numeric values
- very high-dimensional sparse encodings

Because of this, it usually requires the most careful preprocessing among the four models.

---

## 5.2 Final preprocessing steps for Logistic Regression

### Step 1: Start from the base-cleaned dataset
Use the common cleaned dataset after:
- leakage removal
- missing value handling
- feature engineering
- duplicate removal

### Why?
To ensure fairness and consistency.

---

### Step 2: Separate numeric and categorical columns

### Why?
Because numeric and categorical features need different transformations:
- numeric features need scaling
- categorical features need encoding

---

### Step 3: Numerical imputation (if still needed)
Use:
- median imputation for numeric columns

### Why do this?
Because Logistic Regression cannot handle missing values directly.
Median is robust to skew and outliers.

### Why median instead of mean?
Because hotel booking variables such as `lead_time`, `adr`, and `days_in_waiting_list` can be skewed.
Median is less distorted by extreme values.

---

### Step 4: Categorical imputation (if still needed)
Use:
- most frequent category or a defined placeholder

### Why do this?
Because one-hot encoding requires complete categorical input.

### Why not drop missing categories?
Because dropping rows loses data and may remove useful patterns.

---

### Step 5: One-hot encode categorical features
Use:
- `OneHotEncoder(handle_unknown='ignore')`

### Why do this?
Because Logistic Regression requires numerical inputs.
Categorical text values cannot be used directly.

One-hot encoding is appropriate because:
- it avoids false ordinal meaning
- it represents categories clearly
- it works well with linear models

### Why not label encode nominal categories?
Because label encoding would create fake order relationships, such as implying one market segment is larger or smaller than another numerically.

---

### Step 6: Scale numeric features
Use:
- `StandardScaler()`

### Why do this?
Because Logistic Regression is sensitive to scale.
Features with large magnitudes can dominate the optimization process.

Scaling helps:
- faster and more stable convergence
- fairer coefficient learning
- better regularization behavior

### Why is scaling necessary here but not for tree models?
Because Logistic Regression uses a linear optimization process based on feature values directly, while tree models split based on thresholds and are mostly scale-invariant.

---

### Step 7: Consider skew treatment for highly skewed numeric variables
Possible transformations:
- `log1p(lead_time)`
- `log1p(adr)`
- `log1p(days_in_waiting_list)`

### Why do this?
Because heavy skew can make linear decision boundaries less effective.
Reducing skew can improve model stability and fit.

### Why is this recommended rather than mandatory?
Because the usefulness depends on actual data distribution and whether the transformed version improves validation results.

### Why not transform every numeric feature?
Because unnecessary transformation may reduce interpretability and does not always improve performance.

---

### Step 8: Consider multicollinearity checks
Check strong correlations among numeric features.
Optionally remove one of two highly redundant features.

### Why do this?
Because Logistic Regression can become unstable when predictors are strongly correlated.
This can affect:
- coefficient interpretation
- numerical stability
- generalization

### Why is this less important for tree-based models?
Because tree models are less sensitive to correlated inputs.

### Why not remove correlated features blindly?
Because some correlated features may still provide useful information.
Only severe redundancy should be addressed.

---

### Step 9: Consider mild outlier clipping for extreme numeric variables
Possible for:
- `adr`
- `lead_time`
- `days_in_waiting_list`

### Why do this?
Because extreme values can pull a linear model disproportionately.
Mild clipping can improve stability.

### Why only mild clipping?
Because extreme hotel bookings may still be genuine and informative.
We should not aggressively remove real-world business cases.

---

## 5.3 Final summary for Logistic Regression

### Do these steps
- common base preprocessing
- imputation
- one-hot encoding
- scaling
- optional skew reduction
- optional multicollinearity check
- optional mild outlier clipping

### Do not skip scaling
Because Logistic Regression needs scaled numeric input.

### Do not use raw categorical text
Because the model needs numeric features.

### Do not use label encoding for nominal variables
Because it introduces false order.

---

# 6. Decision Tree — Final preprocessing

## 6.1 Why Decision Tree needs different preprocessing

Decision Trees are **non-linear rule-based models**.
They are relatively robust to:
- different scales
- monotonic transformations
- correlated variables

They do not need the same strict preprocessing as Logistic Regression.

---

## 6.2 Final preprocessing steps for Decision Tree

### Step 1: Start from the base-cleaned dataset

### Why?
To keep the comparison fair and grounded in the same cleaned data.

---

### Step 2: Numerical imputation (if needed)
Use:
- median imputation

### Why do this?
Because sklearn Decision Trees do not accept missing values in the usual pipeline.
Median is robust and simple.

### Why not leave missing values?
Because the model cannot train reliably with missing inputs in this setup.

---

### Step 3: Categorical imputation (if needed)
Use:
- most frequent category or placeholder

### Why do this?
Because encoding still requires complete data.

---

### Step 4: Encode categorical variables
Use:
- `OneHotEncoder(handle_unknown='ignore')`

### Why do this?
Because the sklearn Decision Tree implementation still requires numeric input.

### Why not keep text categories directly?
Because the implementation cannot split directly on raw text values.

### Why not prefer label encoding here?
One-hot encoding avoids injecting artificial ordinality and keeps the preprocessing consistent across models.

---

### Step 5: Do not scale numeric variables

### Why not scale?
Because Decision Trees split data using thresholds such as:
- `lead_time < 50`
- `adr >= 100`

The exact scale does not affect the choice of split in the same way it affects linear models.

### Why is scaling unnecessary?
Because trees are scale-invariant for this purpose.

### Why can scaling even be avoided intentionally?
Because it adds extra preprocessing complexity without meaningful benefit.

---

### Step 6: Do not perform skew correction as a default step

### Why not?
Because Decision Trees do not assume normality or linearity.
They can learn split points from skewed distributions naturally.

### When might skew handling still be considered?
Only if a clearly extreme distribution is harming performance, but in general it is not required.

---

### Step 7: Do not remove correlated features as a default step

### Why not?
Because trees are much less sensitive to multicollinearity.
A tree may choose one correlated feature and ignore another.

### Why not spend effort on this here?
Because it usually adds little benefit compared with the cost in complexity.

---

### Step 8: Keep most outliers unless they are clearly invalid

### Why?
Because trees are generally robust to outliers.
They create split rules and are not influenced by magnitude in the same way as linear models.

### Why not aggressively clip values?
Because valid extreme bookings may hold useful signal.

---

### Step 9: Apply tree-specific complexity control during modeling
This is not preprocessing in the narrow sense, but it is important:
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- pruning (`ccp_alpha`)

### Why mention this here?
Because the biggest risk for Decision Tree is **overfitting**, not scaling.

---

## 6.3 Final summary for Decision Tree

### Do these steps
- common base preprocessing
- imputation
- one-hot encoding
- retain valid outliers
- tune tree depth and leaf settings

### Do not scale
Because Decision Trees do not benefit meaningfully from scaling.

### Do not focus on skew correction by default
Because trees can handle skewed distributions.

### Do not prioritize multicollinearity removal
Because trees are not strongly harmed by correlated inputs.

---

# 7. Random Forest — Final preprocessing

## 7.1 Why Random Forest has similar preprocessing to Decision Tree

Random Forest is an **ensemble of decision trees**.
So it inherits many tree-model properties:
- little sensitivity to feature scaling
- good handling of non-linear relationships
- reduced overfitting compared with a single tree
- relatively low sensitivity to correlated inputs

---

## 7.2 Final preprocessing steps for Random Forest

### Step 1: Start from the base-cleaned dataset

### Why?
To preserve fairness and reproducibility.

---

### Step 2: Numerical imputation (if needed)
Use:
- median imputation

### Why do this?
Because Random Forest in sklearn still expects complete numeric input.

---

### Step 3: Categorical imputation (if needed)
Use:
- most frequent category or placeholder

### Why do this?
Because encoded categorical variables cannot contain missing values.

---

### Step 4: One-hot encode categorical variables
Use:
- `OneHotEncoder(handle_unknown='ignore')`

### Why do this?
Because sklearn Random Forest requires numeric input.
One-hot encoding is a consistent and safe way to represent nominal categories.

### Why not use ordinal/label encoding?
Because it may inject false ordering into purely nominal variables.

---

### Step 5: Do not scale numeric variables

### Why not?
Because Random Forest is tree-based and does not depend on distance or coefficient magnitude in the way linear models do.

### Why is scaling not useful here?
Because feature thresholds, not normalized distances, drive the learning process.

---

### Step 6: Do not make skew reduction a required step

### Why not?
Because Random Forest is usually robust to skewed feature distributions.

### When could skew handling still help?
Only in special cases, but normally the gain is limited.

---

### Step 7: Do not remove correlated features by default

### Why not?
Because Random Forest randomly samples features at splits and can handle redundancy better than a single linear model.

### Why keep correlated features?
Because some may still add useful predictive signal across different trees.

---

### Step 8: Keep most outliers unless clearly invalid

### Why?
Because Random Forest is robust to outliers.
Extreme values usually affect only some split choices and do not distort the whole ensemble strongly.

### Why not aggressively cap or remove them?
Because that may remove genuine booking behavior.

---

### Step 9: Use forest-specific regularization via hyperparameters
Important parameters include:
- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`
- `class_weight`

### Why mention this?
Because Random Forest performance is improved more through model tuning than through heavy preprocessing.

---

## 7.3 Final summary for Random Forest

### Do these steps
- common base preprocessing
- imputation
- one-hot encoding
- keep valid outliers
- tune ensemble hyperparameters

### Do not scale
Because Random Forest does not need scaled features.

### Do not prioritize skew correction
Because the model is robust to skew.

### Do not prioritize collinearity removal
Because the model is robust to redundancy.

---

# 8. Gradient Boosting — Final preprocessing

## 8.1 Why Gradient Boosting sits between Logistic Regression and other tree models

Gradient Boosting is also tree-based, so it does **not** need scaling like Logistic Regression.
However, it can be a bit more sensitive than Random Forest to:
- noisy extreme values
- difficult distributions
- parameter choices

So its preprocessing is still lighter than Logistic Regression, but slightly more careful than a basic tree setup.

---

## 8.2 Final preprocessing steps for Gradient Boosting

### Step 1: Start from the base-cleaned dataset

### Why?
To ensure fair comparison with the other models.

---

### Step 2: Numerical imputation (if needed)
Use:
- median imputation

### Why do this?
Because the sklearn pipeline expects complete numeric values.
Median remains robust.

---

### Step 3: Categorical imputation (if needed)
Use:
- most frequent category or placeholder

### Why do this?
Because encoding requires complete values.

---

### Step 4: One-hot encode categorical variables
Use:
- `OneHotEncoder(handle_unknown='ignore')`

### Why do this?
Because Gradient Boosting in sklearn still needs numeric inputs after transformation.

### Why not use raw text categories?
Because the model cannot work directly on strings.

---

### Step 5: Do not scale numeric variables

### Why not?
Because boosting with decision trees is still threshold-based and not scale-sensitive in the way linear models are.

### Why avoid unnecessary scaling?
Because it adds complexity without clear predictive benefit.

---

### Step 6: Consider mild skew handling for very skewed numeric features
Possible transformations:
- `lead_time`
- `adr`
- `days_in_waiting_list`

### Why do this?
Because while tree boosting does not require normality, reducing severe skew can sometimes create cleaner splits and more stable boosting behavior.

### Why only mild handling?
Because the model is still tree-based, so heavy transformation is usually unnecessary.

### Why not make this mandatory?
Because it should depend on actual validation performance.

---

### Step 7: Consider light outlier clipping for extreme numeric values
Especially for highly extreme `adr` values.

### Why do this?
Because boosting sequentially learns from errors, and very extreme observations can sometimes influence the learning path more than in bagging methods like Random Forest.

### Why not aggressively remove outliers?
Because many extreme bookings may still be valid.
Only light control should be considered.

---

### Step 8: Do not prioritize multicollinearity removal

### Why not?
Because tree-based boosting is not strongly affected by correlated predictors in the same way as Logistic Regression.

---

### Step 9: Focus strongly on hyperparameter tuning
Important parameters include:
- `n_estimators`
- `learning_rate`
- `max_depth`
- `subsample`
- `min_samples_leaf`

### Why mention this?
Because boosting performance depends heavily on tuning and can overfit if poorly configured.

---

## 8.3 Final summary for Gradient Boosting

### Do these steps
- common base preprocessing
- imputation
- one-hot encoding
- no scaling
- optional mild skew handling
- optional light outlier clipping
- careful hyperparameter tuning

### Do not scale
Because tree boosting does not require it.

### Do not prioritize multicollinearity removal
Because the model is relatively robust to it.

### Do not aggressively trim data
Because valid extremes may be informative.

---

# 9. Comparison table — final preprocessing by model

| Step | Logistic Regression | Decision Tree | Random Forest | Gradient Boosting |
|---|---|---|---|---|
| Base cleaning | Yes | Yes | Yes | Yes |
| Leakage removal | Yes | Yes | Yes | Yes |
| Missing value handling | Yes | Yes | Yes | Yes |
| Feature engineering | Yes | Yes | Yes | Yes |
| One-hot encoding | Yes | Yes | Yes | Yes |
| Numeric scaling | Yes | No | No | No |
| Skew correction | Recommended | Usually not needed | Usually not needed | Mildly helpful |
| Correlation check | Recommended | Not important | Not important | Not important |
| Outlier clipping | Mildly helpful | Only invalid values | Only invalid values | Light clipping may help |
| Main risk | unstable coefficients / scale sensitivity | overfitting | complexity / computation | overfitting / tuning sensitivity |

---

# 10. Best wording for the report

You can write the preprocessing section like this:

> A common preprocessing pipeline was first applied to the Hotel Booking Demand dataset to ensure fair comparison across all four supervised learning algorithms. This included data auditing, duplicate removal, anomaly inspection, target leakage removal, missing value handling, and feature engineering. After the shared preprocessing stage, model-specific final preprocessing was applied. Logistic Regression required feature scaling and benefited from stronger control of skewness and extreme values due to its linear nature. In contrast, the tree-based models did not require scaling. Decision Tree and Random Forest relied mainly on encoding and imputation, while Gradient Boosting additionally benefited from mild handling of highly skewed or extreme numerical variables.

---

# 11. Final conclusion

The correct strategy for this assignment is **not** to use exactly the same final preprocessing for all four models.

Instead:
- use the **same base preprocessing** for fairness
- use **different final preprocessing** depending on the model’s mathematical behavior

This gives a stronger academic comparison because it shows:
- understanding of the dataset
- understanding of preprocessing
- understanding of the algorithm differences
- better justification in the report and viva

---

# 12. Practical implementation guidance

## Shared base preprocessing should include
- dataset loading
- dataset profiling
- anomaly inspection
- duplicate handling
- leakage removal
- missing value treatment
- feature engineering
- feature grouping
- train/test split

## Logistic Regression final pipeline
- impute
- one-hot encode
- scale numeric features
- optionally transform skewed variables
- optionally clip extreme numeric outliers

## Decision Tree final pipeline
- impute
- one-hot encode
- no scaling
- no mandatory skew handling
- no mandatory correlation removal

## Random Forest final pipeline
- impute
- one-hot encode
- no scaling
- keep valid outliers
- focus on hyperparameter tuning

## Gradient Boosting final pipeline
- impute
- one-hot encode
- no scaling
- optionally handle severe skew lightly
- optionally clip extreme numeric values lightly
- focus on careful tuning
