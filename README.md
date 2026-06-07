# 🚢 Titanic Survival Predictor
### Comparing 5 Classification Models with Full ML Pipeline on the Titanic Dataset

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

---

## Overview

This is a beginner-level supervised learning classification project built to understand how different classification models behave on the same dataset — not just to get a good score, but to understand *why* each model performs the way it does, and what hyperparameter tuning actually changes.

The dataset is the [Titanic Dataset](https://www.kaggle.com/competitions/titanic) (Kaggle). The goal: predict whether a passenger survived based on features like age, class, fare, and family size.

---

## What I Learned

- How to build a full sklearn `Pipeline` combining preprocessing and model in one object
- Why `ColumnTransformer` cleanly separates numeric scaling from categorical encoding
- How feature engineering (extracting Title, FamilySize, Deck) improves model signal
- Why Logistic Regression outperforms Random Forest on small datasets with linear patterns
- What hyperparameter tuning actually fixes — and when it doesn't help (Random Forest on 891 rows)
- Why a single train/test split can be misleading — cross-validation gives the honest score
- The difference between Accuracy and F1-score and when each matters

---

## Results

### Before Tuning

| Model | Accuracy |
|---|---|
| Logistic Regression | 0.8436 |
| SVM | 0.8380 |
| KNN | 0.8212 |
| Random Forest | 0.7933 |
| Decision Tree | 0.7654 |

### After GridSearchCV Tuning (5-fold cross-validation)

| Model | Accuracy | Change |
|---|---|---|
| Logistic Regression | **0.8436** | 0.0000 |
| SVM | **0.8436** | +0.0056 |
| Decision Tree | 0.8324 | +0.0670 ✅ |
| KNN | 0.7989 | -0.0223 |
| Random Forest | 0.7933 | 0.0000 |

> Logistic Regression and SVM tied for best. Decision Tree improved the most after controlling max_depth.

---

## Why Each Model Performed the Way It Did

**Logistic Regression — best performer**
Titanic survival has clear linear patterns: being female, 1st class, and paying a high fare strongly predicts survival. Logistic Regression captures these linear boundaries well. With clean feature engineering providing strong signals, it hit the ceiling of what this dataset allows.

**SVM — tied for best after tuning**
Strong on small, clean, well-preprocessed datasets. Finds the maximum margin boundary between survivors and non-survivors. Tuning `C` and `kernel` gave it a small edge over defaults.

**Decision Tree — most improved by tuning (+0.067)**
Default `max_depth=None` meant it grew fully and memorized the training data — textbook overfitting. GridSearch found a limited `max_depth` that stopped memorization. Biggest lesson in this project: controlling tree depth is critical on small datasets.

**Random Forest — no improvement from tuning**
The default parameters were already optimal for this dataset size. The real issue is that 891 rows is too small for ensemble methods to shine — Random Forest needs thousands of rows to build diverse, decorrelated trees. Not a tuning problem, a data size problem.

**KNN — slight drop after tuning**
The original 0.8212 score was on a single train/test split that was slightly lucky. Cross-validation during GridSearch gave a more honest score. The drop reflects the split being favorable by chance, not the model getting worse.

---

## Steps Taken

### 1. Feature Engineering
- Extracted `Title` from `Name` column (Mr, Mrs, Miss, Rare etc.)
- Created `FamilySize` = `SibSp` + `Parch` + 1
- Created `IsAlone` flag — FamilySize == 1
- Extracted `Deck` from first character of `Cabin`

### 2. Missing Value Handling
- `Age` → filled with median grouped by `Sex` and `Pclass` (smarter than global median)
- `Fare` → filled with overall median
- `Embarked` → filled with mode
- `Cabin` → converted to `Deck`, missing = `'U'` (unknown)

### 3. Preprocessing Pipeline
- `StandardScaler` on numeric features: `Age`, `SibSp`, `Parch`, `Fare`, `FamilySize`, `IsAlone`
- `OneHotEncoder` on categorical features: `Pclass`, `Sex`, `Embarked`, `Title`, `Deck`
- Combined with `ColumnTransformer` — numeric and categorical handled separately in one step
- Entire preprocessing + model wrapped in `Pipeline` — prevents data leakage

### 4. Train/Test Split
- 80/20 split with `stratify=y` — preserves survived/not survived ratio in both sets
- Critical on imbalanced classes (62% died, 38% survived)

### 5. Model Training & Tuning
- All 5 models trained via loop through Pipeline
- GridSearchCV with `cv=5` tuned:
  - Logistic Regression: `C`, `solver`
  - KNN: `n_neighbors`, `weights`, `metric`
  - Decision Tree: `max_depth`, `min_samples_split`, `criterion`
  - Random Forest: `n_estimators`, `max_depth`, `min_samples_split`
  - SVM: `C`, `kernel`, `gamma`
- `n_jobs=-1` used to parallelize search across all CPU cores

### 6. Evaluation
- Accuracy, F1-score, classification report, confusion matrix per model
- Before vs after tuning comparison table

---

## Key Technical Decisions

**Why `ColumnTransformer`?**
Different features need different preprocessing. Numeric features need scaling; categorical features need encoding. `ColumnTransformer` applies the right transformation to the right columns cleanly in one object.

**Why `stratify=y` in train/test split?**
Without stratification, a random split could put most survivors in training and few in test (or vice versa), making evaluation unreliable. Stratification guarantees the class ratio is preserved.

**Why `clf__` prefix in GridSearchCV params?**
Because the classifier lives inside a Pipeline named `clf`. GridSearchCV needs the prefix to know which Pipeline step each parameter belongs to. `clf__C` = the `C` parameter of the `clf` step.

**Why cross-validation instead of single split for tuning?**
A single split can be lucky or unlucky. 5-fold cross-validation trains and tests 5 times on different data splits and averages — a much more honest measure of real performance.

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/titanic-survival-predictor.git
cd titanic-survival-predictor

# 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# 3. Add the dataset
# Download Titanic-Dataset.csv from Kaggle and place in data/

# 4. Run
python main.py
```

---

## Dependencies

```
pandas >= 1.3
numpy >= 1.21
scikit-learn >= 1.0
matplotlib >= 3.4
seaborn >= 0.11
```

---
