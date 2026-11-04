# Bank-Marketing-Success-Prediction-Using-Machine-Learning
This project focuses on predicting whether a bank customer will subscribe to a term deposit using Machine Learning classification algorithms.
"""
INT234 - CA2 Predictive Analytics
Bank Marketing Success Prediction: Comparison of Five Classifiers

Place this file next to the Data folder containing bank-full.csv.
Run: python main.py
"""

from pathlib import Path
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")  # Save charts as PNG; no GUI window needed.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------- 1. PATHS AND SETTINGS --------------------
ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "Data" / "bank-full.csv"
EDA_DIR = ROOT / "outputs" / "eda"
MODEL_DIR = ROOT / "outputs" / "models"
EDA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42


def heading(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(EDA_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close()


# -------------------- 2. LOAD AND INSPECT DATA --------------------
heading("STEP 1: LOAD AND EXPLORE DATA")
if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset nahi mila: {DATA_FILE}\n"
        "Check karo: Rocky Project/Data/bank-full.csv"
    )

df = pd.read_csv(DATA_FILE, sep=";")
df.columns = df.columns.str.strip()
if "y" not in df.columns:
    raise ValueError("Target column 'y' nahi mili. bank-full.csv check karo.")

print("Original dataset shape:", df.shape)
print("\nFirst 5 rows:\n", df.head().to_string())
print("\nColumn data types:\n", df.dtypes.to_string())
print("\nActual null values:\n", df.isna().sum().to_string())
print("\n'unknown' counts:\n", (df == "unknown").sum().to_string())
print("\nTarget counts:\n", df["y"].value_counts().to_string())

original_rows = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"\nExact duplicate rows removed: {original_rows - len(df)}")

if not set(df["y"].unique()).issubset({"yes", "no"}):
    raise ValueError("Target y mein unexpected values hain; yes/no expected hai.")

df["target_binary"] = df["y"].map({"no": 0, "yes": 1}).astype(int)
print(f"Subscription rate: {df['target_binary'].mean():.2%}")

# -------------------- 3. EDA + VISUALIZATIONS --------------------
heading("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")

df.describe(include="all").T.to_csv(EDA_DIR / "data_summary.csv")
df.isna().sum().to_csv(EDA_DIR / "null_counts.csv", header=["null_count"])
(df == "unknown").sum().to_csv(EDA_DIR / "unknown_counts.csv", header=["unknown_count"])

# Chart 1: Target distribution
counts = df["y"].value_counts().reindex(["no", "yes"], fill_value=0)
plt.figure(figsize=(6, 4))
plt.bar(counts.index, counts.values, color=["#4563a5", "#e6a34b"])
plt.title("Term Deposit Subscription: Yes vs No")
plt.xlabel("Subscribed?")
plt.ylabel("Customers")
for i, value in enumerate(counts.values):
    plt.text(i, value, str(value), ha="center", va="bottom")
save_plot("01_target_distribution.png")

# Chart 2: Customer age distribution
plt.figure(figsize=(7, 4))
plt.hist(df["age"], bins=25, edgecolor="white", color="#4563a5")
plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Customers")
save_plot("02_age_distribution.png")

# Chart 3: Subscription rate by occupation
job_rates = df.groupby("job")["target_binary"].mean().sort_values()
job_rates.to_csv(EDA_DIR / "subscription_rate_by_job.csv", header=["subscription_rate"])
plt.figure(figsize=(9, 5))
plt.barh(job_rates.index, job_rates.values * 100, color="#429c92")
plt.title("Subscription Rate by Job")
plt.xlabel("Subscribed (%)")
save_plot("03_subscription_by_job.png")

# Chart 4: Numeric correlation with target; exclude leakage feature 'duration'.
correlation_df = df.drop(columns=["duration"], errors="ignore")
correlations = (
    correlation_df.select_dtypes(include="number")
    .corr()["target_binary"]
    .drop("target_binary")
    .sort_values()
)
correlations.to_csv(EDA_DIR / "numeric_target_correlations.csv", header=["correlation"])
plt.figure(figsize=(8, 5))
plt.barh(correlations.index, correlations.values, color="#7487b8")
plt.axvline(0, color="black", lw=0.7)
plt.title("Numeric Feature Correlations with Subscription")
plt.xlabel("Pearson correlation (not causation)")
save_plot("04_numeric_correlations.png")

# Chart 5: Monthly conversion rates
month_order = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
monthly_rates = df.groupby("month")["target_binary"].mean().reindex(month_order)
monthly_rates.to_csv(EDA_DIR / "monthly_subscription_rates.csv", header=["subscription_rate"])
plt.figure(figsize=(9, 4))
plt.bar(monthly_rates.index, monthly_rates.values * 100, color="#e6a34b")
plt.title("Subscription Rate by Contact Month")
plt.xlabel("Month")
plt.ylabel("Subscribed (%)")
save_plot("05_monthly_subscription.png")

# -------------------- 4. STATISTICAL ANALYSIS --------------------
heading("STEP 3: STATISTICAL ANALYSIS")

# Chi-square: Is job category associated with subscription outcome?
contingency = pd.crosstab(df["job"], df["y"])
chi2, chi_p, _, _ = chi2_contingency(contingency)
cramers_v = np.sqrt(chi2 / (contingency.values.sum() * (min(contingency.shape) - 1)))
print(f"Chi-square (job vs target): statistic={chi2:.3f}, p-value={chi_p:.6g}")
print(f"Cramer's V (association strength): {cramers_v:.4f}")

# Mann-Whitney U: Do age distributions differ between subscription groups?
ages_yes = df.loc[df["target_binary"] == 1, "age"]
ages_no = df.loc[df["target_binary"] == 0, "age"]
u_stat, age_p = mannwhitneyu(ages_yes, ages_no, alternative="two-sided")
print(f"Mann-Whitney U (age by target): statistic={u_stat:.1f}, p-value={age_p:.6g}")
print("Note: Statistical association does NOT establish causation.")

pd.DataFrame([
    {"test": "Chi-square: job vs target", "statistic": chi2, "p_value": chi_p, "effect_size_cramers_v": cramers_v},
    {"test": "Mann-Whitney U: age vs target", "statistic": u_stat, "p_value": age_p, "effect_size_cramers_v": np.nan},
]).to_csv(EDA_DIR / "statistical_tests.csv", index=False)

# -------------------- 5. CLEANING AND PREPROCESSING --------------------
heading("STEP 4: CLEANING + DATA SPLITTING")

# 'duration' is only known AFTER a call ends. Including it would leak
# information unavailable before the call; exclude it from prediction.
X = df.drop(columns=["y", "target_binary", "duration"], errors="ignore").copy()
y = df["target_binary"]

# In job/education, 'unknown' means missing category; impute using TRAIN only.
for col in ["job", "education"]:
    if col in X.columns:
        X[col] = X[col].replace("unknown", np.nan)

# 60% train, 20% validation, 20% untouched test; all stratified.
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25,
    random_state=RANDOM_STATE, stratify=y_train_val
)

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
print("Numeric columns:", num_cols)
print("Categorical columns:", cat_cols)
print(f"Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")

# The pipeline learns imputation/scaling/encoding only from training data.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols),
    ],
    sparse_threshold=0,
)

# -------------------- 6. FIVE ML MODELS --------------------
heading("STEP 5: TRAIN AND COMPARE FIVE ML MODELS")
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced"),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=11, weights="distance", n_jobs=-1),
    "Gaussian Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=9, min_samples_leaf=30, class_weight="balanced", random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=120, min_samples_leaf=5, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1),
}

results = []
for name, classifier in models.items():
    print(f"\nTraining {name} ...", flush=True)
    pipe = Pipeline([
        ("preprocess", clone(preprocessor)),
        ("model", clone(classifier)),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_val)
    prob = pipe.predict_proba(X_val)[:, 1]
    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_val, pred),
        "Precision": precision_score(y_val, pred, zero_division=0),
        "Recall": recall_score(y_val, pred, zero_division=0),
        "F1": f1_score(y_val, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_val, prob),
    }
    results.append(row)
    print(f"Validation: F1={row['F1']:.4f} | ROC-AUC={row['ROC_AUC']:.4f}")

comparison = pd.DataFrame(results).sort_values("F1", ascending=False).reset_index(drop=True)
comparison.to_csv(MODEL_DIR / "five_model_validation_comparison.csv", index=False)
print("\nFive-model validation comparison:\n")
print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# Chart 6: Models compared on VALIDATION set, not on final test set.
plot_df = comparison.set_index("Model")[["F1", "ROC_AUC"]]
ax = plot_df.plot(kind="bar", figsize=(10, 5), ylim=(0, 1), rot=20)
ax.set_title("Five Classifiers: Validation Performance")
ax.set_ylabel("Score")
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig(MODEL_DIR / "06_model_comparison.png", dpi=160, bbox_inches="tight")
plt.close()

# -------------------- 7. FINAL MODEL EVALUATION --------------------
heading("STEP 6: CHOOSE ON VALIDATION; TEST ONLY ONCE")

# Choose using F1 on validation. Do NOT select the model based on test scores.
best_name = comparison.iloc[0]["Model"]
print("Selected model (highest validation F1):", best_name)

# Refit the chosen algorithm on train + validation, then evaluate on test.
best_pipeline = Pipeline([
    ("preprocess", clone(preprocessor)),
    ("model", clone(models[best_name])),
])
best_pipeline.fit(X_train_val, y_train_val)

test_pred = best_pipeline.predict(X_test)
test_prob = best_pipeline.predict_proba(X_test)[:, 1]
metrics = {
    "Model": best_name,
    "Accuracy": accuracy_score(y_test, test_pred),
    "Precision": precision_score(y_test, test_pred, zero_division=0),
    "Recall": recall_score(y_test, test_pred, zero_division=0),
    "F1": f1_score(y_test, test_pred, zero_division=0),
    "ROC_AUC": roc_auc_score(y_test, test_prob),
}
pd.DataFrame([metrics]).to_csv(MODEL_DIR / "final_test_metrics.csv", index=False)
print("\nIndependent TEST scores:")
for metric, value in metrics.items():
    if metric != "Model":
        print(f"{metric:>10}: {value:.4f}")
print("\nClassification report:\n")
report = classification_report(y_test, test_pred, target_names=["no", "yes"], zero_division=0)
print(report)
(MODEL_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

# Chart 7: Confusion matrix
cm = confusion_matrix(y_test, test_pred)
ConfusionMatrixDisplay(cm, display_labels=["no", "yes"]).plot(cmap="Blues", values_format="d")
plt.title(f"Test Confusion Matrix: {best_name}")
plt.tight_layout()
plt.savefig(MODEL_DIR / "07_test_confusion_matrix.png", dpi=160, bbox_inches="tight")
plt.close()

# Chart 8: ROC curve
fpr, tpr, _ = roc_curve(y_test, test_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"{best_name} (AUC={metrics['ROC_AUC']:.3f})")
plt.plot([0, 1], [0, 1], "--", color="gray", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Final Test ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(MODEL_DIR / "08_test_roc_curve.png", dpi=160, bbox_inches="tight")
plt.close()

# Save the trained pipeline: preprocessing + selected model.
joblib.dump(best_pipeline, MODEL_DIR / "best_bank_marketing_model.joblib")

# Example: predict ONE customer from the test features (not for further tuning).
sample = X_test.iloc[[0]]
print("\nOne example prediction (test customer):")
print("Predicted subscription:", "yes" if best_pipeline.predict(sample)[0] == 1 else "no")
print(f"Estimated probability of yes: {best_pipeline.predict_proba(sample)[0, 1]:.2%}")
print("Actual outcome:", "yes" if y_test.iloc[0] == 1 else "no")

heading("PROJECT RUN COMPLETED")
print("EDA charts/reports saved in:", EDA_DIR)
print("Model charts/metrics saved in:", MODEL_DIR)
print("Paper ke results apne actual CSV output se lo; values invent mat karo.")
print("Limitations: historical Portuguese campaign data; correlation != causation;")
print("and deployment on new customers requires fresh validation.")
# Bank Marketing Success Prediction Using Machine Learning

### A Comparative Study of Five Classification Algorithms

**Course:** INT234 – Predictive Analytics  
**Assessment:** CA2 Project  
**Development environment:** Visual Studio Code (VS Code)  
**Language:** Python

## Project Overview

This project uses historical bank marketing campaign data to predict whether a customer will subscribe to a term deposit (`yes` or `no`). The goal is to compare five classification algorithms, understand the effects of an imbalanced target variable, and evaluate the selected model on a separate test set.

The complete workflow is implemented in **one Python file: `main.py`**. Running it loads the dataset, performs exploratory data analysis (EDA), generates visualizations, runs statistical tests, preprocesses the data, compares five classifiers, saves evaluation results, and demonstrates one prediction.

> **Research question:** Which of the five classification algorithms achieves the highest validation F1-score for term-deposit subscription prediction when the post-call `duration` feature is excluded?

## Dataset

- **Name:** Bank Marketing (bank-full.csv)
- **Source:** [UCI Machine Learning Repository – Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Download:** [Official UCI ZIP archive](https://archive.ics.uci.edu/static/public/222/bank+marketing.zip)
- **Records:** 45,211
- **Original columns:** 17 (16 input columns and target `y`)
- **Target:** `y` — `yes` (subscribed) / `no` (did not subscribe)
- **CSV delimiter:** Semicolon (`;`)

### Class Distribution

| Target | Customers | Share |
|---|---:|---:|
| `no` | 39,922 | 88.30% |
| `yes` | 5,289 | 11.70% |
| **Total** | **45,211** | **100.00%** |

**Why this matters:** A classifier predicting `no` for everyone would achieve approximately **88.30% accuracy**, but would fail to identify any subscribers. Therefore, accuracy alone is not a sufficient performance measure; precision, recall, F1-score, and ROC-AUC are also reported.

## Project Structure

```text
Rocky Project/
├── main.py                       # Entire Python implementation
├── README.md                     # Project documentation
├── Data/
│   └── bank-full.csv             # Downloaded UCI dataset
└── outputs/                      # Automatically created when main.py runs
    ├── eda/
    │   ├── data_summary.csv
    │   ├── null_counts.csv
    │   ├── unknown_counts.csv
    │   ├── subscription_rate_by_job.csv
    │   ├── numeric_target_correlations.csv
    │   ├── monthly_subscription_rates.csv
    │   ├── statistical_tests.csv
    │   ├── 01_target_distribution.png
    │   ├── 02_age_distribution.png
    │   ├── 03_subscription_by_job.png
    │   ├── 04_numeric_correlations.png
    │   └── 05_monthly_subscription.png
    └── models/
        ├── five_model_validation_comparison.csv
        ├── final_test_metrics.csv
        ├── classification_report.txt
        ├── 06_model_comparison.png
        ├── 07_test_confusion_matrix.png
        ├── 08_test_roc_curve.png
        └── best_bank_marketing_model.joblib
```

`outputs/` and its subfolders are created automatically; they do not need to exist beforehand.

## How to Run in VS Code (Windows)

1. Download the original dataset from UCI and extract `bank-full.csv` (the download may contain a second ZIP file named `bank.zip`).
2. Create/open a folder called `Rocky Project` in VS Code.
3. Place `main.py` and the `Data` folder inside it as shown above.
4. Open **Terminal → New Terminal** in VS Code.
5. Install the required libraries:

```powershell
py -m pip install pandas numpy matplotlib scipy scikit-learn joblib
```

6. Run the program:

```powershell
py main.py
```

The console displays the full workflow and evaluation metrics. Files are saved under `outputs/eda/` and `outputs/models/`.

**If the dataset is not found:** Verify that the exact path is `Rocky Project/Data/bank-full.csv`. Do not rename it to `bank.csv` or put it beside `main.py`.

## Methodology

### 1. Data Inspection and Cleaning

- Load the semicolon-separated CSV file using pandas.
- Inspect dimensions, data types, missing (`null`) values, and literal `unknown` categories.
- Remove exact duplicate records if found (zero were removed in the reported run).
- Convert the target from `yes`/`no` to `1`/`0`.
- Treat `unknown` values in `job` and `education` as missing before training. Other `unknown` labels are retained as categories in the current implementation.
- **Exclude `duration`** because it represents the duration of the call and is not available before that call finishes. Including it would make a pre-call prediction unrealistic.

### 2. Exploratory Data Analysis (EDA)

The program produces five charts:

1. Term-deposit subscription distribution
2. Customer age distribution
3. Subscription rates by job category
4. Numeric-feature correlations with the target (excluding `duration`)
5. Monthly subscription rates

It also exports dataset summaries and CSV reports for later research-paper tables and figures.

### 3. Statistical Analysis

- **Chi-square test:** Association between job category and subscription outcome.
- **Cramér's V:** Strength of that categorical association.
- **Mann–Whitney U test:** Comparison of age distributions between subscribers and non-subscribers.

Reported outputs:

| Test / Measure | Result |
|---|---:|
| Chi-square statistic (job vs. target) | 836.105 |
| Chi-square p-value | 3.33712e-172 |
| Cramér's V | 0.1360 |
| Mann–Whitney U statistic (age vs. target) | 103,915,109.5 |
| Mann–Whitney U p-value | 0.0628179 |

These statistics indicate associations or differences in sample distributions; **they do not prove causation**.

### 4. Preprocessing and Data Splitting

- **60% training / 20% validation / 20% test**, using stratified splits.
- **Numeric inputs:** Median imputation and standard scaling.
- **Categorical inputs:** Most-frequent imputation and one-hot encoding.
- Preprocessing is fitted only on training data during model comparison, avoiding leakage of validation/test preprocessing statistics into training.
- Five models are compared on the validation set using **F1-score** as the selection criterion.
- The selected algorithm is refitted on training + validation data and evaluated **once** on the held-out test set.

In the reported run, the split sizes were **27,126 training**, **9,042 validation**, and **9,043 test** rows.

### 5. Classification Models

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Gaussian Naive Bayes
4. Decision Tree
5. Random Forest

## Results

### Validation-Set Comparison

The following values are from the actual VS Code execution shown for this project, not hypothetical benchmark results.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **Random Forest** | **0.8647** | 0.4380 | 0.5510 | **0.4881** | **0.8011** |
| Decision Tree | 0.7960 | 0.3096 | 0.6049 | 0.4096 | 0.7722 |
| Gaussian Naive Bayes | 0.8421 | 0.3615 | 0.4565 | 0.4035 | 0.7501 |
| Logistic Regression | 0.7548 | 0.2666 | 0.6257 | 0.3739 | 0.7702 |
| K-Nearest Neighbors | 0.8900 | 0.5924 | 0.1909 | 0.2888 | 0.7394 |

**Selected model:** Random Forest (highest validation F1-score).

### Final Independent Test-Set Results: Random Forest

| Metric | Score |
|---|---:|
| Accuracy | 0.8643 (86.43%) |
| Precision (`yes`) | 0.4360 |
| Recall (`yes`) | 0.5444 |
| F1-score (`yes`) | 0.4842 |
| ROC-AUC | 0.8034 |

The test results are separate from the validation results used for model selection. The example prediction shown in the terminal is illustrative, not evidence of deployment performance.

## Key Findings

- The dataset is **imbalanced**: only 11.70% of customers subscribed.
- **Random Forest** achieved the highest validation F1-score among the five tested configurations.
- **KNN** had higher validation accuracy (89.00%) but much lower recall (19.09%) and F1-score (0.2888) for the `yes` class. This illustrates why accuracy can be misleading when the positive class is rare.
- Random Forest's positive-class **test recall was 54.44%**, meaning that it identified roughly 54 out of every 100 actual subscribers in the test set.
- There remains substantial room for improvement before a system like this should be used for real customer-targeting decisions.

## Limitations and Future Improvements

- The dataset describes historical **Portuguese** bank campaigns; outcomes may not generalize to other countries or newer campaigns.
- Some input fields might not be available at the exact moment an organization wants to make a prediction. Feature availability must be checked for the intended real-world workflow.
- `unknown` labels are not uniformly treated as missing; this is an explicit modelling choice worth testing.
- The comparison uses chosen model settings, **not exhaustive hyperparameter tuning**.
- Test-set results are estimates on this dataset, not guaranteed real-world accuracy.
- Future work could evaluate cross-validation, class-weight and decision-threshold strategies, calibration, and performance on a genuinely newer dataset.
- Model outputs should not be treated as causal explanations or as automatic decisions about customers.

## CA2 Submission Context

This repository covers the **implementation component** of INT234 CA2: dataset processing, EDA, statistical analysis, five classifiers, and evaluation.

The course instructions also require a **six-page IEEE-formatted paper**, project highlights on **GitHub and LinkedIn**, and a **viva**. Those materials are separate deliverables and should be prepared using the actual analysis outputs. Originality and similarity requirements should be checked using the institution's approved process; no similarity percentage is guaranteed by this README.

## Data Attribution

Dataset: **Bank Marketing**, UCI Machine Learning Repository.  
Official dataset page: https://archive.ics.uci.edu/dataset/222/bank+marketing

Please consult the dataset page for its preferred citation and applicable data license before publishing or redistributing the CSV file.
