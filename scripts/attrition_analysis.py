"""
=============================================================================
Predicting Employee Attrition: An HR Analytics Approach
Using Work-Life Balance, Job Satisfaction, and Overtime Data
=============================================================================

Authors : [Author Name(s)]
Dataset : IBM HR Analytics Employee Attrition Dataset
          Source: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
Paper   : [IEEE Conference Paper Title + DOI when available]

Theoretical Grounding:
  - Herzberg's Two-Factor Theory (1968)
  - Job Demands-Resources (JD-R) Model

Pipeline:
  1. Data Loading & Inspection
  2. Preprocessing
  3. Descriptive Statistics
  4. Inferential Tests  (Chi-square + Pearson Correlation)
  5. K-Means Clustering (employee risk segmentation)
  6. Logistic Regression
  7. Random Forest Classification
  8. Kaplan-Meier Survival Analysis
  9. Visualisations (reproduce all paper figures)

Requirements:
  pip install pandas numpy matplotlib seaborn scikit-learn lifelines scipy
=============================================================================
"""

# ---------------------------------------------------------------------------
# 0. IMPORTS
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import chi2_contingency, pearsonr

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             classification_report, silhouette_score)

from lifelines import KaplanMeierFitter

import warnings
warnings.filterwarnings("ignore")

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------------------
# 1. DATA LOADING & INSPECTION
# ---------------------------------------------------------------------------

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the IBM HR Analytics dataset.
    Expected columns (among others):
      Attrition, WorkLifeBalance, JobSatisfaction, OverTime, YearsAtCompany
    """
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumn names:\n{df.columns.tolist()}")
    print(f"\nAttrition value counts:\n{df['Attrition'].value_counts()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum().sum()} total")
    return df


# ---------------------------------------------------------------------------
# 2. PREPROCESSING
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Steps (as described in Section III of the paper):
      - Encode binary string columns to 0/1
      - Check / handle missing values (none found in IBM dataset)
      - Create numeric attrition flag for modelling
    """
    df = df.copy()

    # --- 2a. Encode Attrition: 'Yes' -> 1, 'No' -> 0
    df["Attrition_bin"] = df["Attrition"].map({"Yes": 1, "No": 0})

    # --- 2b. Encode OverTime: 'Yes' -> 1, 'No' -> 0
    df["OverTime_bin"] = df["OverTime"].map({"Yes": 1, "No": 0})

    # --- 2c. Missing value check
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("WARNING: Missing values detected:")
        print(missing[missing > 0])
        # Median imputation for ordinal variables
        for col in ["WorkLifeBalance", "JobSatisfaction"]:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
    else:
        print("No missing values found — no imputation required.")

    # --- 2d. Drop IBM dataset constant columns (if present)
    constant_cols = [c for c in df.columns if df[c].nunique() == 1]
    if constant_cols:
        print(f"Dropping constant columns: {constant_cols}")
        df.drop(columns=constant_cols, inplace=True)

    print(f"\nPreprocessing complete. Shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 3. DESCRIPTIVE STATISTICS
# ---------------------------------------------------------------------------

def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and display descriptive statistics for key variables
    as reported in Table 2 of the paper.
    """
    cols = ["WorkLifeBalance", "JobSatisfaction",
            "Attrition_bin", "OverTime_bin"]
    stats_df = df[cols].agg(["mean", "std", "min", "max"]).T
    stats_df.columns = ["Mean", "Std Dev", "Min", "Max"]
    stats_df = stats_df.round(3)

    print("\n===== TABLE 2: Descriptive Statistics =====")
    print(stats_df.to_string())

    # Attrition rate
    attrition_rate = df["Attrition_bin"].mean() * 100
    overtime_pct   = df["OverTime_bin"].mean() * 100
    print(f"\nOverall attrition rate : {attrition_rate:.1f}%")
    print(f"Overtime prevalence    : {overtime_pct:.1f}%")

    return stats_df


# ---------------------------------------------------------------------------
# 4. INFERENTIAL TESTS
# ---------------------------------------------------------------------------

def pearson_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlations among key variables.
    Reproduces Table 3 of the paper.
    """
    cols = ["Attrition_bin", "WorkLifeBalance",
            "JobSatisfaction", "OverTime_bin"]
    corr_matrix = df[cols].corr(method="pearson")

    print("\n===== TABLE 3: Pearson Correlation Matrix =====")
    print(corr_matrix.round(3).to_string())

    # Significance tests
    print("\n----- Pairwise significance (p-values) -----")
    for c1 in cols:
        for c2 in cols:
            if c1 < c2:
                r, p = pearsonr(df[c1].dropna(), df[c2].dropna())
                sig = "***" if p < 0.001 else ("**" if p < 0.01
                      else ("*" if p < 0.05 else "ns"))
                print(f"  {c1} vs {c2}: r={r:.3f}, p={p:.4f} {sig}")
    return corr_matrix


def chi_square_test(df: pd.DataFrame,
                    predictor: str,
                    outcome: str = "Attrition_bin",
                    label: str = "") -> None:
    """
    Chi-square test of independence between a categorical predictor
    and the binary attrition outcome.
    Reproduces Tables 4 and 5 of the paper.
    """
    ct = pd.crosstab(df[predictor], df[outcome])
    chi2, p, dof, expected = chi2_contingency(ct)

    print(f"\n===== Chi-Square: {label or predictor} vs Attrition =====")
    print(ct)
    print(f"\n  chi² = {chi2:.2f},  p = {p:.4f},  df = {dof}")
    if p < 0.001:
        print("  Result: Highly significant (p < 0.001) ***")
    elif p < 0.05:
        print("  Result: Significant (p < 0.05) *")
    else:
        print("  Result: Not significant")

    # Attrition rates per group
    rate = df.groupby(predictor)["Attrition_bin"].mean() * 100
    print(f"\n  Attrition rate by {predictor}:\n{rate.round(1)}")


# ---------------------------------------------------------------------------
# 5. K-MEANS CLUSTERING
# ---------------------------------------------------------------------------

def kmeans_clustering(df: pd.DataFrame,
                      n_clusters: int = 3) -> pd.DataFrame:
    """
    Segment employees into risk groups based on
    WorkLifeBalance, JobSatisfaction, and OverTime_bin.
    Features are z-score standardised before clustering.
    Silhouette score is reported to validate cluster quality.
    Reproduces Table 6 of the paper.
    """
    features = ["WorkLifeBalance", "JobSatisfaction", "OverTime_bin"]
    X = df[features].copy()

    # Z-score normalisation (mean=0, std=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow check (optional — for transparency)
    inertias = []
    for k in range(2, 8):
        km_tmp = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km_tmp.fit(X_scaled)
        inertias.append(km_tmp.inertia_)

    # Fit final model
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    df = df.copy()
    df["Cluster"] = km.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, df["Cluster"])
    print(f"\n===== K-Means Clustering (k={n_clusters}) =====")
    print(f"  Silhouette score: {sil:.2f}")

    # Cluster profile (TABLE 6)
    profile = df.groupby("Cluster").agg(
        WorkLifeBalance=("WorkLifeBalance", "mean"),
        JobSatisfaction=("JobSatisfaction", "mean"),
        OverTime_rate=("OverTime_bin", "mean"),
        Attrition_rate=("Attrition_bin", "mean"),
        Count=("Attrition_bin", "count")
    ).round(3)
    profile["OverTime_rate"]  = (profile["OverTime_rate"]  * 100).round(1)
    profile["Attrition_rate"] = (profile["Attrition_rate"] * 100).round(1)

    print("\n===== TABLE 6: Cluster Profiles =====")
    print(profile.to_string())

    return df


# ---------------------------------------------------------------------------
# 6. LOGISTIC REGRESSION
# ---------------------------------------------------------------------------

def logistic_regression(df: pd.DataFrame) -> LogisticRegression:
    """
    Train a Logistic Regression model to predict attrition.
    Features : WorkLifeBalance, JobSatisfaction, OverTime_bin
    Split     : 80% train / 20% test (stratified)
    Reproduces Table 7 of the paper.
    """
    features = ["WorkLifeBalance", "JobSatisfaction", "OverTime_bin"]
    X = df[features]
    y = df["Attrition_bin"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)

    print("\n===== Logistic Regression =====")
    print(f"  Accuracy  : {acc*100:.1f}%")
    print(f"  Precision : {prec*100:.1f}%")
    print(f"  Recall    : {rec*100:.1f}%")

    # Coefficients and Odds Ratios (TABLE 7)
    coef_df = pd.DataFrame({
        "Feature"    : features,
        "Coefficient": model.coef_[0].round(3),
        "Odds Ratio" : np.exp(model.coef_[0]).round(3)
    })
    print("\n===== TABLE 7: Logistic Regression Coefficients =====")
    print(coef_df.to_string(index=False))

    return model


# ---------------------------------------------------------------------------
# 7. RANDOM FOREST
# ---------------------------------------------------------------------------

def random_forest(df: pd.DataFrame) -> RandomForestClassifier:
    """
    Train a Random Forest classifier to predict attrition and
    rank feature importances.
    Split: 80% train / 20% test (stratified)
    Reproduces Table 8 of the paper.
    """
    features = ["WorkLifeBalance", "JobSatisfaction", "OverTime_bin"]
    X = df[features]
    y = df["Attrition_bin"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        class_weight="balanced"   # handles attrition class imbalance
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)

    print("\n===== Random Forest =====")
    print(f"  Accuracy  : {acc*100:.1f}%")
    print(f"  Precision : {prec*100:.1f}%")
    print(f"  Recall    : {rec*100:.1f}%")
    print(f"\n{classification_report(y_test, y_pred, zero_division=0)}")

    # Feature importances (TABLE 8)
    fi = pd.DataFrame({
        "Feature"   : features,
        "Importance": (rf.feature_importances_ * 100).round(1)
    }).sort_values("Importance", ascending=False)

    print("===== TABLE 8: Feature Importances =====")
    print(fi.to_string(index=False))

    return rf


# ---------------------------------------------------------------------------
# 8. KAPLAN-MEIER SURVIVAL ANALYSIS
# ---------------------------------------------------------------------------

def survival_analysis(df: pd.DataFrame) -> None:
    """
    Kaplan-Meier estimator to examine retention probability over tenure.
    Duration : YearsAtCompany
    Event    : Attrition_bin (1 = left the company)
    Reproduces Figure 6 of the paper.
    """
    kmf = KaplanMeierFitter()
    T = df["YearsAtCompany"]
    E = df["Attrition_bin"]

    kmf.fit(T, event_observed=E, label="All Employees")

    # Print survival probabilities at key timepoints (paper reports Y1,Y3,Y5)
    print("\n===== Kaplan-Meier Survival Estimates =====")
    for year in [1, 3, 5]:
        prob = kmf.predict(year)
        print(f"  Retention at Year {year}: {prob*100:.1f}%  "
              f"(Attrition: {(1-prob)*100:.1f}%)")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    kmf.plot_survival_function(ax=ax, ci_show=True, color="#2c7bb6")
    ax.set_title("Kaplan-Meier Survival Curve — Employee Retention Over Tenure",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Years at Company", fontsize=10)
    ax.set_ylabel("Survival Probability (Retention)", fontsize=10)
    ax.set_ylim(0, 1)
    ax.axvline(x=3, color="red", linestyle="--", alpha=0.6,
               label="Year 3 — peak attrition window")
    ax.legend()
    plt.tight_layout()
    plt.savefig("figures/survival_analysis_reproduced.png", dpi=150)
    plt.show()
    print("  Figure saved: figures/survival_analysis_reproduced.png")


# ---------------------------------------------------------------------------
# 9. VISUALISATIONS (reproduce paper figures)
# ---------------------------------------------------------------------------

def plot_attrition_by_overtime(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ct = df.groupby("OverTime")["Attrition"].value_counts(normalize=True
         ).unstack().fillna(0) * 100
    ct.plot(kind="bar", ax=ax, color=["#4393c3", "#d6604d"], edgecolor="white")
    ax.set_title("Attrition Rate by Overtime Status", fontweight="bold")
    ax.set_xlabel("Overtime")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticklabels(["No Overtime", "Overtime"], rotation=0)
    ax.legend(["No Attrition", "Attrition"])
    plt.tight_layout()
    plt.savefig("figures/attrition_by_overtime_reproduced.png", dpi=150)
    plt.show()


def plot_attrition_by_wlb(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    rate = df.groupby("WorkLifeBalance")["Attrition_bin"].mean() * 100
    rate.plot(kind="bar", ax=ax, color="#4393c3", edgecolor="white")
    ax.set_title("Attrition Rate by Work-Life Balance Rating", fontweight="bold")
    ax.set_xlabel("Work-Life Balance (1=Low, 4=High)")
    ax.set_ylabel("Attrition Rate (%)")
    ax.set_xticklabels(rate.index, rotation=0)
    plt.tight_layout()
    plt.savefig("figures/attrition_by_wlb_reproduced.png", dpi=150)
    plt.show()


def plot_attrition_by_satisfaction(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    rate = df.groupby("JobSatisfaction")["Attrition_bin"].mean() * 100
    rate.plot(kind="bar", ax=ax, color="#4393c3", edgecolor="white")
    ax.set_title("Attrition Rate by Job Satisfaction", fontweight="bold")
    ax.set_xlabel("Job Satisfaction (1=Very Dissatisfied, 4=Very Satisfied)")
    ax.set_ylabel("Attrition Rate (%)")
    ax.set_xticklabels(rate.index, rotation=0)
    plt.tight_layout()
    plt.savefig("figures/attrition_by_satisfaction_reproduced.png", dpi=150)
    plt.show()


def plot_clusters(df: pd.DataFrame) -> None:
    if "Cluster" not in df.columns:
        print("Run kmeans_clustering() first.")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = {0: "#4393c3", 1: "#d6604d", 2: "#74c476"}
    for cluster, group in df.groupby("Cluster"):
        ax.scatter(group["WorkLifeBalance"],
                   group["JobSatisfaction"],
                   c=palette[cluster],
                   label=f"Cluster {cluster}",
                   alpha=0.5, edgecolors="none", s=40)
    ax.set_title("Employee Clusters: WLB vs Job Satisfaction", fontweight="bold")
    ax.set_xlabel("Work-Life Balance")
    ax.set_ylabel("Job Satisfaction")
    ax.legend()
    plt.tight_layout()
    plt.savefig("figures/employee_clusters_reproduced.png", dpi=150)
    plt.show()


def plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5)
    ax.set_title("Pearson Correlation Matrix", fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/correlation_heatmap_reproduced.png", dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def main():
    import os
    os.makedirs("figures", exist_ok=True)

    # ---- Load ----
    filepath = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    df = load_data(filepath)

    # ---- Preprocess ----
    df = preprocess(df)

    # ---- Descriptive Stats ----
    descriptive_stats(df)

    # ---- Correlations ----
    corr = pearson_correlation_matrix(df)
    plot_correlation_heatmap(corr)

    # ---- Chi-Square Tests ----
    chi_square_test(df, "OverTime_bin", label="Overtime")
    chi_square_test(df, "WorkLifeBalance", label="Work-Life Balance")
    chi_square_test(df, "JobSatisfaction", label="Job Satisfaction")

    # ---- Visualisations ----
    plot_attrition_by_overtime(df)
    plot_attrition_by_wlb(df)
    plot_attrition_by_satisfaction(df)

    # ---- Clustering ----
    df = kmeans_clustering(df, n_clusters=3)
    plot_clusters(df)

    # ---- Logistic Regression ----
    logistic_regression(df)

    # ---- Random Forest ----
    random_forest(df)

    # ---- Survival Analysis ----
    survival_analysis(df)

    print("\n===== Pipeline complete. All results reproduced. =====")


if __name__ == "__main__":
    main()
