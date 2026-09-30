#!/usr/bin/env python3
"""
E-Commerce Product Return Prediction - Model Training Pipeline
==============================================================
This script:
1. Loads the return dataset from `data/ecommerce_returns_dataset.csv`.
2. Preprocesses data with median imputation for missing values,
   scaling for numeric features, and one-hot encoding for categorical features.
3. Splits data using a stratified 80-20 train-test split.
4. Trains 5 candidate classification algorithms:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Decision Tree
   - Random Forest
   - Gradient Boosting
5. Evaluates each algorithm with accuracy, precision, recall, F1, confusion matrices,
   classification reports, and 5-fold stratified cross-validation.
6. Saves model benchmark metrics to `outputs/model_comparison.csv`.
7. Selects the champion model primarily based on Precision and CV stability,
   while factoring in F1-score.
8. Saves the final pipeline to `models/return_prediction_model.pkl`.
9. Extracts and saves feature importance to `outputs/feature_importance.csv`.
10. Prints a clear business & technical summary.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = "data/ecommerce_returns_dataset.csv"
MODEL_PATH = "models/return_prediction_model.pkl"
COMPARISON_PATH = "outputs/model_comparison.csv"
IMPORTANCE_PATH = "outputs/feature_importance.csv"


def prepare_data(data_path: str):
    """Load and validate dataset."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")

    # Exclude order_id from feature space
    feature_cols = [
        "product_category",
        "product_price_inr",
        "discount_percent",
        "delivery_duration_days",
        "customer_purchase_history",
        "payment_method",
    ]
    target_col = "returned"

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    print(
        f"Target distribution: Return rate = {y.mean() * 100:.2f}% "
        f"(1: {y.sum()}, 0: {(1 - y).sum()})"
    )
    return X, y


def build_preprocessor():
    """Create ColumnTransformer for numeric and categorical pipelines."""
    num_cols = [
        "product_price_inr",
        "discount_percent",
        "delivery_duration_days",
        "customer_purchase_history",
    ]
    cat_cols = ["product_category", "payment_method"]

    # Impute missing delivery_duration_days & customer_purchase_history with median
    # followed by StandardScaler for numerical columns
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # One-hot encode categorical columns
    cat_pipeline = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ]
    )
    return preprocessor


def get_feature_names(preprocessor, num_cols, cat_cols):
    """Extract clean feature names from fitted ColumnTransformer."""
    try:
        raw_names = preprocessor.get_feature_names_out()
        clean_names = [
            f.replace("num__", "").replace("cat__", "") for f in raw_names
        ]
        return clean_names
    except Exception:
        # Fallback if get_feature_names_out fails
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        cat_features = cat_encoder.get_feature_names_out(cat_cols)
        return list(num_cols) + list(cat_features)


def train_and_evaluate_models(X, y):
    """Train candidate classifiers, run 5-fold CV, and evaluate on test set."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # 80-20 Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training set: {len(X_train)} samples | Test set: {len(X_test)} samples")

    preprocessor = build_preprocessor()

    # Candidate Models
    candidate_models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
        "Random Forest": RandomForestClassifier(
            random_state=42, n_estimators=100, max_depth=6, min_samples_leaf=3
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42, n_estimators=100, learning_rate=0.08, max_depth=3
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    results = []
    trained_pipelines = {}

    print("\n" + "=" * 70)
    print("BEGINNING MODEL BENCHMARKING (5-Fold Stratified CV + Test Evaluation)")
    print("=" * 70)

    for name, model in candidate_models.items():
        print(f"\n--- Training: {name} ---")
        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", model),
            ]
        )

        # 5-fold Stratified Cross-Validation on training data
        cv_scores = cross_validate(
            pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
        )

        # Fit model on training split
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe

        # Evaluate on holdout test set
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(
            y_test,
            y_pred,
            target_names=["Not Returned (0)", "Returned (1)"],
            zero_division=0,
        )

        cv_prec_mean = cv_scores["test_precision"].mean()
        cv_prec_std = cv_scores["test_precision"].std()
        cv_f1_mean = cv_scores["test_f1"].mean()
        cv_acc_mean = cv_scores["test_accuracy"].mean()
        cv_rec_mean = cv_scores["test_recall"].mean()

        print(f"Test Accuracy : {acc:.4f}")
        print(f"Test Precision: {prec:.4f}")
        print(f"Test Recall   : {rec:.4f}")
        print(f"Test F1-Score : {f1:.4f}")
        print(f"CV Precision  : {cv_prec_mean:.4f} (+/- {cv_prec_std:.4f})")
        print(f"CV F1-Score   : {cv_f1_mean:.4f}")
        print("\nConfusion Matrix [Test]:")
        print(
            f"  TN: {cm[0, 0]:3d}  |  FP: {cm[0, 1]:3d}\n  FN: {cm[1, 0]:3d}  |  TP: {cm[1, 1]:3d}"
        )
        print("\nClassification Report [Test]:\n" + cr)

        # Selection score: Primary weight on CV precision and stability (low std),
        # with secondary balance for F1 to prevent degenerate models.
        # Selection Score = CV_Precision_Mean - (0.5 * CV_Precision_Std) + (0.3 * CV_F1_Mean)
        stability_penalty = 0.5 * cv_prec_std
        selection_score = cv_prec_mean - stability_penalty + (0.3 * cv_f1_mean)

        results.append(
            {
                "Model": name,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1_Score": round(f1, 4),
                "CV_Precision_Mean": round(cv_prec_mean, 4),
                "CV_Precision_Std": round(cv_prec_std, 4),
                "CV_F1_Mean": round(cv_f1_mean, 4),
                "CV_Accuracy_Mean": round(cv_acc_mean, 4),
                "CV_Recall_Mean": round(cv_rec_mean, 4),
                "Selection_Score": round(selection_score, 4),
            }
        )

    # Save model comparison
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values(
        by=["Selection_Score", "CV_Precision_Mean"], ascending=False
    )
    comparison_df.to_csv(COMPARISON_PATH, index=False)
    print(f"\nSaved model comparison metrics to {COMPARISON_PATH}")

    # Select best model
    best_row = comparison_df.iloc[0]
    best_name = best_row["Model"]
    best_pipeline = trained_pipelines[best_name]

    print("\n" + "=" * 70)
    print(f"SELECTED CHAMPION MODEL: {best_name}")
    print("=" * 70)
    print(f"Cross-Validation Precision : {best_row['CV_Precision_Mean']:.4f} (+/- {best_row['CV_Precision_Std']:.4f})")
    print(f"Cross-Validation F1-Score  : {best_row['CV_F1_Mean']:.4f}")
    print(f"Test Set Accuracy          : {best_row['Accuracy']:.4f}")
    print(f"Selection Composite Score  : {best_row['Selection_Score']:.4f}")

    # Rationale explanation
    rationale = (
        f"Selected {best_name} primarily based on high precision ({best_row['CV_Precision_Mean']:.1%}) "
        f"and strong cross-validation stability (std: {best_row['CV_Precision_Std']:.3f}), "
        f"while maintaining a reliable F1-score ({best_row['CV_F1_Mean']:.3f}). "
        f"In reverse logistics, precision is the paramount business metric: an intervention on a predicted "
        f"return (such as withholding discounts, applying friction, or initiating manual verification) "
        f"must not falsely penalize genuine shoppers. Minimizing false positives protects customer trust, "
        f"order conversion rates, and long-term brand equity."
    )

    # Attach model metadata to the pipeline object for seamless deserialization in Streamlit
    fitted_preprocessor = best_pipeline.named_steps["preprocessor"]
    feature_names = get_feature_names(
        fitted_preprocessor,
        [
            "product_price_inr",
            "discount_percent",
            "delivery_duration_days",
            "customer_purchase_history",
        ],
        ["product_category", "payment_method"],
    )

    best_pipeline.model_name_ = best_name
    best_pipeline.metrics_ = best_row.to_dict()
    best_pipeline.feature_names_ = feature_names
    best_pipeline.selection_rationale_ = rationale

    # Save pipeline
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"Saved selected pipeline to {MODEL_PATH}")

    # Feature Importance extraction
    classifier = best_pipeline.named_steps["classifier"]
    importance_df = None

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
                "direction": "Positive Impact on Split",
            }
        ).sort_values(by="importance", ascending=False)
    elif hasattr(classifier, "coef_"):
        coefficients = classifier.coef_[0]
        directions = [
            "Increases Return Risk (+)" if c > 0 else "Decreases Return Risk (-)"
            for c in coefficients
        ]
        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": np.abs(coefficients),
                "coefficient": coefficients,
                "direction": directions,
            }
        ).sort_values(by="importance", ascending=False)

    if importance_df is not None:
        importance_df.to_csv(IMPORTANCE_PATH, index=False)
        print(f"Saved feature importance analysis to {IMPORTANCE_PATH}")
        print("\nTop 5 Most Influential Features:")
        print(importance_df.head(5)[["feature", "importance", "direction"]])

    # Commercial Summary Print
    print("\n" + "=" * 70)
    print("COMMERCIAL & STRATEGIC RATIONALE SUMMARY")
    print("=" * 70)
    print(f"1. Selected Model: {best_name}")
    print(f"2. Commercial Precision Imperative:")
    print(
        "   - E-commerce return mitigation relies on targeted preventive actions (e.g. friction checks,\n"
        "     withholding dynamic discounts, or requiring verified payment methods).\n"
        "   - If precision is low (high False Positive Rate), legitimate customers who have no intention\n"
        "     of returning items are falsely flagged. This causes cart abandonment, degraded trust,\n"
        "     and direct revenue destruction.\n"
        "   - Prioritizing Precision ensures operational interventions are laser-focused on genuinely\n"
        "     risky orders, preserving loyalty among genuine shoppers."
    )
    print(f"3. Model Selection Decision:")
    print(f"   {rationale}")
    print("=" * 70)
    return best_pipeline, comparison_df, importance_df


def run_training_pipeline(data_path: str = DATA_PATH):
    """Convenience helper to load data, run benchmarks, and return trained artifacts."""
    X, y = prepare_data(data_path)
    return train_and_evaluate_models(X, y)


if __name__ == "__main__":
    run_training_pipeline(DATA_PATH)
