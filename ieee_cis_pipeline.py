"""
IEEE-CIS Fraud Detection — Data Loading, Preprocessing & Baseline Model
=========================================================================

SETUP (run these in your terminal / shell, not inside Python):

1. Install Kaggle CLI and dependencies:
     pip install kaggle numpy pandas scikit-learn lightgbm shap

2. Get your Kaggle API credentials:
     - Go to kaggle.com -> Account -> Create New API Token
     - This downloads kaggle.json
     - Place it at ~/.kaggle/kaggle.json (Linux/Mac) or
       C:\\Users\\<you>\\.kaggle\\kaggle.json (Windows)
     - chmod 600 ~/.kaggle/kaggle.json

3. Join the competition (required once, via browser):
     https://www.kaggle.com/c/ieee-fraud-detection/rules
     (click "I Understand and Accept")

4. Download the data:
     kaggle competitions download -c ieee-fraud-detection -p ./data
     cd data && unzip ieee-fraud-detection.zip

   You should now have:
     data/train_transaction.csv
     data/train_identity.csv
     data/test_transaction.csv
     data/test_identity.csv

5. Run this script:
     python ieee_cis_pipeline.py --data_dir ./data

-------------------------------------------------------------------------
What this script does:
  1. Loads and merges the transaction + identity tables (they're split
     because not every transaction has identity/device data attached).
  2. Does basic feature engineering (time features, aggregates, NA counts).
  3. Handles categorical encoding.
  4. Trains a LightGBM baseline with a temporal train/validation split
     (fraud detection should NEVER use random splits — see note below).
  5. Evaluates with PR-AUC (not ROC-AUC — see note below).
  6. Computes SHAP values for the top flagged transactions — this is the
     bridge to the LLM investigation layer from the architecture we
     discussed: each SHAP explanation becomes structured input you hand
     to the LLM agent instead of raw feature vectors.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import lightgbm as lgb


# ---------------------------------------------------------------------
# 1. LOADING & MERGING
# ---------------------------------------------------------------------
def load_data(data_dir: str):
    """
    IEEE-CIS ships transaction and identity data as separate files joined
    on TransactionID. Only ~24% of transactions have identity data, so
    this must be a LEFT join on transaction (never inner — you'd drop
    76% of your data).
    """
    print("Loading transaction tables...")
    train_tx = pd.read_csv(os.path.join(data_dir, "train_transaction.csv"))
    test_tx = pd.read_csv(os.path.join(data_dir, "test_transaction.csv"))

    print("Loading identity tables...")
    train_id = pd.read_csv(os.path.join(data_dir, "train_identity.csv"))
    test_id = pd.read_csv(os.path.join(data_dir, "test_identity.csv"))

    # Known IEEE-CIS quirk: identity columns use underscores in the train
    # file (id_12) but hyphens in the test file (id-12). Standardize to
    # underscores so train/test have matching column names after merge.
    test_id = test_id.rename(columns=lambda c: c.replace("-", "_"))

    print("Merging on TransactionID (left join)...")
    train = train_tx.merge(train_id, on="TransactionID", how="left")
    test = test_tx.merge(test_id, on="TransactionID", how="left")

    print(f"Train shape: {train.shape}  |  Test shape: {test.shape}")
    print(f"Fraud rate: {train['isFraud'].mean():.4%}")
    return train, test


# ---------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TransactionDT is seconds from a reference point (not a real
    # timestamp) — still useful for deriving cyclical time features.
    df["day"] = df["TransactionDT"] // (24 * 60 * 60)
    df["hour"] = (df["TransactionDT"] // 3600) % 24
    df["weekday"] = df["day"] % 7

    # Card + address grouping approximates a "user/account" proxy,
    # since IEEE-CIS has no direct user ID. This is what most public
    # solutions use to build velocity-style aggregate features.
    df["card_addr_id"] = (
        df["card1"].astype(str) + "_" + df["addr1"].astype(str)
    )

    # Amount-based aggregate: deviation from the entity's mean spend —
    # a core fraud signal ("amount 8x above average" from our example).
    group_mean = df.groupby("card_addr_id")["TransactionAmt"].transform("mean")
    group_std = df.groupby("card_addr_id")["TransactionAmt"].transform("std")
    df["amt_to_mean_ratio"] = df["TransactionAmt"] / (group_mean + 1e-3)
    df["amt_zscore"] = (df["TransactionAmt"] - group_mean) / (group_std + 1e-3)

    # Missingness itself is signal in this dataset (device/identity
    # fields are often null for higher-risk transactions).
    id_cols = [c for c in df.columns if c.startswith("id_")]
    if id_cols:
        df["identity_na_count"] = df[id_cols].isna().sum(axis=1)

    # Email domain grouping (raw domains are noisy; bucket into
    # freemail vs. corporate vs. missing, a common IEEE-CIS trick).
    freemail = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com"}
    for col in ["P_emaildomain", "R_emaildomain"]:
        if col in df.columns:
            df[col + "_bucket"] = np.where(
                df[col].isin(freemail), "freemail",
                np.where(df[col].isna(), "missing", "other")
            )

    return df


def encode_categoricals(train: pd.DataFrame, test: pd.DataFrame):
    """Label-encode categorical/object columns, fit on combined
    train+test categories so unseen test-set values don't break."""
    cat_cols = train.select_dtypes(include=["object", "str"]).columns.tolist()
    for col in cat_cols:
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        categories = combined.astype("category").cat.categories
        train[col] = pd.Categorical(train[col].astype(str), categories=categories).codes
        test[col] = pd.Categorical(test[col].astype(str), categories=categories).codes
    return train, test, cat_cols


# ---------------------------------------------------------------------
# 3. TRAIN / VALIDATION SPLIT — TEMPORAL, NOT RANDOM
# ---------------------------------------------------------------------
def temporal_split(train: pd.DataFrame, val_fraction: float = 0.2):
    """
    IMPORTANT: fraud detection models must be validated on a TEMPORAL
    split, never a random shuffle-split. Fraud patterns evolve over
    time, and random splits leak future information into training,
    producing inflated validation scores that collapse in production.
    """
    train_sorted = train.sort_values("TransactionDT")
    cutoff = int(len(train_sorted) * (1 - val_fraction))
    return train_sorted.iloc[:cutoff], train_sorted.iloc[cutoff:]


# ---------------------------------------------------------------------
# 4. MODEL TRAINING
# ---------------------------------------------------------------------
def train_model(train_df, val_df, feature_cols, cat_cols):
    train_set = lgb.Dataset(
        train_df[feature_cols], label=train_df["isFraud"],
        categorical_feature=[c for c in cat_cols if c in feature_cols],
    )
    val_set = lgb.Dataset(
        val_df[feature_cols], label=val_df["isFraud"],
        categorical_feature=[c for c in cat_cols if c in feature_cols],
        reference=train_set,
    )

    # scale_pos_weight compensates for the ~3.5% fraud class imbalance
    fraud_rate = train_df["isFraud"].mean()
    scale_pos_weight = (1 - fraud_rate) / fraud_rate

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 128,
        "max_depth": -1,
        "scale_pos_weight": scale_pos_weight,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )
    return model


def evaluate(model, val_df, feature_cols):
    preds = model.predict(val_df[feature_cols])
    y_true = val_df["isFraud"]

    roc_auc = roc_auc_score(y_true, preds)
    pr_auc = average_precision_score(y_true, preds)  # THE metric that matters here

    print(f"\nValidation ROC-AUC: {roc_auc:.4f}  (reference only — misleading on imbalanced data)")
    print(f"Validation PR-AUC:  {pr_auc:.4f}  (this is what you should optimize/report)")
    return preds


# ---------------------------------------------------------------------
# 4b. MODEL COMPARISON — LightGBM vs. XGBoost vs. Random Forest vs. LogReg
# ---------------------------------------------------------------------
def compare_models(train_df, val_df, feature_cols, cat_cols, lgb_model=None):
    """
    Trains a few alternative model types on the same temporal split and
    compares them on ROC-AUC and PR-AUC (NOT accuracy — see note in
    evaluate() above; accuracy is meaningless on a ~3.5% fraud rate).

    - LightGBM: reuses the already-trained model if passed in, otherwise
      trains fresh. Handles missing values and categoricals natively.
    - XGBoost: closest cousin to LightGBM, also tree-based gradient
      boosting. Also handles missing values natively.
    - Random Forest: a different (bagging, not boosting) tree-ensemble
      approach. Needs missing values filled in (doesn't handle NaN).
    - Logistic Regression: a simple linear baseline. Needs missing
      values filled in AND feature scaling. Useful as a sanity check —
      if it scores close to the tree models, the extra complexity may
      not be earning its keep; if it scores much worse, that tells you
      the fraud signal here is genuinely non-linear.
    """
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    y_train = train_df["isFraud"]
    y_val = val_df["isFraud"]
    fraud_rate = y_train.mean()
    scale_pos_weight = (1 - fraud_rate) / fraud_rate

    results = []

    # --- LightGBM (reuse if already trained) ---
    if lgb_model is None:
        lgb_model = train_model(train_df, val_df, feature_cols, cat_cols)
    lgb_preds = lgb_model.predict(val_df[feature_cols])
    results.append(("LightGBM", lgb_preds))

    # --- XGBoost ---
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=50,
        tree_method="hist",
        enable_categorical=False,  # our cat_cols are already label-encoded ints
    )
    xgb_model.fit(
        train_df[feature_cols], y_train,
        eval_set=[(val_df[feature_cols], y_val)],
        verbose=False,
    )
    xgb_preds = xgb_model.predict_proba(val_df[feature_cols])[:, 1]
    results.append(("XGBoost", xgb_preds))

    # --- Random Forest & Logistic Regression need NaNs filled ---
    train_filled = train_df[feature_cols].fillna(-999)
    val_filled = val_df[feature_cols].fillna(-999)

    print("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",  # RF's equivalent of scale_pos_weight
        n_jobs=-1,
        random_state=42,
    )
    rf_model.fit(train_filled, y_train)
    rf_preds = rf_model.predict_proba(val_filled)[:, 1]
    results.append(("Random Forest", rf_preds))

    print("Training Logistic Regression...")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_filled)
    val_scaled = scaler.transform(val_filled)

    lr_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
    )
    lr_model.fit(train_scaled, y_train)
    lr_preds = lr_model.predict_proba(val_scaled)[:, 1]
    results.append(("Logistic Regression", lr_preds))

    # --- Compare ---
    print(f"\n{'Model':<22}{'ROC-AUC':<12}{'PR-AUC':<12}")
    print("-" * 46)
    comparison = []
    for name, preds in results:
        roc = roc_auc_score(y_val, preds)
        pr = average_precision_score(y_val, preds)
        comparison.append({"model": name, "roc_auc": roc, "pr_auc": pr})
        print(f"{name:<22}{roc:<12.4f}{pr:<12.4f}")

    return comparison


# ---------------------------------------------------------------------
# 5. SHAP EXPLANATIONS -> LLM HANDOFF FORMAT
# ---------------------------------------------------------------------
def explain_top_flags(model, val_df, feature_cols, top_n: int = 5,
                      mid_band_n: int = 5, train_df=None, cat_cols=None):
    """
    Produces the structured evidence packages that feed the LLM
    investigation agent: instead of raw feature vectors, the agent gets a
    ranked list of "why this transaction is risky".

    Picks two groups of transactions:
      - "high":   the top_n highest-scoring transactions (the easy cases)
      - "medium": mid_band_n random transactions scored 0.3-0.8 (the
                  ambiguous band, where an investigation agent adds value)

    For each top risk factor it also records the feature's value and, for
    numeric features, its percentile in the training data. That is how the
    agent can describe opaque columns like V45 ("higher than 99% of
    training transactions") without knowing what they mean.
    """
    import shap

    cat_cols = set(cat_cols or [])
    preds = model.predict(val_df[feature_cols])

    top_idx = np.argsort(preds)[-top_n:][::-1]
    mid_candidates = np.where((preds >= 0.3) & (preds < 0.8))[0]
    rng = np.random.default_rng(42)
    mid_idx = (
        rng.choice(mid_candidates, size=min(mid_band_n, len(mid_candidates)), replace=False)
        if len(mid_candidates) else np.array([], dtype=int)
    )

    selected = [(int(i), "high") for i in top_idx] + [(int(i), "medium") for i in mid_idx]
    row_positions = [i for i, _ in selected]

    explainer = shap.TreeExplainer(model)
    subset = val_df.iloc[row_positions][feature_cols]
    shap_values = explainer.shap_values(subset)

    cases = []
    for k, (row_idx, band) in enumerate(selected):
        row = val_df.iloc[row_idx]
        sv = shap_values[k] if not isinstance(shap_values, list) else shap_values[1][k]
        top_features = sorted(
            zip(feature_cols, sv), key=lambda x: abs(x[1]), reverse=True
        )[:5]

        factors = []
        for f, v in top_features:
            factor = {"feature": f, "shap_value": float(v)}
            if f in cat_cols:
                # Encoded category codes carry no meaning on their own; the
                # agent reads the real value from the raw data instead.
                factor["type"] = "categorical"
            else:
                val = row[f]
                if pd.isna(val):
                    factor["value"] = None  # missing value
                else:
                    factor["value"] = float(val)
                    if train_df is not None:
                        col = train_df[f].dropna()
                        factor["percentile_in_train"] = round(float((col < val).mean() * 100), 1)
            factors.append(factor)

        cases.append({
            "transaction_id": int(row["TransactionID"]),
            "band": band,
            "ml_risk_score": float(preds[row_idx]),
            "transaction_amount": float(row["TransactionAmt"]),
            "top_risk_factors": factors,
        })
    return cases  # <-- saved to evidence_cases.json for investigation_agent.py


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main(data_dir: str, run_comparison: bool = False, top_n: int = 5, mid_band_n: int = 5):
    train, test = load_data(data_dir)

    train = engineer_features(train)
    test = engineer_features(test)
    train, test, cat_cols = encode_categoricals(train, test)

    drop_cols = ["TransactionID", "isFraud", "TransactionDT", "card_addr_id"]
    feature_cols = [c for c in train.columns if c not in drop_cols]

    train_split, val_split = temporal_split(train)

    model = train_model(train_split, val_split, feature_cols, cat_cols)
    evaluate(model, val_split, feature_cols)

    print("\nGenerating SHAP-based evidence packages for top-risk transactions...")
    cases = explain_top_flags(
        model, val_split, feature_cols, top_n=top_n, mid_band_n=mid_band_n,
        train_df=train_split, cat_cols=cat_cols,
    )
    for case in cases:
        print(case)

    with open("evidence_cases.json", "w") as f:
        json.dump(cases, f, indent=2)
    print("\nSaved evidence packages to evidence_cases.json")

    model.save_model("fraud_model.txt")
    print("\nModel saved to fraud_model.txt")

    visualize_model(model)

    if run_comparison:
        print("\n" + "=" * 50)
        print("Comparing against XGBoost, Random Forest, Logistic Regression...")
        print("=" * 50)
        comparison = compare_models(train_split, val_split, feature_cols, cat_cols, lgb_model=model)
        plot_model_comparison(comparison)


# ---------------------------------------------------------------------
# 6. MODEL VISUALIZATION
# ---------------------------------------------------------------------
def visualize_model(model, model_file: str = None):
    """
    Loads a saved LightGBM model (fraud_model.txt) and produces two views:
      1. Feature importance (by gain) - which features mattered most overall
      2. A single tree's structure - how one tree actually splits

    Can be called standalone: pass model_file="fraud_model.txt" to load
    a previously saved model instead of an in-memory Booster.
    """
    import lightgbm as lgb
    import matplotlib.pyplot as plt

    if model_file is not None:
        model = lgb.Booster(model_file=model_file)

    # 1. Feature importance by gain
    ax = lgb.plot_importance(
        model, max_num_features=20, importance_type="gain", figsize=(10, 8)
    )
    ax.set_title("Top 20 features by gain")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()
    print("Saved feature_importance.png")

    # 2. Single tree structure (requires the graphviz system binary,
    #    e.g. `brew install graphviz` on Mac, or apt-get on Linux)
    try:
        graph = lgb.create_tree_digraph(model, tree_index=0)
        graph.render("tree_0", format="png", cleanup=True)
        print("Saved tree_0.png")
    except Exception as e:
        print(f"Skipped tree diagram (graphviz not available?): {e}")


# ---------------------------------------------------------------------
# 7. MODEL COMPARISON PLOT (optional, --compare_models)
# ---------------------------------------------------------------------
def plot_model_comparison(comparison: list):
    """
    Bar chart of PR-AUC across the models compare_models() trained.
    comparison is the list of {"model", "roc_auc", "pr_auc"} dicts that
    compare_models() returns.
    """
    import matplotlib.pyplot as plt

    df = pd.DataFrame(comparison).sort_values("pr_auc", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.6 * len(df))))
    bars = ax.barh(df["model"], df["pr_auc"], color="#4C72B0")
    ax.set_xlabel("PR-AUC (higher is better)")
    ax.set_title("Model comparison — IEEE-CIS Fraud Detection")
    ax.set_xlim(0, 1)
    for bar, val in zip(bars, df["pr_auc"]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center")

    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.close()
    print("\nSaved model_comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument(
        "--compare_models", action="store_true",
        help="Also train XGBoost, Random Forest, and Logistic Regression "
             "for comparison against LightGBM (slower — trains 4 models total)."
    )
    parser.add_argument("--top_n", type=int, default=5,
                        help="Highest-risk cases to pull evidence for (default 5).")
    parser.add_argument("--mid_band_n", type=int, default=5,
                        help="Ambiguous (0.3-0.8 score) cases to sample (default 5).")
    args = parser.parse_args()
    main(args.data_dir, run_comparison=args.compare_models,
         top_n=args.top_n, mid_band_n=args.mid_band_n)
