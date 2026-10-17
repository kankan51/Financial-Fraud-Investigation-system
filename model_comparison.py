"""
IEEE-CIS Fraud Detection — Model Comparison
=============================================

Trains several different model types on the EXACT SAME preprocessed data
and temporal split as ieee_cis_pipeline.py, so the comparison is fair —
same features, same train/validation split, same evaluation metric.

Models compared:
  1. Logistic Regression  — simple linear baseline
  2. Random Forest        — bagged trees, no boosting
  3. XGBoost               — gradient boosting, different implementation than LightGBM
  4. LightGBM              — the baseline model from ieee_cis_pipeline.py
  5. Neural Network (MLP)  — PyTorch feed-forward net trained with backpropagation
  6. Extra Trees           — like Random Forest, with extra randomness in the splits
  7. HistGradientBoosting  — scikit-learn's LightGBM-style boosting
  8. CatBoost              — boosting with native categorical handling
  9. Isolation Forest      — UNSUPERVISED anomaly detection (never sees fraud labels)

SETUP:
  pip install scikit-learn xgboost lightgbm matplotlib torch catboost

USAGE:
  python model_comparison.py --data_dir ./data

Reuses load_data(), engineer_features(), encode_categoricals(), and
temporal_split() directly from ieee_cis_pipeline.py — no duplicated logic,
so if you update preprocessing there, this script stays in sync.
"""

import argparse
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

from ieee_cis_pipeline import (
    load_data,
    engineer_features,
    encode_categoricals,
    temporal_split,
)


def prepare_data(data_dir: str):
    """Runs the same preprocessing pipeline as ieee_cis_pipeline.py."""
    train, test = load_data(data_dir)
    train = engineer_features(train)
    test = engineer_features(test)
    train, test, cat_cols = encode_categoricals(train, test)

    drop_cols = ["TransactionID", "isFraud", "TransactionDT", "card_addr_id"]
    feature_cols = [c for c in train.columns if c not in drop_cols]

    train_split, val_split = temporal_split(train)
    return train_split, val_split, feature_cols, cat_cols


def evaluate_model(name, y_true, preds, train_seconds):
    roc_auc = roc_auc_score(y_true, preds)
    pr_auc = average_precision_score(y_true, preds)
    print(
        f"{name:20s}  PR-AUC: {pr_auc:.4f}   ROC-AUC: {roc_auc:.4f}   train time: {train_seconds:.1f}s"
    )
    return {
        "model": name,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "train_seconds": train_seconds,
    }


# ---------------------------------------------------------------------
# Model 1: Logistic Regression
# ---------------------------------------------------------------------
def run_logistic_regression(train_split, val_split, feature_cols):
    # Linear models need scaled features and can't handle NaNs directly —
    # unlike the tree models, which are why we didn't bother with this
    # for the LightGBM baseline.
    X_train = train_split[feature_cols].fillna(-999)
    X_val = val_split[feature_cols].fillna(-999)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    fraud_rate = train_split["isFraud"].mean()
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # logistic regression's equivalent of scale_pos_weight
        n_jobs=-1,
    )

    start = time.time()
    model.fit(X_train_scaled, train_split["isFraud"])
    elapsed = time.time() - start

    preds = model.predict_proba(X_val_scaled)[:, 1]
    return evaluate_model("Logistic Regression", val_split["isFraud"], preds, elapsed)


# ---------------------------------------------------------------------
# Model 2: Random Forest
# ---------------------------------------------------------------------
def run_random_forest(train_split, val_split, feature_cols):
    X_train = train_split[feature_cols].fillna(-999)
    X_val = val_split[feature_cols].fillna(-999)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    start = time.time()
    model.fit(X_train, train_split["isFraud"])
    elapsed = time.time() - start

    preds = model.predict_proba(X_val)[:, 1]
    return evaluate_model("Random Forest", val_split["isFraud"], preds, elapsed)


# ---------------------------------------------------------------------
# Model 3: XGBoost
# ---------------------------------------------------------------------
def run_xgboost(train_split, val_split, feature_cols):
    X_train = train_split[feature_cols].fillna(-999)
    X_val = val_split[feature_cols].fillna(-999)

    fraud_rate = train_split["isFraud"].mean()
    scale_pos_weight = (1 - fraud_rate) / fraud_rate

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        early_stopping_rounds=50,
        n_jobs=-1,
        random_state=42,
    )

    start = time.time()
    model.fit(
        X_train,
        train_split["isFraud"],
        eval_set=[(X_val, val_split["isFraud"])],
        verbose=False,
    )
    elapsed = time.time() - start

    preds = model.predict_proba(X_val)[:, 1]
    return evaluate_model("XGBoost", val_split["isFraud"], preds, elapsed)


# ---------------------------------------------------------------------
# Model 4: LightGBM (same as the ieee_cis_pipeline.py baseline)
# ---------------------------------------------------------------------
def run_lightgbm(train_split, val_split, feature_cols, cat_cols):
    train_set = lgb.Dataset(
        train_split[feature_cols],
        label=train_split["isFraud"],
        categorical_feature=[c for c in cat_cols if c in feature_cols],
    )
    val_set = lgb.Dataset(
        val_split[feature_cols],
        label=val_split["isFraud"],
        categorical_feature=[c for c in cat_cols if c in feature_cols],
        reference=train_set,
    )

    fraud_rate = train_split["isFraud"].mean()
    scale_pos_weight = (1 - fraud_rate) / fraud_rate

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 128,
        "scale_pos_weight": scale_pos_weight,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    start = time.time()
    model = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )
    elapsed = time.time() - start

    preds = model.predict(val_split[feature_cols])
    return evaluate_model("LightGBM", val_split["isFraud"], preds, elapsed)


# ---------------------------------------------------------------------
# Model 5: Neural Network (MLP) trained with backpropagation
# ---------------------------------------------------------------------
def run_neural_network(train_split, val_split, feature_cols):
    import torch
    import torch.nn as nn

    torch.manual_seed(42)
    np.random.seed(42)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Neural network training on device: {device}")

    # --- Preprocessing: neural nets need clean, scaled numeric input ---
    # Trees split on thresholds, so raw scale and NaNs don't bother them.
    # A neural net multiplies inputs by weights, so unscaled columns
    # (e.g. TransactionAmt in the thousands next to 0/1 flags) would
    # dominate training and NaNs would poison every gradient.
    medians = train_split[feature_cols].median()
    X_train = train_split[feature_cols].fillna(medians).fillna(0)
    X_val = val_split[feature_cols].fillna(medians).fillna(0)

    scaler = StandardScaler()
    # Clip after scaling so a few extreme outliers can't blow up the gradients
    X_train = np.clip(scaler.fit_transform(X_train), -5, 5).astype(np.float32)
    X_val = np.clip(scaler.transform(X_val), -5, 5).astype(np.float32)

    y_train = train_split["isFraud"].values.astype(np.float32)
    y_val = val_split["isFraud"].values

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_val_t = torch.from_numpy(X_val)
    n_features = X_train.shape[1]

    # --- Architecture: input -> 256 -> 128 -> 1 (one raw score, "logit") ---
    model = nn.Sequential(
        nn.Linear(n_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1),
    ).to(device)

    # Class imbalance: the neural-net equivalent of scale_pos_weight.
    # A missed fraud case contributes ~28x more loss than a false alarm.
    fraud_rate = y_train.mean()
    pos_weight = torch.tensor([(1 - fraud_rate) / fraud_rate], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    def predict(X_t):
        model.eval()
        outputs = []
        with torch.no_grad():
            for i in range(0, len(X_t), 8192):
                xb = X_t[i : i + 8192].to(device)
                outputs.append(torch.sigmoid(model(xb)).cpu().numpy().ravel())
        return np.concatenate(outputs)

    batch_size = 2048
    max_epochs = 30
    patience = 4
    best_pr_auc, best_state, epochs_without_improvement = -1.0, None, 0
    n_rows = len(X_train_t)

    start = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = torch.randperm(n_rows)
        running_loss = 0.0

        for i in range(0, n_rows, batch_size):
            idx = permutation[i : i + batch_size]
            xb = X_train_t[idx].to(device)
            yb = y_train_t[idx].to(device)

            # ---- one training step: this is where backpropagation happens ----
            optimizer.zero_grad()  # 1. clear gradients from the last step
            logits = model(xb).squeeze(1)  # 2. FORWARD pass: inputs -> raw scores
            loss = criterion(logits, yb)  # 3. how wrong were the scores?
            loss.backward()  # 4. BACKPROP: chain rule computes d(loss)/d(weight)
            #    for every weight, from output layer back to input
            optimizer.step()  # 5. Adam nudges each weight against its gradient
            # ------------------------------------------------------------------

            running_loss += loss.item() * len(idx)

        val_preds = predict(X_val_t)
        val_pr_auc = average_precision_score(y_val, val_preds)
        print(
            f"  epoch {epoch:2d}  train loss: {running_loss / n_rows:.4f}  val PR-AUC: {val_pr_auc:.4f}"
        )

        # Early stopping: keep the best epoch, stop once it hasn't improved for `patience` epochs
        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  early stopping at epoch {epoch}")
                break

    elapsed = time.time() - start
    model.load_state_dict(best_state)
    preds = predict(X_val_t)
    return evaluate_model("Neural Network (MLP)", y_val, preds, elapsed)


# ---------------------------------------------------------------------
# Model 6: Extra Trees
# ---------------------------------------------------------------------
def run_extra_trees(train_split, val_split, feature_cols):
    X_train = train_split[feature_cols].fillna(-999)
    X_val = val_split[feature_cols].fillna(-999)

    # Same setup as Random Forest, but each split threshold is chosen at
    # random instead of searched for, which makes training faster and the
    # ensemble less prone to overfitting noisy features.
    model = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    start = time.time()
    model.fit(X_train, train_split["isFraud"])
    elapsed = time.time() - start

    preds = model.predict_proba(X_val)[:, 1]
    return evaluate_model("Extra Trees", val_split["isFraud"], preds, elapsed)


# ---------------------------------------------------------------------
# Model 7: HistGradientBoosting (scikit-learn)
# ---------------------------------------------------------------------
def run_hist_gradient_boosting(train_split, val_split, feature_cols):
    # Handles NaN natively, so no fillna needed. Categorical columns are
    # treated as ordinary numeric codes here.
    X_train = train_split[feature_cols]
    X_val = val_split[feature_cols]

    # Fixed number of rounds instead of scikit-learn's built-in early
    # stopping, because that carves out a RANDOM validation slice of the
    # training data, which would undo the temporal-split discipline.
    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=64,
        class_weight="balanced",
        early_stopping=False,
        random_state=42,
    )

    start = time.time()
    model.fit(X_train, train_split["isFraud"])
    elapsed = time.time() - start

    preds = model.predict_proba(X_val)[:, 1]
    return evaluate_model("HistGradientBoosting", val_split["isFraud"], preds, elapsed)


# ---------------------------------------------------------------------
# Model 8: CatBoost
# ---------------------------------------------------------------------
def run_catboost(train_split, val_split, feature_cols, cat_cols):
    from catboost import CatBoostClassifier, Pool

    cat_in_features = [c for c in cat_cols if c in feature_cols]

    # CatBoost treats these columns as true categories (using ordered
    # target statistics) instead of as ordered numbers. They must be
    # integer or string typed, which our label-encoded codes are.
    X_train = train_split[feature_cols].copy()
    X_val = val_split[feature_cols].copy()
    for c in cat_in_features:
        X_train[c] = X_train[c].astype(int)
        X_val[c] = X_val[c].astype(int)

    train_pool = Pool(X_train, train_split["isFraud"], cat_features=cat_in_features)
    val_pool = Pool(X_val, val_split["isFraud"], cat_features=cat_in_features)

    model = CatBoostClassifier(
        iterations=800,
        learning_rate=0.05,
        depth=6,
        auto_class_weights="Balanced",
        eval_metric="AUC",
        early_stopping_rounds=100,
        thread_count=-1,
        random_seed=42,
        verbose=0,
    )

    start = time.time()
    model.fit(train_pool, eval_set=val_pool)
    elapsed = time.time() - start

    preds = model.predict_proba(val_pool)[:, 1]
    return evaluate_model("CatBoost", val_split["isFraud"], preds, elapsed)


# ---------------------------------------------------------------------
# Model 9: Isolation Forest (unsupervised)
# ---------------------------------------------------------------------
def run_isolation_forest(train_split, val_split, feature_cols):
    X_train = train_split[feature_cols].fillna(-999)
    X_val = val_split[feature_cols].fillna(-999)

    # Fit WITHOUT labels: it learns what "normal" looks like and scores
    # how easily each transaction can be isolated from the rest. Fraud
    # is expected to be easier to isolate, but only if it looks unusual.
    model = IsolationForest(
        n_estimators=200,
        max_samples=256,
        n_jobs=-1,
        random_state=42,
    )

    start = time.time()
    model.fit(X_train)
    elapsed = time.time() - start

    # score_samples returns higher = more normal, so negate it. These are
    # anomaly scores, not probabilities, but PR-AUC and ROC-AUC only need
    # a ranking, so they can be compared with the supervised models.
    scores = -model.score_samples(X_val)
    return evaluate_model("Isolation Forest", val_split["isFraud"], scores, elapsed)


# ---------------------------------------------------------------------
# Comparison chart
# ---------------------------------------------------------------------
def plot_comparison(results: list):
    df = pd.DataFrame(results).sort_values("pr_auc", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(5, 0.6 * len(df))))
    bars = ax.barh(df["model"], df["pr_auc"], color="#4C72B0")
    ax.set_xlabel("PR-AUC (higher is better)")
    ax.set_title("Model comparison — IEEE-CIS Fraud Detection")
    ax.set_xlim(0, 1)

    for bar, val in zip(bars, df["pr_auc"]):
        ax.text(
            val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center"
        )

    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.close()
    print("\nSaved model_comparison.png")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main(data_dir: str):
    print("Preparing data (shared across all models)...\n")
    train_split, val_split, feature_cols, cat_cols = prepare_data(data_dir)

    results = []
    print("\nTraining models...\n")
    results.append(run_logistic_regression(train_split, val_split, feature_cols))
    results.append(run_random_forest(train_split, val_split, feature_cols))
    results.append(run_xgboost(train_split, val_split, feature_cols))
    results.append(run_lightgbm(train_split, val_split, feature_cols, cat_cols))
    try:
        results.append(run_neural_network(train_split, val_split, feature_cols))
    except ImportError:
        print("Skipping neural network: PyTorch not installed (pip install torch)")

    results.append(run_extra_trees(train_split, val_split, feature_cols))
    results.append(run_hist_gradient_boosting(train_split, val_split, feature_cols))
    try:
        results.append(run_catboost(train_split, val_split, feature_cols, cat_cols))
    except ImportError:
        print("Skipping CatBoost: not installed (pip install catboost)")
    results.append(run_isolation_forest(train_split, val_split, feature_cols))

    results_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    print("\n=== Final comparison (sorted by PR-AUC) ===")
    print(results_df.to_string(index=False))

    results_df.to_csv("model_comparison_results.csv", index=False)
    print("\nSaved model_comparison_results.csv")

    plot_comparison(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()
    main(args.data_dir)
