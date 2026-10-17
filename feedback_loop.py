"""
Analyst Feedback -> Corrected Labels -> Retraining
====================================================

Closes the loop the original architecture called for: an analyst agrees
or corrects the agent's verdict on the dashboard (saved via its db
capability), and that correction becomes a label LightGBM can retrain on.

    dashboard feedback (db) --export--> feedback.json --this script--> corrected_labels.csv --merge--> retrain

Two ways to get feedback.json:
  1. Manual: open the dashboard's case list, for each case with feedback
     note it down, write feedback.json yourself in the shape below.
  2. Scripted export (needs an API key for the same provider used to
     build the dashboard - not required for this script itself): read
     every doc in the artifact's "feedback" collection and dump it.

feedback.json shape (list of objects, one per case with feedback):
  [{"transaction_id": 3519815, "agree": true,  "correctedVerdict": null,        "note": null},
   {"transaction_id": 3511148, "agree": false, "correctedVerdict": "legitimate", "note": "..."}]

WHY THIS MATTERS ON THIS DATASET SPECIFICALLY
  IEEE-CIS already has ground-truth isFraud labels, so there's no real
  gap for feedback to fill here - this script is the mechanism you'd use
  once this moves to real, unlabeled production transactions, where an
  analyst's corrected verdict IS the label. Run it here to see the
  mechanism work, not because this dataset strictly needs it.

USAGE:
  python feedback_loop.py --feedback feedback.json --data_dir ./data
"""

import argparse
import json
import os

import pandas as pd


def load_feedback(path: str) -> pd.DataFrame:
    with open(path) as f:
        raw = json.load(f)
    rows = []
    for r in raw:
        # An analyst's corrected belief about the label: explicit
        # correction wins; an "agree" confirms the agent's own call;
        # a disagree with no correction given is left as "unresolved"
        # rather than guessed at.
        if r.get("correctedVerdict"):
            corrected = 1 if r["correctedVerdict"] == "fraud" else 0
        elif r.get("agree") and "agent_verdict" in r:
            corrected = 1 if r["agent_verdict"] == "fraud" else 0
        else:
            corrected = None
        rows.append({
            "TransactionID": r["transaction_id"],
            "corrected_is_fraud": corrected,
            "note": r.get("note"),
        })
    df = pd.DataFrame(rows)
    unresolved = df["corrected_is_fraud"].isna().sum()
    if unresolved:
        print(f"{unresolved} feedback entries have no usable label "
              f"(disagreed but gave no correctedVerdict) - skipped.")
    return df.dropna(subset=["corrected_is_fraud"])


def merge_into_training_data(corrected: pd.DataFrame, data_dir: str, out_path: str):
    """
    Writes a corrected copy of train_transaction.csv's isFraud column.
    Never overwrites the original file - retraining should be a
    deliberate, reviewable step, not a silent side effect of running
    this script.
    """
    train = pd.read_csv(os.path.join(data_dir, "train_transaction.csv"))
    before = train["isFraud"].copy()

    merged = train.merge(
        corrected[["TransactionID", "corrected_is_fraud"]],
        on="TransactionID", how="left",
    )
    changed_mask = merged["corrected_is_fraud"].notna() & (merged["corrected_is_fraud"] != merged["isFraud"])
    merged.loc[merged["corrected_is_fraud"].notna(), "isFraud"] = merged.loc[
        merged["corrected_is_fraud"].notna(), "corrected_is_fraud"
    ].astype(int)
    merged = merged.drop(columns=["corrected_is_fraud"])

    n_changed = int(changed_mask.sum())
    n_applied = int(corrected["TransactionID"].isin(train["TransactionID"]).sum())
    print(f"Applied {n_applied} corrections, {n_changed} actually changed the label "
          f"(the rest confirmed what was already there).")

    merged.to_csv(out_path, index=False)
    print(f"Wrote corrected training data to {out_path}")
    print(f"To retrain on it: python ieee_cis_pipeline.py --data_dir <folder containing {os.path.basename(out_path)} "
          f"renamed to train_transaction.csv, plus the untouched identity/test files>")


def main(feedback_path: str, data_dir: str, out_path: str):
    corrected = load_feedback(feedback_path)
    if corrected.empty:
        print("No usable corrections in this feedback file - nothing to merge.")
        return
    merge_into_training_data(corrected, data_dir, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback", default="feedback.json")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--out", default="train_transaction_corrected.csv")
    args = parser.parse_args()
    main(args.feedback, args.data_dir, args.out)
