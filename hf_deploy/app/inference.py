"""
Inference-only wrapper around the trained ParkSense model.
This module NEVER retrains. It loads model/parksense_model.pkl and scores new rows.
"""
import pickle
import pathlib
import numpy as np
import pandas as pd

MODEL_PATH = pathlib.Path(__file__).parent.parent / "model" / "parksense_model.pkl"


class ParkSensePredictor:
    def __init__(self, model_path: pathlib.Path = MODEL_PATH):
        with open(model_path, "rb") as f:
            artifacts = pickle.load(f)

        self.model        = artifacts["model"]
        self.calibrator   = artifacts["calibrator"]
        self.features     = artifacts["features"]
        self.le_junction  = artifacts["le_junction"]
        self.severity_map = artifacts["severity_map"]
        self.heavy_set    = artifacts["heavy_set"]
        self.threshold    = artifacts["optimal_threshold"]
        self.meta         = artifacts["meta"]
        self.fold_results = pd.DataFrame(artifacts["fold_results"])
        self.shap_df      = pd.DataFrame(artifacts["shap_importance"])

    def known_junctions(self):
        return sorted(self.le_junction.classes_.tolist())

    def _encode_junction(self, junction_name: str):
        if junction_name in self.le_junction.classes_:
            return int(self.le_junction.transform([junction_name])[0]), False
        return -1, True  # unseen junction flag

    def predict_single(self, row: dict) -> dict:
        """
        row must contain: junction_name, hour, dow, month_ord,
        roll_mean, roll_std, lag_1d_same_hour, lag_7d_same_hour,
        heavy_pct, compound_pct, max_severity
        """
        junction_enc, unseen = self._encode_junction(row["junction_name"])
        is_weekend = 1 if row["dow"] >= 5 else 0

        feat_vals = {
            "junction_enc":      junction_enc,
            "hour":              row["hour"],
            "dow":               row["dow"],
            "is_weekend":        is_weekend,
            "month_ord":         row["month_ord"],
            "roll_mean":         row["roll_mean"],
            "roll_std":          row["roll_std"],
            "lag_1d_same_hour":  row["lag_1d_same_hour"],
            "lag_7d_same_hour":  row["lag_7d_same_hour"],
            "heavy_pct":         row["heavy_pct"],
            "compound_pct":      row["compound_pct"],
            "max_severity":      row["max_severity"],
        }
        X = np.array([[feat_vals[f] for f in self.features]], dtype=np.float32)

        raw_score = float(self.model.predict_proba(X)[0, 1])
        cal_score = float(np.clip(self.calibrator.transform([raw_score])[0], 0.01, 0.99))
        is_spike  = cal_score >= self.threshold

        return {
            "junction_name":   row["junction_name"],
            "raw_score":       round(raw_score, 4),
            "calibrated_prob": round(cal_score, 4),
            "is_spike":        bool(is_spike),
            "threshold_used":  self.threshold,
            "unseen_junction": unseen,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        results = [self.predict_single(row) for row in df.to_dict(orient="records")]
        return pd.DataFrame(results)

    def model_card(self) -> dict:
        return self.meta
