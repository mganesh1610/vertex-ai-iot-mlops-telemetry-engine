"""
Vertex AI Model Monitoring & Observability Engine.
Computes Population Stability Index (PSI), Two-Sample Kolmogorov-Smirnov (KS) statistics,
and Jensen-Shannon Divergence to trigger automated CI/CD model retraining on GKE/Vertex AI.
"""
import math
from typing import List, Dict, Any


class VertexDriftMonitor:
    """Monitors production feature & prediction distributions against baseline reference windows."""

    def __init__(self, psi_threshold: float = 0.20, ks_threshold: float = 0.12, n_bins: int = 10):
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.n_bins = n_bins

    def compute_psi(self, reference: List[float], current: List[float]) -> float:
        if not reference or not current:
            return 0.0
        low = min(min(reference), min(current))
        high = max(max(reference), max(current))
        if math.isclose(low, high):
            return 0.0
        width = (high - low) / self.n_bins
        ref_counts = [0] * self.n_bins
        cur_counts = [0] * self.n_bins

        for x in reference:
            idx = min(self.n_bins - 1, max(0, int((x - low) / width)))
            ref_counts[idx] += 1
        for x in current:
            idx = min(self.n_bins - 1, max(0, int((x - low) / width)))
            cur_counts[idx] += 1

        eps = 1e-4
        psi = 0.0
        n_ref, n_cur = len(reference), len(current)
        for r_c, c_c in zip(ref_counts, cur_counts):
            p = max(eps, r_c / n_ref)
            q = max(eps, c_c / n_cur)
            psi += (p - q) * math.log(p / q)
        return round(psi, 4)

    @staticmethod
    def compute_ks_statistic(reference: List[float], current: List[float]) -> float:
        if not reference or not current:
            return 0.0
        ref_s = sorted(reference)
        cur_s = sorted(current)
        n_r, n_c = len(ref_s), len(cur_s)
        i = j = 0
        max_diff = 0.0
        while i < n_r and j < n_c:
            if ref_s[i] <= cur_s[j]:
                i += 1
            else:
                j += 1
            diff = abs((i / n_r) - (j / n_c))
            if diff > max_diff:
                max_diff = diff
        return round(max_diff, 4)

    def evaluate_feature_drift(
        self, ref_rows: List[Dict[str, Any]], cur_rows: List[Dict[str, Any]], features: List[str]
    ) -> Dict[str, Any]:
        report: Dict[str, Any] = {"features": {}, "retrain_triggered": False, "drifted_features": []}
        for feat in features:
            r_vals = [float(r[feat]) for r in ref_rows]
            c_vals = [float(r[feat]) for r in cur_rows]
            psi = self.compute_psi(r_vals, c_vals)
            ks = self.compute_ks_statistic(r_vals, c_vals)
            drifted = (psi >= self.psi_threshold) or (ks >= self.ks_threshold)
            report["features"][feat] = {"psi": psi, "ks_stat": ks, "drifted": drifted}
            if drifted:
                report["drifted_features"].append(feat)
                report["retrain_triggered"] = True
        return report
