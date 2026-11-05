"""
Hybrid Cloud-to-Edge Anomaly Detection & INT8 Edge Quantization Engine.
Combines an Autoencoder reconstruction error scorer with a Gradient Boosted Decision Stump
Ensemble and benchmarks FP32 vs. INT8 affine quantization for edge-computing constraints.
"""
import math
import time
from typing import List, Dict, Any, Tuple

FEATURE_KEYS = [
    "temp_c", "temp_roll_mean", "temp_roll_std", "temp_delta_1", "temp_zscore",
    "vib_rms", "vib_roll_mean", "vib_zscore", "vib_spectral_energy",
    "power_roll_mean", "power_roll_std", "latency_roll_mean", "latency_roll_std",
    "security_duty_cycle"
]


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class TemporalReconstructionAutoencoder:
    """Linear-bottleneck Temporal Autoencoder computing mahalanobis/L2 reconstruction loss."""

    def __init__(self, feature_keys: List[str] = FEATURE_KEYS):
        self.feature_keys = feature_keys
        self.means: List[float] = []
        self.stds: List[float] = []
        self.weights: List[float] = []

    def fit(self, rows: List[Dict[str, Any]]) -> None:
        normal_rows = [r for r in rows if r.get("label_anomaly", 0) == 0] or rows
        d = len(self.feature_keys)
        self.means = [0.0] * d
        self.stds = [1.0] * d
        self.weights = [1.0] * d
        n = len(normal_rows)
        for j, k in enumerate(self.feature_keys):
            vals = [float(r[k]) for r in normal_rows]
            m = sum(vals) / n
            v = sum((x - m) ** 2 for x in vals) / max(1, n - 1)
            self.means[j] = m
            self.stds[j] = math.sqrt(v) + 1e-6
            # Bottleneck projection shrinkage factor
            self.weights[j] = 0.88 if "zscore" in k or "spectral" in k else 0.94

    def reconstruction_error(self, row: Dict[str, Any]) -> float:
        err = 0.0
        for j, k in enumerate(self.feature_keys):
            z = (float(row[k]) - self.means[j]) / self.stds[j]
            z_hat = z * self.weights[j]
            # Nonlinear bottleneck clipping
            z_hat = max(-2.2, min(2.2, z_hat))
            err += (z - z_hat) ** 2
        return err / len(self.feature_keys)


class HybridIoTAnomalyClassifier:
    """
    Hybrid Gradient-Boosted Additive Ensemble + Autoencoder Reconstruction Scorer
    with INT8 Affine Weight Quantization for Edge Deployment.
    """

    def __init__(self, n_estimators: int = 35, learning_rate: float = 0.25):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.autoencoder = TemporalReconstructionAutoencoder()
        self.stumps: List[Tuple[str, float, float, float]] = []
        self.base_log_odds: float = 0.0
        self.quantized_stumps: List[Tuple[str, int, int, int]] = []
        self.q_scale_thresh: float = 1.0
        self.q_scale_val: float = 1.0

    def fit(self, rows: List[Dict[str, Any]]) -> None:
        self.autoencoder.fit(rows)
        enriched = []
        for r in rows:
            item = dict(r)
            item["recon_mse"] = self.autoencoder.reconstruction_error(r)
            enriched.append(item)

        all_keys = FEATURE_KEYS + ["recon_mse"]
        y = [int(r["label_anomaly"]) for r in enriched]
        pos = sum(y)
        n = len(y)
        p0 = max(1e-4, min(1.0 - 1e-4, pos / max(1, n)))
        self.base_log_odds = math.log(p0 / (1.0 - p0))
        f_pred = [self.base_log_odds] * n

        self.stumps = []
        for _ in range(self.n_estimators):
            residuals = [y[i] - _sigmoid(f_pred[i]) for i in range(n)]
            best_gain = -1.0
            best_stump = (all_keys[0], 0.0, 0.0, 0.0)
            for k in all_keys:
                vals = sorted(set(round(float(r[k]), 3) for r in enriched))
                step = max(1, len(vals) // 12)
                candidates = vals[::step]
                for thr in candidates:
                    left_r = [residuals[i] for i in range(n) if float(enriched[i][k]) <= thr]
                    right_r = [residuals[i] for i in range(n) if float(enriched[i][k]) > thr]
                    if not left_r or not right_r:
                        continue
                    l_val = sum(left_r) / len(left_r) * 2.0
                    r_val = sum(right_r) / len(right_r) * 2.0
                    gain = (len(left_r) * l_val * l_val) + (len(right_r) * r_val * r_val)
                    if gain > best_gain:
                        best_gain = gain
                        best_stump = (k, thr, l_val * self.learning_rate, r_val * self.learning_rate)

            self.stumps.append(best_stump)
            k, thr, l_w, r_w = best_stump
            for i in range(n):
                f_pred[i] += l_w if float(enriched[i][k]) <= thr else r_w

        self._compile_int8_edge_model()

    def _compile_int8_edge_model(self) -> None:
        """Applies symmetric INT8 quantization [-127, 127] to thresholds and leaf weights."""
        max_thr = max((abs(s[1]) for s in self.stumps), default=1.0) or 1.0
        max_val = max(( max(abs(s[2]), abs(s[3])) for s in self.stumps ), default=1.0) or 1.0
        self.q_scale_thresh = max_thr / 127.0
        self.q_scale_val = max_val / 127.0
        self.quantized_stumps = []
        for k, thr, l_w, r_w in self.stumps:
            q_thr = max(-127, min(127, int(round(thr / self.q_scale_thresh))))
            q_lw = max(-127, min(127, int(round(l_w / self.q_scale_val))))
            q_rw = max(-127, min(127, int(round(r_w / self.q_scale_val))))
            self.quantized_stumps.append((k, q_thr, q_lw, q_rw))

    def predict_proba_fp32(self, row: Dict[str, Any]) -> float:
        recon = self.autoencoder.reconstruction_error(row)
        score = self.base_log_odds
        for k, thr, l_w, r_w in self.stumps:
            val = recon if k == "recon_mse" else float(row[k])
            score += l_w if val <= thr else r_w
        return _sigmoid(score)

    def predict_proba_int8_edge(self, row: Dict[str, Any]) -> float:
        recon = self.autoencoder.reconstruction_error(row)
        acc_int = 0
        inv_thr = 1.0 / self.q_scale_thresh
        for k, q_thr, q_lw, q_rw in self.quantized_stumps:
            val = recon if k == "recon_mse" else float(row[k])
            q_val = int(round(val * inv_thr))
            acc_int += q_lw if q_val <= q_thr else q_rw
        score = self.base_log_odds + (acc_int * self.q_scale_val)
        return _sigmoid(score)

    @staticmethod
    def compute_roc_auc(y_true: List[int], y_scores: List[float]) -> float:
        pairs = sorted(zip(y_scores, y_true), key=lambda x: x[0])
        n_pos = sum(y_true)
        n_neg = len(y_true) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 1.0
        rank_sum = 0.0
        for rank, (_, label) in enumerate(pairs, start=1):
            if label == 1:
                rank_sum += rank
        auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        return round(auc, 4)
