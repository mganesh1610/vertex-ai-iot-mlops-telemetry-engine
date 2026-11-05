"""
High-Dimensional IoT Telemetry Feature Engineering & Vertex AI Feature Store Pipeline.
Supports sliding-window statistical moments, rate-of-change deltas, spectral energy,
and point-in-time feature hydration for BigQuery and Vertex AI Feature Store.
"""
import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple


@dataclass
class TelemetryEvent:
    device_id: str
    sensor_type: str  # thermostat, motion, door_contact, pressure, fridge_temp
    timestamp_sec: float
    temperature_c: float
    humidity_pct: float
    vibration_rms: float
    power_watts: float
    door_state: int
    motion_active: int
    packet_latency_ms: float
    label_anomaly: int = 0


class TelemetryFeatureStorePipeline:
    """Object-oriented feature transformer compatible with Cloud Dataflow & Vertex AI Feature Store."""

    def __init__(self, window_size: int = 12):
        if window_size < 2:
            raise ValueError("window_size must be >= 2")
        self.window_size = window_size

    @staticmethod
    def _mean_std(vals: List[float]) -> Tuple[float, float]:
        n = len(vals)
        if n == 0:
            return 0.0, 0.0
        mean = sum(vals) / n
        var = sum((x - mean) ** 2 for x in vals) / max(1, n - 1)
        return mean, math.sqrt(var)

    @staticmethod
    def _fft_first_harmonic_energy(vals: List[float]) -> float:
        """Computes normalized discrete Fourier first-harmonic spectral energy."""
        n = len(vals)
        if n < 2:
            return 0.0
        mean = sum(vals) / n
        centered = [x - mean for x in vals]
        re = sum(centered[k] * math.cos(2.0 * math.pi * k / n) for k in range(n))
        im = sum(-centered[k] * math.sin(2.0 * math.pi * k / n) for k in range(n))
        return round((re * re + im * im) / (n * n), 6)

    def transform_stream(self, events: List[TelemetryEvent]) -> List[Dict[str, Any]]:
        """Generates point-in-time correct rolling features per device_id without lookahead."""
        by_device: Dict[str, List[TelemetryEvent]] = {}
        for ev in sorted(events, key=lambda e: e.timestamp_sec):
            by_device.setdefault(ev.device_id, []).append(ev)

        feature_rows: List[Dict[str, Any]] = []
        for dev_id, dev_events in by_device.items():
            for idx, ev in enumerate(dev_events):
                start = max(0, idx - self.window_size + 1)
                win = dev_events[start : idx + 1]
                temps = [w.temperature_c for w in win]
                vibs = [w.vibration_rms for w in win]
                pwrs = [w.power_watts for w in win]
                lats = [w.packet_latency_ms for w in win]

                t_mean, t_std = self._mean_std(temps)
                v_mean, v_std = self._mean_std(vibs)
                p_mean, p_std = self._mean_std(pwrs)
                l_mean, l_std = self._mean_std(lats)

                temp_delta = ev.temperature_c - temps[-2] if len(temps) > 1 else 0.0
                temp_zscore = (ev.temperature_c - t_mean) / (t_std + 1e-6)
                vib_zscore = (ev.vibration_rms - v_mean) / (v_std + 1e-6)
                spectral_energy = self._fft_first_harmonic_energy(vibs)
                duty_cycle = sum(w.motion_active or w.door_state for w in win) / len(win)

                feature_rows.append({
                    "entity_id": dev_id,
                    "sensor_type": ev.sensor_type,
                    "feature_timestamp": ev.timestamp_sec,
                    "temp_c": round(ev.temperature_c, 4),
                    "temp_roll_mean": round(t_mean, 4),
                    "temp_roll_std": round(t_std, 4),
                    "temp_delta_1": round(temp_delta, 4),
                    "temp_zscore": round(temp_zscore, 4),
                    "vib_rms": round(ev.vibration_rms, 4),
                    "vib_roll_mean": round(v_mean, 4),
                    "vib_zscore": round(vib_zscore, 4),
                    "vib_spectral_energy": spectral_energy,
                    "power_roll_mean": round(p_mean, 4),
                    "power_roll_std": round(p_std, 4),
                    "latency_roll_mean": round(l_mean, 4),
                    "latency_roll_std": round(l_std, 4),
                    "security_duty_cycle": round(duty_cycle, 4),
                    "label_anomaly": ev.label_anomaly,
                })
        return feature_rows
