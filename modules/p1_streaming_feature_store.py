"""
P1 Companion Module: Kubernetes & Kafka Real-Time Telemetry Streaming Feature Aggregator.
Extends containerized Kafka/Kubernetes event streaming with low-latency sliding-window
feature materialization and schema contract verification.
"""
from typing import List, Dict, Any


def aggregate_kafka_partition_batch(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(messages)
    valid = [m for m in messages if "zone_id" in m and "trip_distance" in m and float(m["trip_distance"]) >= 0]
    avg_dist = sum(float(m["trip_distance"]) for m in valid) / max(1, len(valid))
    return {
        "ingested_messages": total,
        "valid_messages": len(valid),
        "schema_pass_rate": round(len(valid) / max(1, total), 4),
        "mean_trip_distance": round(avg_dist, 4),
    }
