"""
GCP Vertex AI Pipelines (Kubeflow DSL / TFX) Compiler & Champion-Challenger Registry Gate.
Generates declarative Vertex AI Pipeline specs for BigQuery extraction, Feature Store sync,
GPU training, INT8 edge compilation, and Vertex AI Model Registry deployment.
"""
from typing import Dict, Any


def compile_vertex_pipeline_spec(
    project_id: str = "cloud-mlops-iot-prod",
    region: str = "us-central1",
    bq_source_table: str = "cloud-mlops-iot-prod.iot_telemetry.ton_iot_sensor_events",
    feature_store_id: str = "iot_device_featurestore",
    min_roc_auc_gate: float = 0.95,
) -> Dict[str, Any]:
    """Builds a Vertex AI Pipelines v2.1 compatible DAG specification."""
    return {
        "pipelineInfo": {
            "name": "vertex-ai-iot-telemetry-mlops-pipeline",
            "description": "End-to-end Kubeflow/Vertex AI pipeline for Cloud-to-Edge IoT anomaly detection",
        },
        "sdkVersion": "kfp-2.7.0",
        "schemaVersion": "2.1.0",
        "defaultPipelineRoot": f"gs://{project_id}-vertex-pipelines/runs",
        "components": {
            "comp-bigquery-dataflow-ingest": {
                "executorLabel": "exec-dataflow-ingest",
                "inputDefinitions": {"parameters": {"bq_table": {"parameterType": "STRING"}}},
                "outputDefinitions": {"artifacts": {"raw_dataset": {"artifactType": {"schemaTitle": "system.Dataset"}}}},
            },
            "comp-vertex-feature-store-sync": {
                "executorLabel": "exec-feature-store-sync",
                "inputDefinitions": {" parameters": {"featurestore_id": {"parameterType": "STRING"}}},
                "outputDefinitions": {"artifacts": {"feature_view": {"artifactType": {"schemaTitle": "system.Dataset"}}}},
            },
            "comp-gpu-hybrid-trainer": {
                "executorLabel": "exec-gpu-trainer",
                "machineSpec": {"machineType": "n1-standard-8", "acceleratorType": "NVIDIA_TESLA_T4", "acceleratorCount": 1},
                "outputDefinitions": {
                    "artifacts": {
                        "fp32_model": {"artifactType": {"schemaTitle": "system.Model"}},
                        "int8_edge_model": {"artifactType": {"schemaTitle": "system.Model"}},
                        "metrics": {"artifactType": {"schemaTitle": "system.Metrics"}},
                    }
                },
            },
            "comp-champion-challenger-gate": {
                "executorLabel": "exec-eval-gate",
                "inputDefinitions": {"parameters": {"min_roc_auc": {"parameterType": "NUMBER_DOUBLE"}}},
            },
            "comp-vertex-model-registry-deploy": {
                "executorLabel": "exec-model-registry-deploy",
                "inputDefinitions": {"parameters": {"gke_cluster": {"parameterType": "STRING"}}},
            },
        },
        "runtimeConfig": {
            "parameterValues": {
                "project_id": project_id,
                "region": region,
                "bq_source_table": bq_source_table,
                "feature_store_id": feature_store_id,
                "min_roc_auc_gate": min_roc_auc_gate,
            }
        },
    }
