-- BigQuery Offline Store & Vertex AI Feature Store Entity Table DDL
CREATE TABLE IF NOT EXISTS `cloud_mlops_iot_prod.iot_telemetry.device_rolling_features`
(
  entity_id STRING NOT NULL OPTIONS(description="Unique IoT sensor / device identifier"),
  sensor_type STRING NOT NULL OPTIONS(description="Device class: thermostat, motion, door_contact, fridge_temp"),
  feature_timestamp TIMESTAMP NOT NULL OPTIONS(description="Point-in-time event timestamp for Vertex AI Feature Store"),
  temp_c FLOAT64,
  temp_roll_mean FLOAT64,
  temp_roll_std FLOAT64,
  temp_delta_1 FLOAT64,
  temp_zscore FLOAT64,
  vib_rms FLOAT64,
  vib_roll_mean FLOAT64,
  vib_zscore FLOAT64,
  vib_spectral_energy FLOAT64,
  power_roll_mean FLOAT64,
  power_roll_std FLOAT64,
  latency_roll_mean FLOAT64,
  latency_roll_std FLOAT64,
  security_duty_cycle FLOAT64,
  label_anomaly INT64
)
PARTITION BY DATE(feature_timestamp)
CLUSTER BY entity_id, sensor_type;
