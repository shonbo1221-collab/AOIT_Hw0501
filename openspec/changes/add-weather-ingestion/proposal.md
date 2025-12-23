# Change: Add Weather Ingestion Capability

## Why
We need to ingest real-time weather data from the Taiwan Central Weather Administration (CWA) to populate our pipeline. This is the first step in the data pipeline.

## What Changes
- Add a new capability `weather-ingestion`
- Create a Python API client to fetch data from CWA Open Data API
- Implement data parsing for location and temperature

## Impact
- Affected specs: `weather-ingestion` (new)
- Affected code: `src/ingestion.py` (new)
