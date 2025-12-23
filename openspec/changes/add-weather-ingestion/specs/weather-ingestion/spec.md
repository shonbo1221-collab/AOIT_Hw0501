## ADDED Requirements

### Requirement: CWA Data Fetching
The system SHALL retrieve weather data from the CWA Open Data API using a provided API key.

#### Scenario: Successful Data Fetch
- **WHEN** the ingestion service is triggered with a valid API key
- **THEN** it returns a list of weather records containing 'locationName' and 'temperature'

#### Scenario: Invalid API Key
- **WHEN** the ingestion service is triggered with an invalid API key
- **THEN** it raises an authentication error
