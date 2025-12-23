# Project Context

## Purpose
Taiwan Weather Data Pipeline System.
The goal is to develop a complete weather data pipeline that:
1. Downloads weather data from the Central Weather Administration (CWA) API.
2. Parses the JSON data to extract location and temperature information.
3. Stores the data in an SQLite database.
4. Visualizes the data using a Streamlit application, mimicking the CWA's interface.

## Tech Stack
- Python 3.x
- Streamlit (Web Interface)
- SQLite (Database)
- requests (API Fetching)
- pandas (Data Processing)

## Project Conventions

### Code Style
- Follow PEP 8 guidelines for Python code.
- Use explicit variable names.
- Type hinting is encouraged.

### Architecture Patterns
- ETL Pipeline: Extract (API), Transform (Parsing), Load (SQLite).
- Streamlit for frontend visualization.
- Modular design: separate database handling, API fetching, and UI logic.

### Testing Strategy
- Manual testing of API responses.
- Validation of data insertion into SQLite.
- Visual verification of Streamlit dashboard.

### Git Workflow
- Feature branches for new capabilities.
- Direct commits to main allowed for initial setup.

## Domain Context
- Central Weather Administration (CWA) API structure.
- Taiwan geography (regions/locations) for grouping data.

## Important Constraints
- CWA API rate limits and key management.
- Local SQLite database (no remote DB server required).

## External Dependencies
- CWA Open Data API (requires API Token).
