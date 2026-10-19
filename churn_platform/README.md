# AI-Powered Customer Churn Prediction Platform

This project is the implementation scaffold for the MCA academic churn prediction platform described in the project plan. The application illustrates a local prototype architecture with:

- a schema-aware data-loading layer
- a reusable preprocessing and model training pipeline
- a FastAPI service with health and prediction endpoints
- a Streamlit dashboard for analyst review
- a CQRS-inspired separation between query and command services

## How to use the current implementation

Run the whole local project from one entry point on Windows from the project folder:

```cmd
start_project.bat
```

The launcher creates a Python 3.11 virtual environment if needed, installs or updates dependencies, retrains the model and report on every start, then opens the API and dashboard in separate windows. It selects available local ports and opens the dashboard in your browser.

Open the API docs at the `/docs` URL printed by the launcher. The dashboard URL is printed there as well. Select optional customer fields, submit multiple records, and review their risk bands and percentages in the session chart and table.

To run the tests manually from the same project folder:

```cmd
.venv\Scripts\python.exe -m pytest tests -q
```

## Project structure

- app/ - shared core domain and service logic
- api/ - FastAPI entry point
- dashboard/ - Streamlit analyst interface
- tests/ - regression and contract checks
