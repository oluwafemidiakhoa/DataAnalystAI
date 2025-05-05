# DataVisionAI Workspace (Power BI on Steroids - vNext) 🚀

## Overview

DataVisionAI is an AI-Native Analytics Workspace designed to streamline the entire data analysis lifecycle. It empowers users to connect to various data sources, profile data, perform AI-assisted cleaning and transformation, explore data visually, engineer relevant features, conduct advanced analysis using natural language queries, generate insightful reports with AI narratives, track KPIs, and receive actionable recommendations. This project leverages Google's Gemini models as an integrated AI co-pilot.

Built with Python, Streamlit, SQLAlchemy, Pandas, Plotly, and the Google Generative AI SDK.

**Status:** *Under Development*

## Key Features (Vision)

*   **Unified Workflow:** Guides users seamlessly through Connect & Profile -> Clean & Transform -> Explore & Engineer -> Analyze & Insight -> Report & Recommend.
*   **Multiple Data Sources:** Connect to Files (CSV, Excel, JSON), PostgreSQL, MySQL, SQLite, MongoDB, MS SQL Server, Oracle DB. (Cloud storage planned).
*   **AI-Powered Profiling:** Automated data profiling with statistical summaries and AI-generated insights into data quality and structure.
*   **Intelligent Cleaning & Transformation:** Receive AI suggestions for cleaning steps (missing values, duplicates, type issues) and transform data using natural language commands (requires safe code execution).
*   **Advanced Feature Engineering:** AI suggestions for creating new features (date parts, interactions, aggregations, encodings, etc.) based on data and analysis goals.
*   **Natural Language Querying (NLQ):** Ask questions in plain English to query connected SQL databases (auto-generates SQL). (NLQ for Pandas/Mongo planned).
*   **Deep AI Insights:** Go beyond basic summaries to get AI analysis of trends, anomalies, key drivers, and potential segments in your data.
*   **Automated Analytics:** Features for automated Forecasting and Segmentation (Clustering).
*   **Root Cause Analysis (Chat):** Interactive chat interface to investigate anomalies with AI guidance.
*   **Dynamic Reporting:** Generate narrative reports combining text summaries, KPIs, visualizations, and (optionally) AI-generated images.
*   **KPI Management:** Define, track, and visualize Key Performance Indicators.
*   **Actionable Recommendations:** Receive AI-generated business recommendations based on analysis, with rationale and feedback mechanisms.
*   **(Planned) Project Management:** Organize analyses within projects (requires database).
*   **(Planned) Data Catalog:** Browse and understand connected data sources (#3).
*   **(Planned) Data Quality Monitoring:** Define rules and monitor data quality (#2).
*   **(Planned) Collaboration:** User accounts and shared workspaces (#13).
*   **(Planned) Secure Code Execution:** Safe sandbox for applying AI-generated code (#6).
*   **(Planned) Export Options:** Export reports to Markdown, PDF, PPTX (#11).

## Technology Stack

*   **Frontend:** Streamlit (Multi-Page App)
*   **Backend Core:** Python 3.10+
*   **AI Engine:** Google Gemini API (via `google-generativeai` SDK)
*   **Data Handling:** Pandas, NumPy
*   **Databases (Connectors):**
    *   SQLAlchemy (Core ORM)
    *   psycopg2-binary (PostgreSQL)
    *   mysql-connector-python (MySQL)
    *   pymongo (MongoDB)
    *   pyodbc (MS SQL Server - *Requires system ODBC driver*)
    *   oracledb (Oracle DB - *May require Oracle Instant Client*)
*   **Visualization:** Plotly, Plotly Express
*   **Configuration:** Pydantic-Settings, Python-Dotenv
*   **Application Persistence (Optional):** SQLAlchemy with SQLite (default) or PostgreSQL/MySQL (configurable via `DATABASE_URL`)
*   **Optional Libraries (for Advanced Features):**
    *   `scikit-learn`: For Segmentation, Encoding, Polynomial Features, etc.
    *   `statsmodels`, `prophet`: For Forecasting.
    *   `ydata-profiling`: For enhanced automated profiling.
    *   `passlib[bcrypt]`, `python-jose[cryptography]`: For Authentication.
    *   `pydantic[email]`: For email validation in configuration/models.
    *   `reportlab`, `python-pptx`: For PDF/PPTX report export.
    *   `APScheduler` / `Celery` / `python-rq` + `redis`: For background tasks/scheduling.
    *   `boto3`, `google-cloud-storage`: For cloud storage interaction.

## Project Structure
Use code with caution.
Markdown
powerbi-on-steroids-vnext/
├── .streamlit/
│ └── config.toml # Streamlit UI/Server config
├── app.py # Main Streamlit app entrypoint & navigation
├── pages/ # Streamlit page files (core workflow + advanced features)
│ ├── 0_📚_Data_Catalog.py # Placeholder
│ ├── 1_🔗_Connect_Profile.py
│ ├── 2_✨_Clean_Transform.py
│ ├── 3_🛠️_Explore_Engineer.py
│ ├── 4_📊_Analyze_Insight.py
│ ├── 5_📈_Report_Recommend.py
│ ├── 6_🚦_Quality_Monitor.py # Placeholder
│ ├── 7_⚙️_Settings.py # Placeholder
│ ├── auth_login.py # Placeholder
│ ├── auth_signup.py # Placeholder
│ ├── project_dashboard.py# Placeholder
│ └── _mocks.py # Mock backend functions for UI dev
├── frontend/ # Supporting frontend assets
│ ├── assets/ # Images, logos (e.g., Logo.png)
│ ├── components/ # Reusable Streamlit components (e.g., kpi_card.py)
│ ├── styles/ # Custom CSS (style.css)
│ └── utils.py # Frontend helper functions
├── backend/ # Core application logic
│ ├── init.py
│ ├── core/ # App config, logging setup
│ │ ├── init.py
│ │ ├── config.py
│ │ └── logging_config.py
│ ├── database/ # Database interaction (connectors, models, crud, session)
│ │ ├── init.py
│ │ ├── connectors.py
│ │ ├── models.py
│ │ ├── crud.py
│ │ └── session.py
│ ├── data_processing/ # Data loading, profiling, cleaning, transformation, features
│ │ ├── init.py
│ │ ├── loader.py # Optional: Separate file loading
│ │ ├── profiler.py
│ │ ├── cleaner.py
│ │ ├── transformer.py
│ │ └── feature_engineer.py
│ ├── analysis/ # Analysis execution, NLQ, insights, forecasting, segmentation
│ │ ├── init.py
│ │ ├── query_executor.py
│ │ ├── nlq_processor.py
│ │ ├── insight_generator.py
│ │ ├── forecaster.py
│ │ └── segmenter.py
│ ├── reporting/ # Report building, visualization, KPIs, exporting
│ │ ├── init.py
│ │ ├── visualizer.py
│ │ ├── report_builder.py
│ │ ├── kpi_manager.py
│ │ └── exporters.py # Placeholder for PDF/PPTX logic
│ ├── recommendations/ # Generating recommendations, handling feedback
│ │ ├── init.py
│ │ └── recommender.py
│ ├── llm/ # Centralized Gemini API interaction
│ │ ├── init.py
│ │ └── gemini_utils.py
│ ├── auth/ # Authentication logic (Optional)
│ │ ├── init.py
│ │ ├── models.py # Pydantic schemas
│ │ ├── security.py
│ │ └── service.py
│ ├── sandbox/ # Secure code execution (Placeholder - CRITICAL)
│ │ ├── init.py
│ │ └── executor.py
│ ├── tasks/ # Background task scheduling & jobs (Optional)
│ │ ├── init.py
│ │ ├── scheduler.py
│ │ └── jobs.py
│ └── utils.py # General backend utilities
├── tests/ # Unit and integration tests
│ └── ...
├── .gitignore # Files to ignore in Git
├── requirements.txt # Python dependencies
├── README.md # This file
└── .env # LOCAL ONLY: API keys, secrets (add to .gitignore!)
## Setup (Local Development)

1.  **Prerequisites:**
    *   Python 3.10 or higher.
    *   `pip` and `venv` (standard library).
    *   Git.
    *   **(If using MSSQL/Oracle):** Install necessary system-level ODBC drivers or Oracle Instant Client.
    *   **(If using background tasks):** Install Redis or RabbitMQ message broker.
    *   **(If using Docker sandbox):** Install Docker Desktop.

2.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd powerbi-on-steroids-vnext
    ```

3.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    # Windows:
    .\.venv\Scripts\activate
    # macOS/Linux:
    source .venv/bin/activate
    ```

4.  **Install dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    # Optional: Install extras if needed, e.g., for email validation
    # pip install "pydantic[email]"
    # Optional: Install testing requirements
    # pip install pytest pytest-cov pytest-mock
    ```

5.  **Configure Environment (`.env`):**
    *   Create a file named `.env` in the project root directory.
    *   Add your **required** Google Gemini API key:
        ```dotenv
        GEMINI_API_KEY=YOUR_ACTUAL_GEMINI_API_KEY_HERE
        ```
    *   **CRITICAL:** Generate a strong secret key for JWT tokens (e.g., run `openssl rand -hex 32` in your terminal) and add it:
        ```dotenv
        JWT_SECRET_KEY=your_generated_strong_32_byte_hex_secret_here
        ```
    *   Add any optional configurations needed (e.g., `DATABASE_URL` if not using the default SQLite, SMTP details, Redis URL). Refer to `backend/core/config.py` for available settings.
    *   **Ensure `.env` is listed in your `.gitignore` file.**

6.  **Create Initial Directories:**
    *   Manually create the default directories expected by the configuration (if they don't exist):
        ```bash
        mkdir uploads
        mkdir reports_output
        ```

7.  **Initialize Application Database (First Run):**
    *   The application attempts to create the necessary tables in the database defined by `DATABASE_URL` (defaults to `app_persistence.db`) on first startup. Ensure the path is writable.

8.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

9.  Access the application in your browser (usually `http://localhost:8501`).

## Deployment (Hugging Face Spaces - Streamlit SDK Recommended)

1.  **Create HF Space:** Use the "Streamlit" SDK option. Select appropriate hardware (CPU Upgrade recommended).
2.  **Push Code:** Push your repository to the Space (ensure `.env` is NOT committed).
3.  **Set Secrets:** In Space Settings -> Secrets, add:
    *   `GEMINI_API_KEY`: Your Google Gemini API key.
    *   `JWT_SECRET_KEY`: Your strong, generated JWT secret key.
    *   Add other secrets corresponding to your `.env` variables if needed (e.g., `DATABASE_URL`, `SMTP_PASSWORD`).
4.  **Dependencies:** HF Spaces will install packages from `requirements.txt`. Ensure necessary system dependencies (like ODBC drivers if using MSSQL) are handled (this is easier with the native SDK, but complex dependencies might still require a Docker deployment).
5.  The Space should build and run `app.py` automatically.

## Running Tests (Local)

(Assuming `pytest` and related libraries are installed)

```bash
# Activate your virtual environment first
pytest tests/
# Run with coverage:
# pytest --cov=backend --cov=frontend tests/ -v
Use code with caution.
Contributing
(Detail how others can contribute, coding standards, PR process, etc. - if applicable)
License
(Specify your project's license, e.g., MIT, Apache 2.0. Defaulting to MIT if unspecified is common).
MIT License

Copyright (c) [Year] [Your Name/Organization]

Permission is hereby granted... (Full MIT license text)"# DataAnalystAI" 
