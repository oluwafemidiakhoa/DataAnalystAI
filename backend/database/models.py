# backend/database/models.py
import datetime
import logging
from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, ForeignKey, Text, JSON, Boolean, Float,
    Enum # Import Enum type
)
from sqlalchemy.orm import relationship, DeclarativeBase, Mapped, mapped_column, Session
from sqlalchemy.sql import func
from sqlalchemy.engine import Engine # Import Engine for type hint
from typing import List, Optional, Dict, Any # Import typing helpers

# --- Setup Logger ---
# Define logger here if create_db_and_tables is called directly from this module during setup
logger = logging.getLogger(__name__)

# --- Base Model Definition ---
class Base(DeclarativeBase):
    """Base class for SQLAlchemy models with type hints."""
    pass

# --- User Authentication & Project Management Models (#13) ---

class User(Base):
    """Model for user accounts."""
    __tablename__ = 'users'
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    username: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=True) # Optional username
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(200))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False) # For admin roles
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    projects: Mapped[List["Project"]] = relationship("Project", secondary="project_users", back_populates="users") # Many-to-Many

class Project(Base):
    """Model for organizing analyses into projects."""
    __tablename__ = 'projects'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text)
    owner_id: Mapped[Optional[int]] = mapped_column(ForeignKey('users.id')) # Can be nullable if projects can be unassigned initially?
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime(timezone=True), onupdate=func.now())

    users: Mapped[List["User"]] = relationship("User", secondary="project_users", back_populates="projects") # Many-to-Many
    owner: Mapped[Optional["User"]] = relationship() # One-to-Many from User (owner) perspective implicitly handled if owner_id is FK

    # Relationships to other project-specific entities
    data_sources: Mapped[List["DataSource"]] = relationship(back_populates="project")
    reports: Mapped[List["Report"]] = relationship(back_populates="project")
    kpis: Mapped[List["KPI"]] = relationship(back_populates="project")
    etl_pipelines: Mapped[List["EtlPipeline"]] = relationship(back_populates="project")
    recommendations: Mapped[List["Recommendation"]] = relationship(back_populates="project")

# Association Table for Many-to-Many between Project and User
class ProjectUser(Base):
    __tablename__ = 'project_users'
    project_id: Mapped[int] = mapped_column(ForeignKey('projects.id'), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'), primary_key=True)
    role: Mapped[str] = mapped_column(String(50), default='viewer') # e.g., 'owner', 'editor', 'viewer'

# --- Data Catalog & Lineage Models (#3) ---

class DataSource(Base):
    """Model to store connection information and catalog metadata."""
    __tablename__ = 'data_sources'
    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[Optional[int]] = mapped_column(ForeignKey('projects.id')) # Link to project
    name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False) # 'postgresql', 'file', 'mongodb', 's3', 'gcs'
    connection_details: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON) # Store encrypted credentials/path/URI/bucket
    description: Mapped[Optional[str]] = mapped_column(Text) # AI or User generated description
    tags: Mapped[Optional[List[str]]] = mapped_column(JSON) # List of tags (e.g., 'PII', 'Sales', 'Finance')
    schema_cache: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON) # Cache schema
    last_profiled_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    created_by_id: Mapped[Optional[int]] = mapped_column(ForeignKey('users.id')) # Track who added it

    project: Mapped[Optional["Project"]] = relationship(back_populates="data_sources")
    created_by: Mapped[Optional["User"]] = relationship()
    # Relationships for Lineage
    lineage_inputs: Mapped[List["DataLineage"]] = relationship("DataLineage", foreign_keys="[DataLineage.output_datasource_id]", back_populates="output_datasource")
    lineage_outputs: Mapped[List["DataLineage"]] = relationship("DataLineage", foreign_keys="[DataLineage.input_datasource_id]", back_populates="input_datasource")
    quality_rules: Mapped[List["DataQualityRule"]] = relationship(back_populates="data_source")


class DataLineage(Base):
    """Model to track basic data lineage between sources/processes."""
    __tablename__ = 'data_lineage'
    id: Mapped[int] = mapped_column(primary_key=True)
    input_datasource_id: Mapped[Optional[int]] = mapped_column(ForeignKey('data_sources.id')) # Source data
    output_datasource_id: Mapped[Optional[int]] = mapped_column(ForeignKey('data_sources.id')) # Resulting data (could be same source if modified in place conceptually)
    process_type: Mapped[str] = mapped_column(String(100)) # e.g., 'cleaning', 'transformation', 'feature_engineering', 'report_generation'
    process_details: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON) # Store parameters or code snippet hash?
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    input_datasource: Mapped[Optional["DataSource"]] = relationship("DataSource", foreign_keys=[input_datasource_id], back_populates="lineage_outputs")
    output_datasource: Mapped[Optional["DataSource"]] = relationship("DataSource", foreign_keys=[output_datasource_id], back_populates="lineage_inputs")

# --- Data Quality Models (#2) ---

class DataQualityRule(Base):
    """Model to define data quality rules."""
    __tablename__ = 'data_quality_rules'
    id: Mapped[int] = mapped_column(primary_key=True)
    data_source_id: Mapped[int] = mapped_column(ForeignKey('data_sources.id'), nullable=False)
    rule_name: Mapped[str] = mapped_column(String(200), nullable=False)
    rule_type: Mapped[str] = mapped_column(String(50), nullable=False) # e.g., 'not_null', 'unique', 'min_value', 'max_value', 'pattern', 'custom_sql'
    column_name: Mapped[Optional[str]] = mapped_column(String(255)) # Column the rule applies to (optional for table-level rules)
    rule_parameters: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False) # e.g., {'min': 0}, {'pattern': '^[A-Z]+$'}, {'query': 'SELECT...'}
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    data_source: Mapped["DataSource"] = relationship(back_populates="quality_rules")
    violations: Mapped[List["DataQualityViolation"]] = relationship(back_populates="rule")


class DataQualityViolation(Base):
    """Model to log data quality rule violations."""
    __tablename__ = 'data_quality_violations'
    id: Mapped[int] = mapped_column(primary_key=True)
    rule_id: Mapped[int] = mapped_column(ForeignKey('data_quality_rules.id'), nullable=False)
    check_timestamp: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    status: Mapped[str] = mapped_column(String(20), default='failed') # 'failed', 'passed' (optional to log passes)
    violation_count: Mapped[Optional[int]] = mapped_column(Integer) # Number of rows failing the rule
    violation_details: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON) # e.g., sample violating rows/values

    rule: Mapped["DataQualityRule"] = relationship(back_populates="violations")


# --- ETL Pipeline Models (#4 - Optional) ---

class EtlPipeline(Base):
    """Model to store definition of saved ETL/Cleaning pipelines."""
    __tablename__ = 'etl_pipelines'
    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[Optional[int]] = mapped_column(ForeignKey('projects.id'))
    name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text)
    steps: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, nullable=False) # Store sequence of steps (type, params)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    created_by_id: Mapped[Optional[int]] = mapped_column(ForeignKey('users.id'))

    project: Mapped[Optional["Project"]] = relationship(back_populates="etl_pipelines")
    created_by: Mapped[Optional["User"]] = relationship()


# --- Reporting & KPI Models (Enhanced) ---

class Report(Base):
    """Model to store generated reports/dashboards."""
    __tablename__ = 'reports'
    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[Optional[int]] = mapped_column(ForeignKey('projects.id'))
    name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    report_type: Mapped[str] = mapped_column(String(50)) # 'dashboard', 'narrative_summary', 'forecast', 'segmentation'
    configuration: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON) # Store dashboard layout, chart configs, forecast params etc.
    narrative_content: Mapped[Optional[str]] = mapped_column(Text) # Store generated narrative text
    # Store generated artifacts? Links or BLOBs? Linking might be better.
    # artifact_links: Mapped[Optional[Dict[str, str]]] = mapped_column(JSON) # e.g., {'pdf': 'path/to/report.pdf', 'chart1_json': 'path/to/chart.json'}
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    created_by_id: Mapped[Optional[int]] = mapped_column(ForeignKey('users.id'))

    project: Mapped[Optional["Project"]] = relationship(back_populates="reports")
    created_by: Mapped[Optional["User"]] = relationship()


class KPI(Base):
    """Model to store KPI definitions."""
    __tablename__ = 'kpis'
    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[Optional[int]] = mapped_column(ForeignKey('projects.id'))
    name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text)
    calculation_logic: Mapped[Optional[str]] = mapped_column(Text) # How it's calculated (SQL, Pandas, or notes)
    data_source_id: Mapped[Optional[int]] = mapped_column(ForeignKey('data_sources.id')) # Optional link to primary source data
    target_value: Mapped[Optional[float]] = mapped_column(Float)
    alert_threshold_upper: Mapped[Optional[float]] = mapped_column(Float) # Optional upper bound alert
    alert_threshold_lower: Mapped[Optional[float]] = mapped_column(Float) # Optional lower bound alert
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    created_by_id: Mapped[Optional[int]] = mapped_column(ForeignKey('users.id'))

    project: Mapped[Optional["Project"]] = relationship(back_populates="kpis")
    created_by: Mapped[Optional["User"]] = relationship()
    data_source: Mapped[Optional["DataSource"]] = relationship()
    historical_values: Mapped[List["KpiValue"]] = relationship(back_populates="kpi")


class KpiValue(Base):
    """Model to store historical KPI values."""
    __tablename__ = 'kpi_values'
    id: Mapped[int] = mapped_column(primary_key=True)
    kpi_id: Mapped[int] = mapped_column(ForeignKey('kpis.id'), nullable=False)
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), index=True, server_default=func.now())
    value: Mapped[float] = mapped_column(Float, nullable=False)
    # Optional: Store calculation context/parameters if they can vary
    # calculation_context: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    kpi: Mapped["KPI"] = relationship(back_populates="historical_values")


# --- Recommendation & Feedback Models (#12) ---

class Recommendation(Base):
    """Model to store generated recommendations."""
    __tablename__ = 'recommendations'
    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[Optional[int]] = mapped_column(ForeignKey('projects.id'))
    recommendation_text: Mapped[str] = mapped_column(Text, nullable=False)
    rationale: Mapped[Optional[str]] = mapped_column(Text)
    confidence: Mapped[Optional[str]] = mapped_column(String(50)) # 'High', 'Medium', 'Low'
    impact_estimate: Mapped[Optional[str]] = mapped_column(String(100)) # Qualitative or Quantitative note
    source_analysis_id: Mapped[Optional[int]] = mapped_column(ForeignKey('reports.id')) # Optional link to report that generated it
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    project: Mapped[Optional["Project"]] = relationship(back_populates="recommendations")
    feedback: Mapped[List["RecommendationFeedback"]] = relationship(back_populates="recommendation")


class RecommendationFeedback(Base):
    """Model to store user feedback on recommendations."""
    __tablename__ = 'recommendation_feedback'
    id: Mapped[int] = mapped_column(primary_key=True)
    recommendation_id: Mapped[int] = mapped_column(ForeignKey('recommendations.id'), nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'), nullable=False) # Who gave feedback
    rating: Mapped[Optional[str]] = mapped_column(String(50)) # e.g., 'Helpful', 'Not Helpful', 'Actioned'
    comment: Mapped[Optional[str]] = mapped_column(Text)
    feedback_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    recommendation: Mapped["Recommendation"] = relationship(back_populates="feedback")
    user: Mapped["User"] = relationship()


# --- Database Setup Function (Keep as before, called externally) ---
# Remember to configure your database connection URL (e.g., in config.py or env vars)

def get_engine(db_url: str) -> Engine:
    """Creates the SQLAlchemy engine."""
    # Add connect_args if needed, e.g., for SQLite with Streamlit
    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    logger.info(f"Creating database engine for URL: {db_url[:30]}...") # Log prefix only
    return create_engine(db_url, connect_args=connect_args, echo=False) # Set echo=True for SQL logging

def create_db_and_tables(engine: Engine):
    """Creates database tables based on the defined models."""
    logger.info("Attempting to create database tables...")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables verified/created successfully.")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}", exc_info=True)
        raise # Re-raise the exception after logging

# Example call (usually done once at application startup):
# from backend.core.config import settings
# if settings.database_url:
#     db_engine = get_engine(settings.database_url)
#     create_db_and_tables(db_engine)