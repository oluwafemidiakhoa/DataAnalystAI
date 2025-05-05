# backend/database/crud.py
# Create, Read, Update, Delete operations for SQLAlchemy models

import logging
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError # Import IntegrityError
from sqlalchemy import select, update, delete # Use select, update, delete for SQLAlchemy 2.0 style queries
from typing import List, Optional, Dict, Any, Type
import datetime

# --- Model Imports ---
# Import specific models needed
try:
    from . import models # Relative import of models module
    # Explicitly import classes needed for annotations if type checkers struggle
    from .models import (
        Base, User, Project, ProjectUser, DataSource, DataLineage,
        DataQualityRule, DataQualityViolation, EtlPipeline, Report,
        KPI, KpiValue, Recommendation, RecommendationFeedback
    )
    MODELS_AVAILABLE = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to import SQLAlchemy models from .models in crud.py: {e}", exc_info=True)
    raise ImportError(f"Could not import database models: {e}") from e

logger = logging.getLogger(__name__) # Define logger for the module scope

# --- Generic Helper ---
def get_object_by_id(db: Session, model: Type[Base], object_id: int) -> Optional[Base]:
    """Generic function to get any model instance by its ID."""
    logger.debug(f"Querying for {model.__name__} with ID: {object_id}")
    try:
        statement = select(model).where(model.id == object_id)
        result = db.execute(statement).scalar_one_or_none()
        return result
    except SQLAlchemyError as e:
        logger.error(f"DB error retrieving {model.__name__} ID {object_id}: {e}", exc_info=True)
        raise # Re-raise DB errors
    except Exception as e:
        logger.error(f"Unexpected error retrieving {model.__name__} ID {object_id}: {e}", exc_info=True)
        raise

# --- Commit Helper ---
def _commit_and_refresh(db: Session, db_object: Base):
    """Commits the session and refreshes the object."""
    try:
        db.commit()
        db.refresh(db_object)
    except SQLAlchemyError as e:
        logger.error(f"Database commit/refresh error: {e}", exc_info=True)
        db.rollback()
        raise
    except Exception as e:
        logger.error(f"Unexpected error during commit/refresh: {e}", exc_info=True)
        db.rollback()
        raise


# --- User CRUD (#13) ---

def get_user(db: Session, user_id: int) -> Optional[User]:
    """Retrieves a user by ID."""
    return get_object_by_id(db, User, user_id)

def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Retrieves a user by email."""
    logger.debug(f"Querying for user by email: {email}")
    try:
        statement = select(User).where(User.email == email)
        return db.execute(statement).scalar_one_or_none()
    except Exception as e: logger.error(f"Error retrieving user by email '{email}': {e}", exc_info=True); raise

# Expects HASHED password
def create_user(db: Session, email: str, hashed_password: str, full_name: Optional[str] = None, username: Optional[str] = None) -> User:
    """Creates a new user with a pre-hashed password."""
    logger.info(f"Attempting to create user: {email}")
    db_user = User(email=email, hashed_password=hashed_password, full_name=full_name, username=username)
    try:
        db.add(db_user)
        _commit_and_refresh(db, db_user)
        logger.info(f"Successfully created user '{email}' with ID {db_user.id}")
        return db_user
    except IntegrityError as e:
         logger.warning(f"Integrity error (likely duplicate email/username) creating user '{email}': {e}")
         db.rollback()
         raise ValueError(f"User with email '{email}' or username '{username}' likely already exists.") from e
    except Exception as e: logger.error(f"Error creating user '{email}': {e}", exc_info=True); db.rollback(); raise

# --- Project CRUD (#13) ---

def create_project(db: Session, name: str, description: Optional[str] = None, owner_id: Optional[int] = None) -> Project:
    """Creates a new project."""
    logger.info(f"Attempting to create project: {name}")
    db_project = Project(name=name, description=description, owner_id=owner_id)
    try:
        db.add(db_project); _commit_and_refresh(db, db_project)
        logger.info(f"Successfully created project '{name}' with ID {db_project.id}")
        return db_project
    except Exception as e: logger.error(f"Error creating project '{name}': {e}", exc_info=True); db.rollback(); raise

def get_project(db: Session, project_id: int) -> Optional[Project]:
    """Retrieves a project by ID."""
    return get_object_by_id(db, Project, project_id)

def get_projects_for_user(db: Session, user_id: int) -> List[Project]:
    """Retrieves projects associated with a user via the ProjectUser association."""
    logger.debug(f"Querying projects for user ID: {user_id}")
    try:
        statement = select(Project).join(ProjectUser).where(ProjectUser.user_id == user_id)
        results = db.execute(statement).scalars().unique().all() # Use unique() if duplicates possible
        return list(results)
    except Exception as e: logger.error(f"Error retrieving projects for user {user_id}: {e}", exc_info=True); raise

# --- DataSource CRUD (#3) ---

def create_data_source(db: Session, name: str, source_type: str, connection_details: Optional[Dict] = None, description: Optional[str] = None, tags: Optional[List[str]] = None, project_id: Optional[int] = None, created_by_id: Optional[int] = None) -> DataSource:
    """Creates a new data source entry."""
    logger.info(f"Attempting to create data source: {name} (Type: {source_type})")
    # SECURITY: Connection details should be encrypted before this point!
    db_ds = DataSource(name=name, source_type=source_type, connection_details=connection_details, description=description, tags=tags, project_id=project_id, created_by_id=created_by_id)
    try:
        db.add(db_ds); _commit_and_refresh(db, db_ds)
        logger.info(f"Successfully created data source '{name}' with ID {db_ds.id}")
        return db_ds
    except Exception as e: logger.error(f"Error creating data source '{name}': {e}", exc_info=True); db.rollback(); raise

def get_data_source(db: Session, data_source_id: int) -> Optional[DataSource]:
    """Retrieves a data source by ID."""
    return get_object_by_id(db, DataSource, data_source_id)

def get_all_datasources(db: Session, project_id: Optional[int] = None) -> List[DataSource]:
    """Retrieves all data sources, optionally filtered by project."""
    logger.debug(f"Querying all datasources (Project ID: {project_id or 'All'})")
    try:
        statement = select(DataSource)
        if project_id is not None:
             statement = statement.where(DataSource.project_id == project_id)
        return list(db.execute(statement).scalars().all())
    except Exception as e: logger.error(f"Error getting datasources: {e}", exc_info=True); raise

def update_data_source_schema(db: Session, data_source_id: int, schema_cache: Dict) -> Optional[DataSource]:
    """Updates the schema cache and last profiled time for a data source."""
    logger.info(f"Updating schema cache for data source ID: {data_source_id}")
    db_ds = get_data_source(db, data_source_id)
    if db_ds:
        try:
            db_ds.schema_cache = schema_cache
            db_ds.last_profiled_at = datetime.datetime.now(datetime.timezone.utc)
            _commit_and_refresh(db, db_ds)
            logger.info(f"Schema cache updated for data source '{db_ds.name}'.")
            return db_ds
        except Exception as e: logger.error(f"Error updating schema for DS {data_source_id}: {e}", exc_info=True); db.rollback(); raise
    else:
        logger.warning(f"Data source ID {data_source_id} not found for schema update.")
        return None

# --- DataQualityRule CRUD (#2) ---

def create_quality_rule(db: Session, data_source_id: int, rule_name: str, rule_type: str, rule_parameters: Dict, column_name: Optional[str] = None) -> DataQualityRule:
    # Assuming project_id link comes via data_source_id -> project_id if needed
    logger.info(f"Attempting to create quality rule '{rule_name}' for DS ID {data_source_id}")
    db_rule = DataQualityRule(data_source_id=data_source_id, rule_name=rule_name, rule_type=rule_type, rule_parameters=rule_parameters, column_name=column_name)
    try:
        db.add(db_rule); _commit_and_refresh(db, db_rule)
        logger.info(f"Successfully created quality rule '{rule_name}' with ID {db_rule.id}")
        return db_rule
    except Exception as e: logger.error(f"Error creating quality rule '{rule_name}': {e}", exc_info=True); db.rollback(); raise

def get_rules_for_datasource(db: Session, data_source_id: int, only_active: bool = True) -> List[DataQualityRule]:
    """Gets rules for a specific data source."""
    logger.debug(f"Querying rules for DS ID: {data_source_id} (Active Only: {only_active})")
    try:
        statement = select(DataQualityRule).where(DataQualityRule.data_source_id == data_source_id)
        if only_active:
            statement = statement.where(DataQualityRule.is_active == True)
        return list(db.execute(statement).scalars().all())
    except Exception as e: logger.error(f"Error getting rules for DS {data_source_id}: {e}", exc_info=True); raise

# --- DataQualityViolation CRUD (#2) ---

def log_quality_violation(db: Session, rule_id: int, status: str = 'failed', count: Optional[int] = None, details: Optional[Dict] = None) -> DataQualityViolation:
    """Logs a data quality check result (violation or pass)."""
    logger.info(f"Logging quality check result for rule ID {rule_id}. Status: {status}")
    db_violation = DataQualityViolation(rule_id=rule_id, status=status, violation_count=count, violation_details=details)
    try:
        db.add(db_violation); _commit_and_refresh(db, db_violation)
        logger.info(f"Successfully logged violation/result ID {db_violation.id} for rule {rule_id}")
        return db_violation
    except Exception as e: logger.error(f"Error logging violation for rule {rule_id}: {e}", exc_info=True); db.rollback(); raise

def get_violation_history(db: Session, data_source_id: Optional[int] = None, rule_id: Optional[int] = None, limit: int = 50) -> List[DataQualityViolation]:
     """Gets recent violation history, optionally filtered."""
     logger.debug(f"Querying violation history (DS_ID: {data_source_id}, RuleID: {rule_id}, Limit: {limit})")
     try:
          statement = select(DataQualityViolation).order_by(DataQualityViolation.check_timestamp.desc())
          if rule_id:
               statement = statement.where(DataQualityViolation.rule_id == rule_id)
          elif data_source_id:
               # Join needed if filtering by datasource ID
               statement = statement.join(DataQualityRule).where(DataQualityRule.data_source_id == data_source_id)
          statement = statement.limit(limit)
          # Eager load the related rule for display purposes? Optional performance tweak.
          # from sqlalchemy.orm import joinedload
          # statement = statement.options(joinedload(DataQualityViolation.rule))
          return list(db.execute(statement).scalars().all())
     except Exception as e: logger.error(f"Error getting violation history: {e}", exc_info=True); raise

# --- EtlPipeline CRUD (#4) ---

def save_etl_pipeline(db: Session, name: str, steps: List[Dict], description: Optional[str] = None, project_id: Optional[int] = None, created_by_id: Optional[int] = None) -> EtlPipeline:
    # Add update logic if needed
    logger.info(f"Attempting to save ETL pipeline: {name}")
    db_pipeline = EtlPipeline(name=name, steps=steps, description=description, project_id=project_id, created_by_id=created_by_id)
    try: db.add(db_pipeline); _commit_and_refresh(db, db_pipeline); logger.info(f"Saved ETL pipeline '{name}' ID {db_pipeline.id}"); return db_pipeline
    except Exception as e: logger.error(f"Error saving pipeline '{name}': {e}", exc_info=True); db.rollback(); raise

def get_etl_pipeline(db: Session, pipeline_id: int) -> Optional[EtlPipeline]:
    return get_object_by_id(db, EtlPipeline, pipeline_id)

# --- Report CRUD ---

def save_report(db: Session, name: str, report_type: str, configuration: Optional[Dict] = None, narrative_content: Optional[str] = None, project_id: Optional[int] = None, created_by_id: Optional[int] = None) -> Report:
    # Add update logic if needed
    logger.info(f"Attempting to save report: {name} (Type: {report_type})")
    db_report = Report(name=name, report_type=report_type, configuration=configuration, narrative_content=narrative_content, project_id=project_id, created_by_id=created_by_id)
    try: db.add(db_report); _commit_and_refresh(db, db_report); logger.info(f"Saved report '{name}' ID {db_report.id}"); return db_report
    except Exception as e: logger.error(f"Error saving report '{name}': {e}", exc_info=True); db.rollback(); raise

def get_report(db: Session, report_id: int) -> Optional[Report]:
    return get_object_by_id(db, Report, report_id)

# --- KPI CRUD (Enhanced) ---

def create_or_update_kpi(db: Session, name: str, description: Optional[str] = None, calculation_logic: Optional[str] = None, target_value: Optional[float] = None, alert_upper: Optional[float] = None, alert_lower: Optional[float] = None, is_active: bool = True, project_id: Optional[int] = None, created_by_id: Optional[int] = None, data_source_id: Optional[int] = None) -> KPI:
    """Creates a new KPI or updates an existing one by name."""
    logger.info(f"Attempting create/update KPI: {name}")
    # Use scalar_one_or_none for SQLAlchemy 2.0 style
    stmt = select(KPI).where(KPI.name == name)
    if project_id: # KPIs should likely be project-specific if projects exist
        stmt = stmt.where(KPI.project_id == project_id)
    db_kpi = db.execute(stmt).scalar_one_or_none()

    if db_kpi:
        logger.info(f"Updating existing KPI '{name}' (ID: {db_kpi.id})")
        # Update only fields that are provided (not None) to avoid accidental nulling?
        # Or assume caller provides the full desired state? Assuming full update for now.
        db_kpi.description = description
        db_kpi.calculation_logic = calculation_logic
        db_kpi.target_value = target_value
        db_kpi.alert_threshold_upper = alert_upper
        db_kpi.alert_threshold_lower = alert_lower
        db_kpi.is_active = is_active
        db_kpi.project_id = project_id # Allow changing project?
        db_kpi.data_source_id = data_source_id
        # created_by_id typically doesn't change
    else:
        logger.info(f"Creating new KPI '{name}'")
        db_kpi = KPI(
            name=name, description=description, calculation_logic=calculation_logic,
            target_value=target_value, alert_threshold_upper=alert_upper, alert_threshold_lower=alert_lower,
            is_active=is_active, project_id=project_id, created_by_id=created_by_id, data_source_id=data_source_id
        )
        db.add(db_kpi)
    try:
        _commit_and_refresh(db, db_kpi) # Use helper
        logger.info(f"Successfully saved KPI '{name}' with ID {db_kpi.id}")
        return db_kpi
    except Exception as e: logger.error(f"Error saving KPI '{name}': {e}", exc_info=True); db.rollback(); raise

def get_kpi(db: Session, kpi_id: int) -> Optional[KPI]:
    return get_object_by_id(db, KPI, kpi_id)

def get_kpi_by_name(db: Session, name: str, project_id: Optional[int] = None) -> Optional[KPI]:
    """Retrieves a KPI by its name, optionally filtered by project."""
    logger.debug(f"Querying for KPI by name: {name} (Project: {project_id or 'Any'})") # Corrected log formatting
    try:
        statement = select(KPI).where(KPI.name == name)
        if project_id is not None:
             statement = statement.where(KPI.project_id == project_id)
        return db.execute(statement).scalar_one_or_none() # Use scalar_one_or_none
    except Exception as e:
        logger.error(f"Error retrieving KPI '{name}': {e}", exc_info=True)
        raise

def get_active_kpis(db: Session, project_id: Optional[int] = None) -> List[KPI]:
    """Retrieves all active KPIs, optionally filtered by project."""
    logger.debug(f"Querying active KPIs (Project ID: {project_id or 'All'})")
    try:
        statement = select(KPI).where(KPI.is_active == True)
        if project_id is not None:
            statement = statement.where(KPI.project_id == project_id)
        return list(db.execute(statement).scalars().all()) # Use list() for clarity
    except Exception as e: logger.error(f"Error retrieving active KPIs: {e}", exc_info=True); raise

def save_kpi_value(db: Session, kpi_id: int, value: float) -> KpiValue:
    """Saves a new historical value for a KPI."""
    logger.debug(f"Saving value {value} for KPI ID {kpi_id}")
    db_value = KpiValue(kpi_id=kpi_id, value=value)
    try:
        db.add(db_value); _commit_and_refresh(db, db_value)
        return db_value
    except Exception as e: logger.error(f"Error saving KPI value for ID {kpi_id}: {e}", exc_info=True); db.rollback(); raise

def get_latest_kpi_value(db: Session, kpi_id: int) -> Optional[KpiValue]:
    """Retrieves the most recent historical value for a KPI."""
    logger.debug(f"Querying latest value for KPI ID {kpi_id}")
    try:
        statement = select(KpiValue).where(KpiValue.kpi_id == kpi_id).order_by(KpiValue.timestamp.desc()).limit(1)
        return db.execute(statement).scalar_one_or_none()
    except Exception as e: logger.error(f"Error getting latest KPI value for ID {kpi_id}: {e}", exc_info=True); raise

# --- Recommendation CRUD (#12) ---

def create_recommendation(db: Session, text: str, rationale: Optional[str] = None, confidence: Optional[str] = None, impact: Optional[str] = None, project_id: Optional[int] = None, source_report_id: Optional[int] = None) -> Recommendation:
    logger.info("Attempting to create recommendation")
    db_rec = Recommendation(recommendation_text=text, rationale=rationale, confidence=confidence, impact_estimate=impact, project_id=project_id, source_analysis_id=source_report_id)
    try:
        db.add(db_rec); _commit_and_refresh(db, db_rec)
        logger.info(f"Created recommendation ID {db_rec.id}")
        return db_rec
    except Exception as e: logger.error(f"Error creating recommendation: {e}", exc_info=True); db.rollback(); raise

def save_recommendation_feedback(db: Session, recommendation_id: int, user_id: int, rating: Optional[str] = None, comment: Optional[str] = None) -> RecommendationFeedback:
    logger.info(f"Saving feedback for rec ID {recommendation_id} by user ID {user_id}")
    db_feedback = RecommendationFeedback(recommendation_id=recommendation_id, user_id=user_id, rating=rating, comment=comment)
    try:
        db.add(db_feedback); _commit_and_refresh(db, db_feedback)
        logger.info(f"Saved feedback ID {db_feedback.id}")
        return db_feedback
    except Exception as e: logger.error(f"Error saving recommendation feedback: {e}", exc_info=True); db.rollback(); raise

def get_recent_recommendation_feedback(db: Session, project_id: int, limit: int = 10) -> List[RecommendationFeedback]:
     """Gets recent feedback for recommendations within a project."""
     logger.debug(f"Querying recent feedback for project ID {project_id} (limit {limit})")
     try:
          # Eager load related recommendation and user for display efficiency
          from sqlalchemy.orm import joinedload
          statement = select(RecommendationFeedback)\
               .options(joinedload(RecommendationFeedback.recommendation), joinedload(RecommendationFeedback.user))\
               .join(RecommendationFeedback.recommendation)\
               .where(Recommendation.project_id == project_id)\
               .order_by(RecommendationFeedback.feedback_at.desc())\
               .limit(limit)
          return list(db.execute(statement).scalars().unique().all()) # Use unique() due to joins
     except Exception as e: logger.error(f"Error getting recent feedback for project {project_id}: {e}", exc_info=True); raise


# --- Add more specific CRUD functions as needed ---
# e.g., get_datasource_by_name, get_project_users, update_user_profile, etc.