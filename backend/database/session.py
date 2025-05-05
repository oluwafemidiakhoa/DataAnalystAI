# backend/database/session.py
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Generator # Import Generator for type hint

# Get DB URL from config
try:
    from backend.core.config import settings # Get DB URL from config
    DB_URL = settings.database_url if settings else None
except (ImportError, RuntimeError, AttributeError):
     logging.error("Failed to import settings for DB session.", exc_info=True)
     DB_URL = None


logger = logging.getLogger(__name__)
engine = None
SessionLocal = None

if DB_URL:
    try:
        # Add connect_args if needed, e.g., for SQLite with Streamlit
        connect_args = {"check_same_thread": False} if DB_URL.startswith("sqlite") else {}
        engine = create_engine(DB_URL, connect_args=connect_args, pool_pre_ping=True, echo=False) # Set echo=True for SQL logging
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        logger.info("Database session factory configured.")
    except Exception as e:
        logger.error(f"Failed to create database engine or session factory for URL '{DB_URL[:30]}...': {e}", exc_info=True)
        # Ensure SessionLocal remains None if setup fails
        engine = None
        SessionLocal = None
else:
    logger.error("Database URL not configured in settings. Cannot create session factory.")


@contextmanager
def get_db_session() -> Generator[Session, None, None]: # Use Generator type hint
    """Provides a transactional database session scope using a context manager."""
    if SessionLocal is None:
        logger.error("SessionLocal not configured. Cannot get DB session.")
        # Raising error might be better than yielding None if DB is critical
        raise RuntimeError("Database session factory not initialized due to configuration errors.")

    db: Session = SessionLocal() # Create a new session
    logger.debug(f"DB Session {id(db)} opened.")
    try:
        yield db # Provide the session to the 'with' block
        db.commit() # Commit on successful block execution
        logger.debug(f"DB Session {id(db)} committed.")
    except Exception as e:
        logger.error(f"DB Session {id(db)} error occurred, rolling back.", exc_info=True)
        db.rollback() # Rollback on error
        raise # Re-raise the exception after rollback
    finally:
        logger.debug(f"DB Session {id(db)} closed.")
        db.close() # Always close session

# Optional: Function to create tables (might live elsewhere, e.g., main app or migrations)
# def init_db():
#     if engine:
#         from .models import Base # Import Base here if needed
#         logger.info("Initializing database tables...")
#         Base.metadata.create_all(bind=engine)
#         logger.info("Database tables initialized.")
#     else:
#         logger.error("Cannot initialize DB tables: Engine not created.")