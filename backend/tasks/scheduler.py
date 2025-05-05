# backend/tasks/scheduler.py
# Setup and manage scheduled tasks (e.g., using APScheduler for in-process)

import logging
from typing import Optional

# --- Choose ONE scheduler library ---

# Option A: APScheduler (Simpler, In-Process)
APS_AVAILABLE = False
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.jobstores.memory import MemoryJobStore
    # from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore # If persistence needed
    # from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor
    APS_AVAILABLE = True
except ImportError:
    logging.warning("APScheduler not installed (`pip install apscheduler`). Background task scheduling disabled.")

# Option B: Celery/RQ (More Scalable, Requires External Broker like Redis)
# Requires different setup involving worker processes and broker connection.
# Example placeholder:
CELERY_APP = None
# try:
#     from .celery_app import app as celery_app # Assuming celery app defined elsewhere
#     CELERY_APP = celery_app
# except ImportError:
#     logging.info("Celery not configured/found.")


logger = logging.getLogger(__name__)

# --- Global Scheduler Instance (using APScheduler example) ---
scheduler: Optional[BackgroundScheduler] = None

def initialize_scheduler():
    """Initializes and starts the background task scheduler."""
    global scheduler
    if not APS_AVAILABLE:
        logger.warning("Cannot initialize scheduler: APScheduler library not available.")
        return

    if scheduler and scheduler.running:
        logger.warning("Scheduler already running.")
        return

    logger.info("Initializing background scheduler (APScheduler)...")
    try:
        jobstores = {'default': MemoryJobStore()} # Basic in-memory store
        # executors = {'default': ThreadPoolExecutor(10)} # Max 10 concurrent threads
        scheduler = BackgroundScheduler(
            jobstores=jobstores,
            # executors=executors,
            timezone='UTC' # Use UTC for consistency
        )
        scheduler.start()
        logger.info("Background scheduler started successfully.")

        # --- Add Default Scheduled Jobs Here ---
        # Example: Schedule a data quality check job to run daily
        try:
             from .jobs import run_all_quality_checks_job # Import job function
             scheduler.add_job(
                  run_all_quality_checks_job,
                  trigger=CronTrigger(hour=2, minute=0), # Run daily at 2 AM UTC
                  id='daily_quality_check',
                  name='Daily Data Quality Checks',
                  replace_existing=True
             )
             logger.info("Scheduled 'daily_quality_check' job.")
        except ImportError:
             logger.error("Could not import or schedule 'run_all_quality_checks_job'.")
        except Exception as job_err:
             logger.error(f"Failed to schedule default job 'daily_quality_check': {job_err}", exc_info=True)

        # Add other default jobs (e.g., proactive insight generation)

    except Exception as e:
        logger.error(f"Failed to start background scheduler: {e}", exc_info=True)
        scheduler = None # Ensure scheduler is None if start fails

def shutdown_scheduler():
    """Shuts down the scheduler gracefully."""
    global scheduler
    if scheduler and scheduler.running:
        logger.info("Shutting down background scheduler...")
        try:
            scheduler.shutdown()
            logger.info("Scheduler shut down.")
        except Exception as e:
            logger.error(f"Error shutting down scheduler: {e}", exc_info=True)
        scheduler = None


# --- Call Initialization (e.g., in main app startup or separate process) ---
# initialize_scheduler()
# Ensure shutdown_scheduler() is called on app exit (e.g., using atexit)
# import atexit
# atexit.register(shutdown_scheduler)