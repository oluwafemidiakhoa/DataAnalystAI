# backend/core/logging_config.py
import logging
import sys
from backend.core.config import settings # Import settings to use log level

def setup_logging():
    """Configures the root logger for the application."""

    log_level_str = settings.log_level.upper() if settings else "INFO"
    log_level = getattr(logging, log_level_str, logging.INFO) # Default to INFO if invalid

    # Create formatter
    log_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create handler (console)
    console_handler = logging.StreamHandler(sys.stdout) # Use stdout for compatibility with container logs
    console_handler.setFormatter(log_formatter)

    # Get the root logger and configure it
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers (if any) to avoid duplicate logs
    # Important if this function might be called multiple times, though usually called once
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Add the console handler
    root_logger.addHandler(console_handler)

    # Set levels for noisy libraries if needed
    # logging.getLogger("urllib3").setLevel(logging.WARNING)
    # logging.getLogger("matplotlib").setLevel(logging.INFO)

    logger = logging.getLogger(__name__) # Get a logger for this module itself
    logger.info(f"Root logger configured with level: {log_level_str}")
    # Test message levels (only visible if level is set appropriately)
    # logger.debug("This is a debug message.")
    # logger.info("This is an info message.")
    # logger.warning("This is a warning message.")
    # logger.error("This is an error message.")
    # logger.critical("This is a critical message.")


# Call setup_logging() early in your application's lifecycle.
# You might call it once in your main backend entry point or relevant module
# that gets loaded early. For Streamlit, importing it might suffice if called globally.
# However, placing the call explicitly might be safer.
# E.g., in backend/__init__.py or a main backend orchestrator file if you create one.
# setup_logging() # Consider where best to call this once.