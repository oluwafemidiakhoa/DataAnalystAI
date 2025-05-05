# backend/core/config.py
import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
# ** FIX: Import Path from pathlib, not directly from pydantic **
from pydantic import Field, EmailStr, RedisDsn, AmqpDsn, FilePath, AnyHttpUrl, SecretStr
from typing import Optional, Literal
import logging
from pathlib import Path # <--- Import Python's built-in Path

# --- Get Logger Instance ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Determine Paths ---
PROJECT_ROOT_ENV = os.environ.get("PROJECT_ROOT")
if PROJECT_ROOT_ENV:
    PROJECT_ROOT = Path(PROJECT_ROOT_ENV).resolve()
    logger.info(f"Project root set from environment variable: {PROJECT_ROOT}")
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    logger.warning(f"PROJECT_ROOT env var not set. Assuming project root: {PROJECT_ROOT}")

ENV_FILE_PATH = PROJECT_ROOT / '.env'
DEFAULT_DATA_PATH = PROJECT_ROOT / 'data'
logger.debug(f"Calculated PROJECT_ROOT: {PROJECT_ROOT}")
logger.debug(f"Calculated ENV_FILE_PATH: {ENV_FILE_PATH}")

# --- Load .env File ---
logger.info(f"Checking for .env file at: {ENV_FILE_PATH}")
if ENV_FILE_PATH.is_file():
    logger.info(".env file FOUND.")
    dotenv_loaded = load_dotenv(dotenv_path=ENV_FILE_PATH, verbose=True, override=False)
    logger.info(f".env file load attempt complete. Dotenv loaded: {dotenv_loaded}")
else:
    logger.info(f".env file NOT FOUND at {ENV_FILE_PATH}. Relying on environment variables.")
# Debug prints removed for clarity now


# --- Application Settings Model ---
class AppSettings(BaseSettings):
    """
    Centralized Application Settings using Pydantic BaseSettings.
    Reads variables from environment variables or a .env file (case-insensitive).
    """
    # --- Core Settings ---
    app_name: str = Field(default="DataVisionAI Workspace")
    environment: Literal['development', 'staging', 'production'] = Field(default='development')
    log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = Field(default="INFO")
    debug_mode: bool = Field(default=False)

    # --- Gemini API Settings ---
    gemini_api_key: SecretStr = Field(..., description="REQUIRED")
    gemini_default_model: str = Field(default="gemini-1.5-flash")
    gemini_advanced_model: str = Field(default="gemini-1.5-pro")

    # --- Auth Settings ---
    jwt_secret_key: SecretStr = Field(default="!!!SET_A_STRONG_SECRET_KEY_IN_CONFIG!!!")
    jwt_algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=30)

    # --- Database Settings (App Persistence) ---
    database_url: Optional[str] = Field(default=f"sqlite:///{PROJECT_ROOT / 'app_persistence.db'}")

    # --- Email Settings ---
    smtp_host: Optional[str] = Field(default=None)
    smtp_port: Optional[int] = Field(default=587)
    smtp_user: Optional[str] = Field(default=None)
    smtp_password: Optional[SecretStr] = Field(default=None)
    smtp_tls: bool = Field(default=True)
    emails_from_email: Optional[EmailStr] = Field(default=None)
    emails_from_name: Optional[str] = Field(default="DataVisionAI Workspace")

    # --- Task Queue Settings ---
    redis_url: Optional[RedisDsn] = Field(default=None)
    task_queue_enabled: bool = Field(default=False)

    # --- File/Path Settings ---
    # ** FIX: Use pathlib.Path for type hint, not pydantic.Path or DirectoryPath **
    # This allows the directory creation code below to handle existence check.
    default_upload_dir: Optional[Path] = Field(default=PROJECT_ROOT / "uploads")
    default_report_dir: Optional[Path] = Field(default=PROJECT_ROOT / "reports_output")

    # --- Feature Flags ---
    enable_visual_etl: bool = Field(default=False)
    enable_forecasting: bool = Field(default=True)
    enable_segmentation: bool = Field(default=True)
    enable_quality_monitoring: bool = Field(default=True)


    # --- Pydantic Settings Configuration ---
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE_PATH) if ENV_FILE_PATH.is_file() else None,
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )

# --- Instantiate Settings ---
settings: Optional[AppSettings] = None
try:
    logger.info("Attempting to instantiate AppSettings...")
    settings = AppSettings()
    logger.info("AppSettings instantiation complete.")

    # --- Post-Initialization Validation/Setup ---
    # Check key existence
    if not settings.gemini_api_key or not settings.gemini_api_key.get_secret_value():
        logger.critical("CRITICAL FAILURE: GEMINI_API_KEY missing/empty.")
    else: logger.info("Gemini API Key loaded (existence verified).")
    # Check JWT placeholder
    if settings.jwt_secret_key.get_secret_value() == "!!!SET_A_STRONG_SECRET_KEY_IN_CONFIG!!!":
        logger.critical("SECURITY WARNING: Default JWT_SECRET_KEY in use!")

    # --- Create default directories AFTER settings load ---
    logger.info("Ensuring required directories exist...")
    dirs_to_create = [settings.default_upload_dir, settings.default_report_dir]
    for dir_path_obj in dirs_to_create:
        if dir_path_obj: # Check if path is set in settings
             try:
                 # dir_path_obj is now a pathlib.Path object from Pydantic
                 dir_path_obj.mkdir(parents=True, exist_ok=True)
                 logger.info(f"Ensured directory exists: {dir_path_obj}")
             except Exception as e: logger.error(f"Failed to create directory {dir_path_obj}: {e}")
        else: logger.warning("Directory path setting not configured, skipping creation.")

    # Log loaded settings (mask secrets)
    settings_dict = settings.model_dump(); masked_settings = {}
    for k, v in settings_dict.items():
        if isinstance(v, SecretStr): masked_settings[k] = v.get_secret_value()[:1]+'********' if v.get_secret_value() else None
        elif isinstance(v, str) and ('key' in k.lower() or 'password' in k.lower() or 'secret' in k.lower()): masked_settings[k] = v[:1]+'********' if v else None
        else: masked_settings[k] = v
    logger.info(f"Application settings loaded: {masked_settings}")

except Exception as e:
    logger.critical(f"CRITICAL: Failed to initialize application settings: {e}", exc_info=True)
    settings = None
    raise RuntimeError(f"Failed to initialize application settings: {e}") from e

# --- How to Use ---
# from backend.core.config import settings
# if settings: api_key = settings.gemini_api_key.get_secret_value() ...