# backend/auth/security.py
# Password hashing, JWT token creation/verification

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Any, Dict # Added Dict for type hinting
from pydantic import SecretStr # Use SecretStr for keys

# --- Dependency Imports with Error Handling ---
PASSLIB_AVAILABLE = False
JOSE_AVAILABLE = False
SETTINGS_AVAILABLE_FOR_AUTH = False
# Secure Defaults (used ONLY if settings/libs fail)
# IMPORTANT: Generate a strong, unique secret key for production using: openssl rand -hex 32
DEFAULT_JWT_SECRET_KEY = "insecure_default_key_MUST_BE_REPLACED_123!"
SECRET_KEY = DEFAULT_JWT_SECRET_KEY # Initialize with default
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Define logger early
logger = logging.getLogger(__name__)

# --- Passlib Import and Context Initialization ---
pwd_context = None
try:
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    PASSLIB_AVAILABLE = True
    logger.info("Passlib (bcrypt) context initialized for password hashing.")
except ImportError:
    logging.warning("passlib not installed (`pip install passlib[bcrypt]`). Password hashing/verification disabled.")
except Exception as e:
    logging.error(f"Failed to initialize passlib CryptContext: {e}", exc_info=True)
    # Ensure pwd_context is None if init fails
    pwd_context = None

# --- Jose Import ---
try:
    from jose import JWTError, jwt
    JOSE_AVAILABLE = True
    logger.info("python-jose library loaded for JWT handling.")
except ImportError:
    logging.warning("python-jose not installed (`pip install python-jose[cryptography]`). JWT handling disabled.")

# --- Load Settings Safely ---
try:
    from backend.core.config import settings # Assume config.py exists and defines settings
    if settings: # Check if settings object was successfully created
        # Use SecretStr for sensitive values from config
        # Use .get() for attributes that might not be present if config loading partially failed
        jwt_secret_value = getattr(settings, "jwt_secret_key", None)
        SECRET_KEY = jwt_secret_value.get_secret_value() if isinstance(jwt_secret_value, SecretStr) else str(jwt_secret_value or DEFAULT_JWT_SECRET_KEY)

        ALGORITHM = getattr(settings, "jwt_algorithm", ALGORITHM)
        ACCESS_TOKEN_EXPIRE_MINUTES = getattr(settings, "access_token_expire_minutes", ACCESS_TOKEN_EXPIRE_MINUTES)
        SETTINGS_AVAILABLE_FOR_AUTH = True

        # Critical security warning if using default secret key
        if SECRET_KEY == DEFAULT_JWT_SECRET_KEY or SECRET_KEY == "!!!SET_A_STRONG_SECRET_KEY_IN_CONFIG!!!": # Check against old placeholder too
            logger.critical(
                "SECURITY WARNING: JWT_SECRET_KEY is not set in config or is using an "
                "insecure default placeholder! Generate a strong secret key using 'openssl rand -hex 32' "
                "and set it in your .env file or environment variables."
                )
            # Optionally: Raise an error in production environments
            # if settings.environment == 'production': raise ValueError("Insecure JWT_SECRET_KEY in production!")
        else:
             logger.info(f"JWT settings loaded from config: Algorithm={ALGORITHM}, ExpiresIn={ACCESS_TOKEN_EXPIRE_MINUTES}min")
    else:
         logger.error("Settings object is None after import. Using insecure auth defaults.")
except (ImportError, AttributeError, RuntimeError) as e:
     logger.error(f"Could not load/access settings for auth security: {e}. Using insecure hardcoded defaults.", exc_info=True)
     # Defaults are already set above


# --- Password Hashing Functions ---
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against a hashed password using passlib."""
    if not PASSLIB_AVAILABLE or not pwd_context:
        logger.error("Passlib unavailable or context failed initialization. Cannot verify password.")
        return False
    if not plain_password or not hashed_password:
         logger.warning("Attempted password verification with empty plain password or hash.")
         return False
    try:
        is_valid = pwd_context.verify(plain_password, hashed_password)
        # ** FIX: Corrected the log message string **
        if not is_valid:
            logger.debug("Password verification failed: Hash mismatch.") # Simple, correct string
        else:
             logger.debug("Password verification successful.")
        # ** END FIX **
        return is_valid
    except Exception as e:
        # Catch potential errors from passlib (e.g., invalid hash format, algorithm issues)
        logger.error(f"Error during password verification process: {e}", exc_info=True)
        return False

def get_password_hash(password: str) -> str:
    """Hashes a plain password using passlib (bcrypt)."""
    if not PASSLIB_AVAILABLE or not pwd_context:
        logger.error("Passlib unavailable or context failed initialization. Cannot hash password.")
        raise RuntimeError("Password hashing library (passlib) is not configured correctly.")
    if not password:
         raise ValueError("Cannot hash an empty password.")
    try:
        hashed = pwd_context.hash(password)
        logger.debug("Password hashed successfully.")
        return hashed
    except Exception as e:
        logger.error(f"Error hashing password: {e}", exc_info=True)
        raise ValueError("Password hashing failed unexpectedly.") from e


# --- JWT Token Handling Functions ---
def create_access_token(data: Dict[str, Any], expires_delta_minutes: Optional[int] = None) -> Optional[str]:
    """
    Creates a JWT access token with an expiry time.

    Args:
        data: Dictionary containing the payload data (claims) for the token.
              Must include identifier like 'user_id' or 'email' for 'sub' claim.
        expires_delta_minutes: Optional token validity duration in minutes.
                               Uses configured default if None.

    Returns:
        The encoded JWT string, or None if creation fails.
    """
    if not JOSE_AVAILABLE:
        logger.error("python-jose unavailable. Cannot create access token.")
        return None
    if not SECRET_KEY or SECRET_KEY == DEFAULT_JWT_SECRET_KEY: # Extra check for safety
         logger.critical("Cannot create token: Insecure or missing JWT_SECRET_KEY.")
         return None

    to_encode = data.copy()
    # Determine expiry
    effective_expire_minutes = expires_delta_minutes if expires_delta_minutes is not None else ACCESS_TOKEN_EXPIRE_MINUTES
    expire = datetime.now(timezone.utc) + timedelta(minutes=effective_expire_minutes)

    # Add standard claims
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})

    # Set subject claim ('sub')
    if "user_id" in data: to_encode.setdefault("sub", str(data["user_id"]))
    elif "email" in data: to_encode.setdefault("sub", data["email"])
    else: logger.warning("No 'user_id' or 'email' found in token data for 'sub' claim.")

    try:
        # Log payload before encoding (excluding sensitive parts if necessary)
        log_payload = {k:v for k,v in to_encode.items() if k not in ['exp','iat'] and 'password' not in k}
        logger.debug(f"Encoding JWT with payload: {log_payload} expiring at {expire}")
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    except JWTError as e:
        logger.error(f"Error encoding JWT token: {e}", exc_info=True)
        return None
    except Exception as e:
         logger.error(f"Unexpected error during JWT encoding: {e}", exc_info=True)
         return None


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verifies a JWT access token and returns the payload if valid and not expired.

    Args:
        token: The JWT string to verify.

    Returns:
        The decoded payload dictionary if the token is valid and not expired,
        otherwise None.
    """
    if not JOSE_AVAILABLE:
        logger.error("python-jose unavailable. Cannot verify token.")
        return None
    if not token:
         logger.warning("Attempted to verify an empty token.")
         return None
    if not SECRET_KEY or SECRET_KEY == DEFAULT_JWT_SECRET_KEY:
        logger.critical("Cannot verify token: Insecure or missing JWT_SECRET_KEY.")
        return None

    try:
        # Decode the token. Verifies signature and expiry based on 'exp' claim.
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
            options={"verify_aud": False} # Adjust audience verification if needed
        )

        # Basic validation of payload content
        subject = payload.get("sub")
        expiry = payload.get("exp") # Already verified by jwt.decode, but good to log
        if not subject:
            logger.warning("Token verification failed: Subject (sub) claim missing.")
            return None

        logger.debug(f"Token verified successfully for subject: {subject}, expires: {datetime.fromtimestamp(expiry, timezone.utc) if expiry else 'N/A'}")
        return payload

    except jwt.ExpiredSignatureError:
        logger.info("Token verification failed: Expired signature.")
        return None # Specific indication of expiry
    except jwt.JWTClaimsError as e: # Handle invalid claims (e.g., bad 'exp' format before decode check)
         logger.warning(f"Token verification failed: Invalid claims - {e}")
         return None
    except JWTError as e: # Catch other JOSE errors (invalid signature, format, etc.)
        logger.warning(f"Token verification failed: JWT Error - {e}")
        return None
    except Exception as e: # Catch unexpected errors
         logger.error(f"Unexpected error during token verification: {e}", exc_info=True)
         return None