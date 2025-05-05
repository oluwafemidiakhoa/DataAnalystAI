# backend/auth/service.py
# User authentication and registration services

import logging
from sqlalchemy.orm import Session # Import Session for type hinting
from typing import Optional
from datetime import timedelta

# --- Pydantic Models Import ---
# ** FIX: Import UserCreate from the correct auth models file **
try:
    from .models import UserCreate
except ImportError:
    logger = logging.getLogger(__name__) # Define logger if import fails early
    logger.error("Failed to import UserCreate from backend.auth.models.", exc_info=True)
    # Define a dummy class to prevent NameErrors downstream, but log critical error
    class UserCreate: pass
    st.error("Critical Error: Auth models not found.") # Display error if Streamlit context available

# --- Database Components Import ---
DB_AVAILABLE = False
try:
    from backend.database import crud, models as db_models # db_models alias avoids name clash
    from backend.database.session import get_db_session # Example session management
    DB_AVAILABLE = True
except ImportError:
    logging.warning("Database components unavailable for auth service.")
    # Define dummy crud/models if needed for structure testing
    class crud: get_user_by_email = lambda *args, **kwargs: None; create_user = lambda *args, **kwargs: type('DummyUser',(),{'email':'dummy'})()
    class db_models: User = type('DummyUser',(),{})
    def get_db_session(): logger.error("Dummy DB session context used!"); yield None


# --- Security Utilities Import ---
AUTH_UTILS_AVAILABLE = False
try:
    from .security import get_password_hash, verify_password, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
    AUTH_UTILS_AVAILABLE = True
except ImportError:
    logging.warning("Auth security utilities unavailable (passlib or python-jose likely missing).")
    # Define dummy security functions if needed
    def get_password_hash(p): logger.error("Passlib unavailable."); return f"unhashed_{p}"
    def verify_password(p1,p2): logger.error("Passlib unavailable."); return False
    def create_access_token(d,e=None): logger.error("python-jose unavailable."); return None
    ACCESS_TOKEN_EXPIRE_MINUTES = 0


logger = logging.getLogger(__name__)

class AuthServiceError(Exception):
    """Custom exception for auth service errors."""
    pass

# --- Service Functions ---

def register_new_user(user_in: UserCreate, db: Session) -> db_models.User:
    """
    Registers a new user in the database after hashing the password.

    Args:
        user_in: Pydantic model containing user details (email, password, etc.).
        db: SQLAlchemy database session.

    Returns:
        The created database User object (SQLAlchemy model).

    Raises:
        AuthServiceError: If dependencies are missing, email exists, hashing fails, or DB error occurs.
    """
    if not DB_AVAILABLE or not AUTH_UTILS_AVAILABLE:
        raise AuthServiceError("Auth service dependencies (DB or Security Utils) are not available.")
    # Validate input type (redundant if using FastAPI/type hints, but good practice)
    if not isinstance(user_in, UserCreate):
         # This case shouldn't happen if type hints are enforced upstream
         logger.error(f"Invalid input type for register_new_user: {type(user_in)}")
         raise AuthServiceError("Invalid user data provided.")

    logger.info(f"Attempting registration for email: {user_in.email}")

    # 1. Check if user already exists
    try:
        existing_user = crud.get_user_by_email(db, email=user_in.email)
        if existing_user:
            logger.warning(f"Registration failed: Email '{user_in.email}' already registered.")
            raise AuthServiceError(f"Email '{user_in.email}' is already registered.")
    except Exception as e:
        logger.error(f"Database error checking for existing user '{user_in.email}': {e}", exc_info=True)
        raise AuthServiceError("Could not verify user existence due to a database error.") from e

    # 2. Hash the password
    try:
        hashed_password = get_password_hash(user_in.password)
    except ValueError as hash_err: # Catch specific error from get_password_hash
         logger.error(f"Password hashing failed during registration: {hash_err}", exc_info=True)
         raise AuthServiceError("Could not process password.") from hash_err
    except Exception as e: # Catch any other unexpected error during hashing
        logger.error(f"Unexpected error during password hashing: {e}", exc_info=True)
        raise AuthServiceError("An unexpected error occurred during password processing.") from e

    # 3. Create user via CRUD
    try:
        created_user = crud.create_user(
            db=db,
            email=user_in.email,
            password=hashed_password, # Pass the HASHED password
            full_name=user_in.full_name,
            username=user_in.username # Include username if added to models
        )
        logger.info(f"User '{user_in.email}' registered successfully with ID {created_user.id}.")
        return created_user
    except Exception as e:
         logger.error(f"Database error during user creation for '{user_in.email}': {e}", exc_info=True)
         # Don't expose internal DB errors directly
         raise AuthServiceError("Registration failed due to a database error.") from e


def authenticate_user(username: str, password: str, db: Session) -> Optional[db_models.User]:
    """
    Authenticates a user by username/email and password.

    Args:
        username: The user's email or username provided during login.
        password: The plain text password provided during login.
        db: SQLAlchemy database session.

    Returns:
        The authenticated database User object (SQLAlchemy model) or None if authentication fails.
    """
    if not DB_AVAILABLE or not AUTH_UTILS_AVAILABLE:
        logger.error("Auth service dependencies are not available for authentication.")
        return None

    logger.info(f"Attempting authentication for user: {username}")
    user = None
    try:
        # Allow login with email or username
        # Assumes crud.get_user_by_email handles the lookup correctly
        user = crud.get_user_by_email(db, email=username)
        # Optional: If username login is supported and email lookup failed:
        # if not user and '@' not in username:
        #    user = crud.get_user_by_username(db, username=username) # Requires this crud function

    except Exception as e:
         logger.error(f"Database error retrieving user '{username}': {e}", exc_info=True)
         # Return None on DB error during lookup? Or raise? Returning None is safer for auth flow.
         return None

    if not user:
        logger.warning(f"Authentication failed: User '{username}' not found.")
        return None
    if not user.is_active:
         logger.warning(f"Authentication failed: User '{username}' is inactive.")
         return None

    # Verify password using the security utility
    if not verify_password(password, user.hashed_password):
        logger.warning(f"Authentication failed: Invalid password for user '{username}'.")
        return None

    logger.info(f"User '{username}' (ID: {user.id}) authenticated successfully.")
    return user


def create_login_token(user: db_models.User) -> Optional[str]:
    """
    Generates a JWT access token for a given authenticated user.

    Args:
        user: The authenticated database User object.

    Returns:
        The encoded JWT string, or None if token creation fails.
    """
    if not AUTH_UTILS_AVAILABLE:
         logger.error("Cannot create token: Auth security utils unavailable.")
         return None

    # Use expire minutes from settings if available, otherwise default
    expire_minutes = ACCESS_TOKEN_EXPIRE_MINUTES
    access_token_expires = timedelta(minutes=expire_minutes)

    # Data to encode in the token (keep minimal for security)
    # Using user ID as the 'subject' is common and secure
    token_data = {"sub": str(user.id)}
    # Add other non-sensitive claims if needed (e.g., roles)
    # token_data["roles"] = ["admin"] if user.is_superuser else ["user"]

    try:
        access_token = create_access_token(
            data=token_data, expires_delta=access_token_expires
        )
        if access_token:
            logger.info(f"Access token created successfully for user ID {user.id}")
        else:
             logger.error(f"Token creation returned None for user ID {user.id}")
        return access_token
    except Exception as e:
         logger.error(f"Unexpected error during token creation for user ID {user.id}: {e}", exc_info=True)
         return None