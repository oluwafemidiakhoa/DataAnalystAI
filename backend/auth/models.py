# backend/auth/models.py
# Pydantic models (schemas) for API request/response validation related to Authentication.
# These define the expected data structure for inputs and outputs.
# Database ORM models (User table) should remain in backend/database/models.py

from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import Optional

# --- User Schemas ---

class UserBase(BaseModel):
    """Base schema for user properties shared across create/read."""
    email: EmailStr = Field(..., example="user@example.com")
    full_name: Optional[str] = Field(None, max_length=200, example="John Doe")
    username: Optional[str] = Field(None, max_length=100, example="johndoe")
    is_active: bool = Field(default=True, description="User account is active")
    is_superuser: bool = Field(default=False, description="User has admin privileges")

class UserCreate(UserBase):
    """Schema for creating a new user via API request."""
    # Inherits email, full_name, username from UserBase
    # is_active and is_superuser default to False/True in UserBase unless overridden here
    password: str = Field(..., min_length=8, max_length=100, example="SecureP@ssw0rd1", description="User's chosen password")

    # Explicitly set defaults for creation if different from Base
    is_active: bool = True
    is_superuser: bool = False

class UserUpdate(BaseModel):
    """Schema for updating user information (all fields optional)."""
    # Use BaseModel directly as all fields are optional
    email: Optional[EmailStr] = Field(None, example="new.user@example.com")
    full_name: Optional[str] = Field(None, max_length=200, example="Johnathan Doe")
    username: Optional[str] = Field(None, max_length=100, example="john.doe")
    password: Optional[str] = Field(None, min_length=8, max_length=100, example="NewSecureP@ssw0rd2", description="New password (if changing)")
    is_active: Optional[bool] = None
    is_superuser: Optional[bool] = None

class UserRead(UserBase):
    """Schema for returning user data via API (excluding password)."""
    # Inherits fields from UserBase
    id: int = Field(..., description="Unique user identifier")

    # Pydantic V2 uses model_config instead of class Config
    model_config = ConfigDict(
        from_attributes = True # Enable creating this model from ORM objects (like SQLAlchemy models)
    )

# --- Token Schemas ---

class Token(BaseModel):
    """Schema for the JWT access token response."""
    access_token: str = Field(..., description="The JWT access token string")
    token_type: str = Field(default="bearer", description="The type of token (typically 'bearer')")

class TokenPayload(BaseModel):
    """Schema for data encoded within the JWT payload (standard claims)."""
    # Standard JWT claims: https://tools.ietf.org/html/rfc7519#section-4.1
    sub: str = Field(..., description="Subject of the token (usually user ID or email)")
    exp: Optional[int] = Field(None, description="Expiration time (Unix timestamp)")
    iat: Optional[int] = Field(None, description="Issued at time (Unix timestamp)")
    # Add custom claims if needed
    # roles: Optional[List[str]] = None
    # username: Optional[str] = None # Can be included, but sub is standard for identity

# --- Login Schema ---

class LoginRequest(BaseModel):
    """Schema for login request data."""
    # Often uses 'username' field which can accept either email or actual username
    username: str = Field(..., description="User's email or username for login", example="user@example.com")
    password: str = Field(..., example="SecureP@ssw0rd1")