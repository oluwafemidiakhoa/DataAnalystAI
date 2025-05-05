# pages/auth_signup.py (Placeholder)
# NOTE: Requires a proper backend API or library like streamlit-authenticator for real security.

import streamlit as st
import logging

# Assume auth service and Pydantic model available
try:
    from backend.auth.service import register_new_user
    from backend.auth.models import UserCreate
    from backend.database.session import get_db_session
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Sign Up", layout="centered")
st.header("🚀 Sign Up for DataVisionAI")

if not AUTH_AVAILABLE:
    st.error("Authentication service is not available. Cannot sign up.", icon="🚨")
    st.stop()

# Check if user is already logged in
if st.session_state.get("authenticated_user"):
    st.success(f"You are already logged in as {st.session_state.authenticated_user.get('email', 'user')}.")
    if st.button("Go to App"):
        st.switch_page("app.py")
    st.stop()


with st.form("signup_form"):
    full_name = st.text_input("Full Name", key="signup_name")
    email = st.text_input("Email*", key="signup_email")
    password = st.text_input("Password*", type="password", key="signup_pass", help="Minimum 8 characters")
    confirm_password = st.text_input("Confirm Password*", type="password", key="signup_confirm")
    submitted = st.form_submit_button("Sign Up", type="primary", use_container_width=True)

    if submitted:
        if not email or not password or not confirm_password:
            st.warning("Please fill in all required fields (*).")
        elif password != confirm_password:
            st.error("Passwords do not match.")
        elif len(password) < 8:
             st.error("Password must be at least 8 characters long.")
        else:
            try:
                 user_data = UserCreate(email=email, password=password, full_name=full_name) # Validate with Pydantic
                 with get_db_session() as db: # Replace with your actual session logic
                      new_user = register_new_user(user_in=user_data, db=db)
                 st.success(f"Account created successfully for {new_user.email}! Please log in.")
                 logger.info(f"User {new_user.email} registered.")
                 time.sleep(2)
                 st.switch_page("pages/auth_login.py") # Redirect to login
            except Exception as e: # Catch errors from validation or registration service
                 st.error(f"Registration failed: {e}")
                 logger.error(f"Signup failed for {email}: {e}", exc_info=True)


st.divider()
st.caption("Already have an account?")
if st.button("Login Instead"):
    st.switch_page("pages/auth_login.py")