# pages/auth_login.py (Placeholder)
# NOTE: Requires a proper backend API or library like streamlit-authenticator for real security.

import streamlit as st
import logging

# Assume auth service is available
try:
    from backend.auth.service import authenticate_user, create_login_token
    from backend.database.session import get_db_session # How to get session? Depends on setup.
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Login", layout="centered") # Use centered layout for login

st.header("🔒 Login to DataVisionAI")

if not AUTH_AVAILABLE:
    st.error("Authentication service is not available. Cannot log in.", icon="🚨")
    st.stop()

# Check if user is already logged in (using session state)
if st.session_state.get("authenticated_user"):
    st.success(f"You are already logged in as {st.session_state.authenticated_user.get('email', 'user')}.")
    if st.button("Go to App"):
        st.switch_page("app.py") # Or specific page
    st.stop()


with st.form("login_form"):
    email_or_username = st.text_input("Email or Username", key="login_user")
    password = st.text_input("Password", type="password", key="login_pass")
    submitted = st.form_submit_button("Login", type="primary", use_container_width=True)

    if submitted:
        if not email_or_username or not password:
            st.warning("Please enter both email/username and password.")
        else:
            try:
                # --- WARNING: Getting DB session here is non-trivial in pure Streamlit ---
                # This typically requires context management or a dedicated session setup.
                # Using a placeholder context manager for demonstration.
                with get_db_session() as db: # Replace with your actual session logic
                     user = authenticate_user(username=email_or_username, password=password, db=db)

                if user:
                    logger.info(f"User {user.email} successfully logged in.")
                    st.toast("Login successful!", icon="✅")
                    # Generate token (optional if using session state directly)
                    # access_token = create_login_token(user)
                    # Store login state securely (e.g., token in cookie/local storage via components)
                    # For simplicity, store user info in session state (less secure for tokens)
                    st.session_state.authenticated_user = {"id": user.id, "email": user.email, "name": user.full_name}
                    st.session_state.logged_in = True # Simple flag
                    time.sleep(1) # Allow toast to show
                    st.switch_page("app.py") # Redirect to main app page
                else:
                    st.error("Incorrect email/username or password.")
            except Exception as e:
                st.error(f"An error occurred during login: {e}")
                logger.error(f"Login failed for {email_or_username}: {e}", exc_info=True)

st.divider()
st.caption("Don't have an account?")
if st.button("Sign Up Instead"):
    st.switch_page("pages/auth_signup.py")