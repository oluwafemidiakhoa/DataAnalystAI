# backend/notifications/alerter.py
# Logic for sending alerts (e.g., email) based on events.

import logging
import smtplib
from email.message import EmailMessage
from typing import List, Dict, Any, Optional

# Assuming access to central settings for SMTP config
try:
    from backend.core.config import settings
    SETTINGS_AVAILABLE = True
except (ImportError, RuntimeError):
    settings = None
    SETTINGS_AVAILABLE = False
    logging.warning("Could not import settings for alerter. Email notifications disabled.")

logger = logging.getLogger(__name__)

def _send_email_alert(subject: str, content_html: str, recipients: List[str]):
    """Internal helper to send email using configured SMTP settings."""
    if not SETTINGS_AVAILABLE or not all([settings.smtp_host, settings.smtp_port, settings.emails_from_email]):
        logger.error("SMTP settings not configured. Cannot send email alert.")
        return False
    if not recipients:
        logger.warning("No recipients provided for email alert.")
        return False

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = f"{settings.emails_from_name or settings.app_name} <{settings.emails_from_email}>"
    msg['To'] = ", ".join(recipients)
    msg.set_content("Please enable HTML emails to view this alert.") # Fallback text
    msg.add_alternative(content_html, subtype='html') # HTML content

    try:
        logger.info(f"Connecting to SMTP server: {settings.smtp_host}:{settings.smtp_port}")
        server: Optional[smtplib.SMTP] = None # Initialize with proper type hint
        if settings.smtp_tls:
             # Use SMTP_SSL for implicit SSL (usually port 465) or starttls for explicit TLS (usually port 587)
             if settings.smtp_port == 465: # Common SSL port
                 server = smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, timeout=10)
             else: # Assume STARTTLS for other ports like 587
                 server = smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10)
                 server.starttls()
        else:
            server = smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10)

        # Login if credentials provided
        if settings.smtp_user and settings.smtp_password:
            logger.info(f"Logging in as SMTP user: {settings.smtp_user}")
            server.login(settings.smtp_user, settings.smtp_password)

        logger.info(f"Sending email alert '{subject}' to: {recipients}")
        server.send_message(msg)
        logger.info("Email alert sent successfully.")
        server.quit()
        return True
    except smtplib.SMTPAuthenticationError as e:
        logger.error(f"SMTP Authentication failed: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Failed to send email alert: {e}", exc_info=True)
        if server:
            try: server.quit() # Ensure server is closed
            except Exception: pass
    return False


def send_quality_alert(datasource_name: str, results: List[Dict[str, Any]], recipients: Optional[List[str]] = None):
    """Formats and sends a data quality violation alert."""
    if recipients is None: recipients = [] # Get default recipients from config/DB later?
    if not recipients: logger.warning("No recipients configured for quality alerts."); return

    failed_rules = [r for r in results if r.get('status') == 'failed']
    error_rules = [r for r in results if r.get('status') == 'error']
    if not failed_rules and not error_rules: return # Don't alert if nothing failed/errored

    subject = f"🚨 Data Quality Alert: Issues detected in '{datasource_name}'"
    # --- HTML Content (Example) ---
    content = f"""
    <html><body>
    <h2>Data Quality Alert</h2>
    <p>Issues detected during the recent quality check for data source: <strong>{datasource_name}</strong></p>
    """
    if failed_rules:
        content += "<h3>Failed Rules:</h3><ul>"
        for rule in failed_rules:
            content += f"<li><strong>{rule.get('rule_name', 'Unnamed Rule')}</strong> ({rule.get('rule_type','N/A')} on '{rule.get('column_name','N/A')}') - {rule.get('violation_count','N/A')} violations. Details: {rule.get('details','N/A')}</li>"
        content += "</ul>"
    if error_rules:
         content += "<h3 style='color:red;'>Errored Rules:</h3><ul>"
         for rule in error_rules:
              content += f"<li><strong>{rule.get('rule_name', 'Unnamed Rule')}</strong> - Error executing check. Details: {rule.get('details','N/A')}</li>"
         content += "</ul>"
    content += "<p>Please review the data source and quality rules in the application.</p>"
    content += f"<p><small>Alert generated at {datetime.datetime.now(datetime.timezone.utc)} UTC</small></p>"
    content += "</body></html>"
    # --- End HTML ---

    _send_email_alert(subject, content, recipients)


def send_proactive_insight_alert(datasource_name: str, anomalies: List[Dict[str, Any]], recipients: Optional[List[str]] = None):
    """Formats and sends a proactive insight/anomaly alert."""
    if recipients is None: recipients = []
    if not recipients: logger.warning("No recipients configured for proactive alerts."); return
    if not anomalies: return # Don't send if no anomalies found

    subject = f"💡 Proactive Insight: Notable patterns detected in '{datasource_name}'"
    # --- HTML Content (Example) ---
    content = f"""
    <html><body>
    <h2>Proactive Insight Alert</h2>
    <p>The system detected notable patterns or potential anomalies during recent analysis of data source: <strong>{datasource_name}</strong></p>
    <h3>Detected Items:</h3>
    <ul>
    """
    for item in anomalies:
         severity_color = "orange" if item.get('severity') == 'Medium' else "red" if item.get('severity') == 'High' else "gray"
         content += f"<li><strong>{item.get('type', 'Insight')}</strong> (<span style='color:{severity_color};'>{item.get('severity','Unknown')}</span>): {item.get('details','N/A')}</li>"
    content += "</ul>"
    content += "<p>Consider investigating these findings further in the application.</p>"
    content += f"<p><small>Alert generated at {datetime.datetime.now(datetime.timezone.utc)} UTC</small></p>"
    content += "</body></html>"
    # --- End HTML ---

    _send_email_alert(subject, content, recipients)