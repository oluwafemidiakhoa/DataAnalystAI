# frontend/components/notifications.py
# Component to display alerts or proactive insights

import streamlit as st
from typing import List, Dict, Any

def display_notifications(notifications: List[Dict[str, Any]], title: str = "🔔 Notifications & Alerts"):
    """
    Displays a list of notifications using Streamlit elements.

    Args:
        notifications: A list of notification dictionaries. Expected keys:
                       'message' (str), 'type' (str: 'info', 'success', 'warning', 'error'),
                       'timestamp' (optional datetime/str), 'details' (optional str).
        title: The title for the notification area.
    """
    if not notifications:
        st.caption("_No new notifications._")
        return

    with st.container(border=True):
        st.subheader(title)
        # Display latest first?
        for i, notification in enumerate(reversed(notifications)):
            msg = notification.get('message', 'Notification')
            ntype = notification.get('type', 'info')
            details = notification.get('details')
            timestamp = notification.get('timestamp', '') # Format timestamp nicely if available

            icon = "ℹ️"
            display_func = st.info
            if ntype == 'success': icon = "✅"; display_func = st.success
            elif ntype == 'warning': icon = "⚠️"; display_func = st.warning
            elif ntype == 'error': icon = "🚨"; display_func = st.error

            # Use expander for details? Or just caption?
            with st.expander(f"{icon} {msg} ({timestamp})", expanded=i==0): # Expand newest
                 if details:
                      st.markdown(details) # Display details if provided
                 # Add actions like "Dismiss" here later
                 # st.button("Dismiss", key=f"dismiss_noti_{i}")

# Example Usage:
# if __name__ == '__main__':
#     st.header("Notification Component Example")
#     mock_notifications = [
#         {'message': 'Data quality check failed for Sales Data', 'type': 'error', 'details': 'Rule "Revenue > 0" failed on 15 rows.', 'timestamp': '10:30 AM'},
#         {'message': 'Proactive Insight: User engagement spiked yesterday.', 'type': 'success', 'timestamp': '09:15 AM'},
#         {'message': 'Scheduled report "Weekly Summary" is ready.', 'type': 'info', 'timestamp': '08:00 AM'},
#         {'message': 'Low disk space warning on server X.', 'type': 'warning', 'timestamp': '07:55 AM'},
#     ]
#     display_notifications(mock_notifications)
#     st.subheader("Empty State")
#     display_notifications([])