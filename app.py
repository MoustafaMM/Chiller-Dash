# app.py
from python_hvac_iot_dashboard import app as application

# Expose the Dash server instance to Gunicorn
server = application.server
