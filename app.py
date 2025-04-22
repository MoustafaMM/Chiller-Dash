import subprocess
import sys
# app.py
from python_hvac_iot_dashboard import app as application

# This exposes the Dash server object to gunicorn
server = application.server


# Paths to your scripts (use forward slashes or raw strings)
SENDER_SCRIPT = "d:/Master/GUI/Sensor.py"
DASHBOARD_SCRIPT = "d:/Master/GUI/python hvac_iot_dashboard.py"
PYTHON_EXEC = "C:/Users/moust/AppData/Local/Programs/Python/Python310/python.exe"

def run_scripts():
    # Verify files exist before running
    for script in [SENDER_SCRIPT, DASHBOARD_SCRIPT]:
        if not subprocess.os.path.exists(script):
            print(f"Error: {script} not found!")
            sys.exit(1)

    # Start sender in a new process
    sender_process = subprocess.Popen([PYTHON_EXEC, SENDER_SCRIPT], shell=True)
    print("Started MQTT Sender...")

    # Start dashboard in a new process
    dashboard_process = subprocess.Popen([PYTHON_EXEC, DASHBOARD_SCRIPT], shell=True)
    print("Started Dash Dashboard...")

    # Wait for both to finish (Ctrl+C to stop)
    try:
        sender_process.wait()
        dashboard_process.wait()
    except KeyboardInterrupt:
        print("Stopping both scripts...")
        sender_process.terminate()
        dashboard_process.terminate()

if __name__ == "__main__":
    run_scripts()