import paho.mqtt.client as mqtt
import time
import random
import json

# ✅ MQTT Broker Settings
MQTT_BROKER = "mqtt.eclipseprojects.io"
MQTT_PORT = 1883
MQTT_TOPIC = "hvac/sensor"
SEND_INTERVAL = 2  # Time in seconds between messages

# ✅ Create MQTT Client
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

# ✅ Ensure robust connection to the broker
def connect_mqtt():
    while True:
        try:
            client.connect(MQTT_BROKER, MQTT_PORT, 60)
            print(f"✅ Connected to MQTT Broker: {MQTT_BROKER}")
            return
        except Exception as e:
            print(f"❌ Connection failed: {e}. Retrying in 5 seconds...")
            time.sleep(5)

connect_mqtt()

# ✅ Simulated Sensor Data Categories
data_categories = {
    "Voltage (V)": lambda: round(random.uniform(200, 240), 2),
    "Current (I)": lambda: round(random.uniform(5, 15), 2),
    "Power (P)": lambda: round(random.uniform(500, 2000), 2),
    "Frequency (F)": lambda: round(random.uniform(49, 51), 2),
    "Vibration": lambda: round(random.uniform(0, 5), 2),
    "Temp (T)": lambda: round(random.uniform(20, 35), 2),
    "Flow Rate": lambda: round(random.uniform(10, 100), 2),
}

# ✅ MQTT Disconnection Handling
def on_disconnect(client, userdata, rc):
    print(f"⚠️ Disconnected from MQTT Broker! Reconnecting...")
    connect_mqtt()

client.on_disconnect = on_disconnect

try:
    while True:
        # ✅ Collect sensor readings
        sensor_values = {category: generator() for category, generator in data_categories.items()}
        payload = json.dumps(sensor_values)  # Convert to JSON
        
        # ✅ Publish sensor data
        try:
            client.publish(MQTT_TOPIC, payload)
            print(f"📤 Sent: {payload}")
        except Exception as e:
            print(f"❌ Publish failed: {e}")
        
        time.sleep(SEND_INTERVAL)  # Wait before sending next batch

except KeyboardInterrupt:
    print("\n🚦 Sensor Simulation Stopped")
    client.disconnect()
