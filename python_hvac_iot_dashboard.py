import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import paho.mqtt.client as mqtt
import time
import json
from collections import deque

# MQTT Broker Settings
MQTT_BROKER = "mqtt.eclipseprojects.io"
MQTT_PORT = 1883
MQTT_TOPIC = "hvac/sensor"

# Live Sensor Data Storage
MAX_DATA_POINTS = 50
sensor_data = {
    "time": deque(maxlen=MAX_DATA_POINTS),
    "Voltage (V)": deque(maxlen=MAX_DATA_POINTS),
    "Current (I)": deque(maxlen=MAX_DATA_POINTS),
    "Power (P)": deque(maxlen=MAX_DATA_POINTS),
    "Frequency (F)": deque(maxlen=MAX_DATA_POINTS),
    "Vibration": deque(maxlen=MAX_DATA_POINTS),
    "Temp (T)": deque(maxlen=MAX_DATA_POINTS),
    "Flow Rate": deque(maxlen=MAX_DATA_POINTS)
}

# Define Dashboard Sections
sections = {
    "Data Pre-processing": ["Voltage (V)", "Current (I)", "Power (P)", "Frequency (F)", "Vibration", "Temp (T)", "Flow Rate"],
    "Features Extraction": ["Time (T)", "Frequency (F)", "T-F"],
    "Features Selection": ["AI", "Hybrid"],
    "Forecasting": ["Linear Regression", "Neural Networks", "Decision Tree"],
    "Fault Diagnosis": ["Anomaly Detection", "Predictive Maintenance"]
}

# Initialize Dash App
app = dash.Dash(__name__)
app.title = "Chiller Dashboard"

# Title Bar
title_bar = html.Div(
    id="title-bar",
    children=[
        html.Button("☰", id="menu-button", n_clicks=0,
                    style={"width": "50px", "height": "50px", "background": "#0d6efd", "color": "white",
                           "border": "none", "font-size": "22px", "cursor": "pointer", "border-radius": "5px",
                           "margin-right": "15px"}),
        html.H2("Chiller Dashboard", style={"textAlign": "center", "color": "white", "margin": "0px",
                                            "flex-grow": "1", "padding": "15px"})
    ],
    style={"display": "flex", "alignItems": "center", "width": "100%", "background": "#0d6efd",
           "padding": "0px", "position": "fixed", "top": "0px", "left": "0px", "right": "0px",
           "box-shadow": "0px 2px 5px rgba(0,0,0,0.1)", "height": "60px", "z-index": "1000"}
)

# Sidebar
sidebar = html.Div(
    id="sidebar",
    children=[
        html.Div(style={"height": "60px"}),
        html.Label("Data Pre-processing:", style={"font-weight": "bold", "color": "#333"}),
        dcc.Dropdown(
            id="preprocess-dropdown",
            options=[{"label": opt, "value": opt} for opt in sections["Data Pre-processing"]],
            value=None,
            placeholder="Select an option...",
            clearable=True,
            style={"margin-bottom": "30px"}
        ),
        html.Label("Features Extraction:", style={"font-weight": "bold", "color": "#333"}),
        dcc.Dropdown(
            id="feature-extraction-dropdown",
            options=[{"label": opt, "value": opt} for opt in sections["Features Extraction"]],
            value=None,
            placeholder="Select an option...",
            clearable=True,
            style={"margin-bottom": "30px"}
        ),
        html.Label("Features Selection:", style={"font-weight": "bold", "color": "#333"}),
        dcc.Dropdown(
            id="feature-selection-dropdown",
            options=[{"label": opt, "value": opt} for opt in sections["Features Selection"]],
            value=None,
            placeholder="Select an option...",
            clearable=True,
            style={"margin-bottom": "30px"}
        ),
        html.Label("Forecasting:", style={"font-weight": "bold", "color": "#333"}),
        dcc.Dropdown(
            id="forecasting-dropdown",
            options=[{"label": opt, "value": opt} for opt in sections["Forecasting"]],
            value=None,
            placeholder="Select an option...",
            clearable=True,
            style={"margin-bottom": "30px"}
        ),
        html.Label("Fault Diagnosis:", style={"font-weight": "bold", "color": "#333"}),
        dcc.Dropdown(
            id="fault-diagnosis-dropdown",
            options=[{"label": opt, "value": opt} for opt in sections["Fault Diagnosis"]],
            value=None,
            placeholder="Select an option...",
            clearable=True,
            style={"margin-bottom": "30px"}
        ),
    ],
    style={"width": "250px", "padding": "15px", "background": "#f8f9fa", "position": "fixed",
           "left": "-250px", "top": "0px", "bottom": "0px", "box-shadow": "2px 0px 5px rgba(0,0,0,0.1)",
           "transition": "left 0.4s ease-in-out"}
)

# Main Content
content = html.Div(
    children=[
        title_bar,
        html.Div(style={"height": "80px"}),
        html.Div([
            html.H3("Selected Options: None", id="graph-title",
                    style={"textAlign": "center", "color": "#fff", "padding": "10px",
                           "background-color": "#0d6efd", "border-radius": "5px", "margin-bottom": "10px"}),
            dcc.Graph(id="live-graph", style={"height": "calc(90vh - 120px)"}),
        ], style={"border": "2px solid #0d6efd", "padding": "10px", "border-radius": "10px",
                  "background": "white", "margin": "10px", "box-shadow": "0px 4px 6px rgba(0,0,0,0.1)"}),
        dcc.Interval(id="interval-update", interval=1000, n_intervals=0)
    ],
    id="main-content",
    style={"margin-left": "0px", "padding": "15px", "background-color": "#ffffff",
           "transition": "margin-left 0.4s ease-in-out", "margin-top": "60px"}
)

# App Layout
app.layout = html.Div([sidebar, content])

# MQTT Callbacks
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        timestamp = time.strftime('%H:%M:%S')
        sensor_data["time"].append(timestamp)
        for category in sensor_data.keys():
            if category != "time" and category in payload:
                sensor_data[category].append(payload[category])
    except:
        pass

mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()

# Callbacks
@app.callback(
    [Output("sidebar", "style"), Output("main-content", "style")],
    Input("menu-button", "n_clicks"),
    [State("sidebar", "style"), State("main-content", "style")]
)
def toggle_sidebar(n_clicks, sidebar_style, content_style):
    if n_clicks is None or n_clicks % 2 == 0:
        sidebar_style["left"] = "-250px"
        content_style["margin-left"] = "0px"
    else:
        sidebar_style["left"] = "0px"
        content_style["margin-left"] = "250px"
    return sidebar_style, content_style

@app.callback(
    Output("graph-title", "children"),
    Input("preprocess-dropdown", "value")
)
def update_title(preprocess_value):
    return f"Selected Option: {preprocess_value}" if preprocess_value else "Selected Options: None"

@app.callback(
    Output("live-graph", "figure"),
    Input("interval-update", "n_intervals"),
    Input("preprocess-dropdown", "value")
)
def update_graph(n_intervals, preprocess_value):
    if not sensor_data["time"] or not preprocess_value:
        return go.Figure(
            data=[go.Scatter(x=[], y=[], mode="lines+markers")],
            layout=go.Layout(title="No Data Available", xaxis={"title": "Time"}, yaxis={"title": "Sensor Value"})
        )
    trace = go.Scatter(
        x=list(sensor_data["time"]),
        y=list(sensor_data[preprocess_value]),
        mode="lines+markers",
        name=preprocess_value
    )
    layout = go.Layout(
        title=f"Live HVAC Sensor Data: {preprocess_value}",
        xaxis={"title": "Time"},
        yaxis={"title": preprocess_value},
        template="plotly_white"
    )
    return {"data": [trace], "layout": layout}

# ✅ Expose server for deployment
server = app.server

# Local Run
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
