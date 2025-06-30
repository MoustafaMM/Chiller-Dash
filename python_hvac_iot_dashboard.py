import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State, ALL # ALL might not be needed for current changes
import plotly.graph_objs as go
import paho.mqtt.client as mqtt
import time
import json
from collections import deque
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from scipy.stats import zscore # Ensure zscore is imported if used for fd_z_score

# MQTT Broker Settings
MQTT_BROKER = "mqtt.eclipseprojects.io"
MQTT_PORT = 1883
MQTT_TOPIC = "hvac/sensor_detailed"

# --- Define Sensor Groupings based on your tables ---
sensor_groupings = {
    "Temperature": [
        {"label": "Chilled Water Setpoint (TWE_set)", "value": "TWE_set"},
        {"label": "Evaporator Water In (TEI)", "value": "TEI"},
        {"label": "Evaporator Water Out (TEO)", "value": "TEO"},
        {"label": "Condenser Water In (TCI)", "value": "TCI"},
        {"label": "Condenser Water Out (TCO)", "value": "TCO"},
        {"label": "Shared HX Water In (TSI)", "value": "TSI"},
        {"label": "Shared HX Water Out (TSO)", "value": "TSO"},
        {"label": "Building Water In (TBI)", "value": "TBI"},
        {"label": "Building Water Out (TBO)", "value": "TBO"},
        {"label": "City Water In (TWI)", "value": "TWI"},
        {"label": "City Water Out (TWO)", "value": "TWO"},
        {"label": "Sat. Refrigerant Temp Evaporator (TRE)", "value": "TRE"},
        {"label": "Refrigerant Subcooling (TRC_sub)", "value": "TRC_sub"},
        {"label": "Refrigerant Suction Temp (T_suc)", "value": "T_suc"},
        {"label": "Refrigerant Suction Superheat (Tsh_suc)", "value": "Tsh_suc"},
        {"label": "Refrigerant Discharge Temp (TR_dis)", "value": "TR_dis"},
        {"label": "Refrigerant Discharge Superheat (Tsh_dis)", "value": "Tsh_dis"},
        {"label": "Evaporator Approach Temp (TEA)", "value": "TEA"},
        {"label": "Condenser Approach Temp (TCA)", "value": "TCA"},
        {"label": "Oil Sump Temp (TO_sump)", "value": "TO_sump"},
        {"label": "Oil Feed Temp (TO_feed)", "value": "TO_feed"},
        {"label": "Condenser Water Temp Diff (TWCD)", "value": "TWCD"},
        {"label": "Evaporator Water Temp Diff (TWED)", "value": "TWED"},
    ],
    "Pressure": [
        {"label": "Refrigerant Evaporator Pressure (PRE)", "value": "PRE"},
        {"label": "Refrigerant Condenser Pressure (PRC)", "value": "PRC"},
        {"label": "Compressor Pressure Lift (P_lift)", "value": "P_lift"},
        {"label": "Oil Feed Pressure (PO_feed)", "value": "PO_feed"},
        {"label": "Oil Net Pressure (PO_net)", "value": "PO_net"},
    ],
    "Flow Rate": [
        {"label": "Evaporator Water Flow (FWE)", "value": "FWE"},
        {"label": "Condenser Water Flow (FWC)", "value": "FWC"},
        {"label": "City Water Flow (FWW)", "value": "FWW"},
        {"label": "Hot Water Flow (FWH)", "value": "FWH"},
        {"label": "Condenser Water Bypass Flow (FWB)", "value": "FWB"},
    ],
    "Power and Performance": [
        {"label": "Compressor Power (KW)", "value": "KW"},
        {"label": "Amps", "value": "Amps"},
        {"label": "Percent Max Rated Load Amps (RLA%)", "value": "RLA%"},
        {"label": "Heat Balance (kW)", "value": "Heat Balance (kW)"},
        {"label": "Coefficient of Performance (COP)", "value": "COP"},
        {"label": "Compressor Efficiency (kW/ton)", "value": "kW/ton"},
        {"label": "Condenser Heat Rejection (Cond Tons)", "value": "Cond Tons"},
        {"label": "Evaporator Cooling Rate (Evap Tons)", "value": "Evap Tons"},
    ],
    "Valve Position": [
        {"label": "Small Steam Valve (VSS)", "value": "VSS"},
        {"label": "Large Steam Valve (VSL)", "value": "VSL"},
        {"label": "Hot Water Valve (VH)", "value": "VH"},
        {"label": "3-way Mixing Valve (VM)", "value": "VM"},
        {"label": "Condenser Valve (VC)", "value": "VC"},
        {"label": "Evaporator Valve (VE)", "value": "VE"},
        {"label": "City Water Valve (VW)", "value": "VW"},
    ],
}

# Live Sensor Data Storage
MAX_DATA_POINTS = 50
sensor_data = {"time": deque(maxlen=MAX_DATA_POINTS)}
all_sensor_keys = []
for group_name, sensors_in_group in sensor_groupings.items():
    for sensor_item in sensors_in_group:
        all_sensor_keys.append(sensor_item['value'])
        sensor_data[sensor_item['value']] = deque(maxlen=MAX_DATA_POINTS)

# Define Pre-processing Method Options
preprocessing_methods = [
    {"label": "Spike Removal (Placeholder)", "value": "spike_removal_placeholder"},
    {"label": "Fill: Forward/Backward Fill", "value": "fill_ffill_bfill"},
    {"label": "Fill: Linear Interpolation", "value": "fill_linear_interpolate"},
    {"label": "Fill: Mean Imputation", "value": "fill_mean"},
    {"label": "Scale: Min-Max Scaling", "value": "scale_min_max"},
    {"label": "Scale: Z-score Standardization", "value": "scale_z_score"},
    {"label": "Scale: Robust Scaling", "value": "scale_robust"},
]

# Fault Diagnosis Options
fault_diagnosis_options_list = [
    {"label": "Anomaly Detection (Z-score)", "value": "fd_z_score"},
    {"label": "Isolation Forest (Placeholder)", "value": "fd_isolation_forest"},
    {"label": "Autoencoder (Placeholder)", "value": "fd_autoencoder"},
    {"label": "SVM Classifier (Placeholder)", "value": "fd_svm_classifier"},
    {"label": "Rule-Based System (Placeholder)", "value": "fd_rule_based"}
]

# Define Advanced Dashboard Sections for Sidebar
sidebar_sections = {
    "Features Extraction": ["Time (T)", "Frequency (F)", "T-F Domain"],
    "Feature Selection": ["Filter Methods (Placeholder)", "Wrapper Methods (Placeholder)", "Embedded Methods (Placeholder)", "PCA (Placeholder)"],
    "Forecasting": [
        "ARIMA", "LSTM (Neural Network)", "Prophet",
        "Support Vector Regression (SVR)", "Random Forest Regressor"
    ],
    "Fault Diagnosis": fault_diagnosis_options_list
}

# Initialize Dash App
app = dash.Dash(__name__, suppress_callback_exceptions=True)
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
sidebar_type_options = [{"label": group_name, "value": group_name} for group_name in sensor_groupings.keys()]
sidebar = html.Div(
    id="sidebar",
    children=[
        html.Div(style={"height": "60px"}),  # Spacer for title bar
        html.Label("Data Collection Visualization:", style={"font-weight": "bold", "color": "#333"}),
        dcc.Dropdown(
            id="data-collection-type-dropdown",
            options=sidebar_type_options,
            placeholder="Select Sensor Group...",
            style={"margin-bottom": "5px"}
        ),
        dcc.Dropdown(
            id="data-collection-sensor-dropdown",
            placeholder="Select Sensor...",
            style={"margin-bottom": "20px"}
        ),

        html.Label("Data Pre-processing:", style={"font-weight": "bold", "color": "#333"}),
        dcc.Dropdown(
            id="preprocess-method-dropdown",
            options=preprocessing_methods,
            placeholder="Select Method...",
            style={"margin-bottom": "5px"}
        ),
        dcc.Dropdown(
            id="preprocess-sensorgroup-dropdown",
            options=sidebar_type_options,
            placeholder="Select Sensor Group...",
            style={"margin-bottom": "5px"}
        ),
        dcc.Dropdown(
            id="preprocess-sensor-dropdown",
            placeholder="Select Sensor for Display...",
            style={"margin-bottom": "20px"}
        ),

        html.Label("Features Extraction:", style={"font-weight": "bold", "color": "#333"}),
        dcc.Dropdown(
            id="feature-extraction-dropdown", # Controls the shared graph
            options=[{"label": opt, "value": opt} for opt in sidebar_sections["Features Extraction"]],
            placeholder="Select feature extraction...",
            style={"margin-bottom": "5px"}
        ),

        html.Label("Feature Selection:", style={"font-weight": "bold", "color": "#333"}), # New Section
        dcc.Dropdown(
            id="feature-selection-dropdown", # Controls the shared graph
            options=[{"label": opt, "value": opt} for opt in sidebar_sections["Feature Selection"]],
            placeholder="Select feature selection...",
            style={"margin-bottom": "20px"}
        ),

        html.Label("Forecasting:", style={"font-weight": "bold", "color": "#333"}),
        dcc.Dropdown(
            id="forecasting-dropdown",
            options=[{"label": opt, "value": opt} for opt in sidebar_sections["Forecasting"]],
            placeholder="Select forecasting method...",
            style={"margin-bottom": "20px"}
        ),

        html.Label("Fault Diagnosis:", style={"font-weight": "bold", "color": "#333"}),
        dcc.Dropdown(
            id='fault-diagnosis-method-sidebar-dropdown',
            options=sidebar_sections["Fault Diagnosis"], # Uses the fault_diagnosis_options_list
            placeholder="Select diagnosis method...",
            style={"margin-bottom": "5px"}
        ),
        dcc.Dropdown(
            id='fd-sensorgroup-sidebar-dropdown',
            options=sidebar_type_options,
            placeholder="Select sensor group for diagnosis...",
            style={"margin-bottom": "5px"}
        ),
        dcc.Dropdown(
            id='fd-sensor-sidebar-dropdown',
            placeholder="Select sensor for diagnosis...",
            style={"margin-bottom": "10px"}
        ),
        html.Button("Run Diagnosis", id='fd-run-button', n_clicks=0, className="btn btn-primary", style={"margin-bottom": "20px", "width":"100%"}),

    ],
    style={"width": "280px", "padding": "15px", "background": "#f8f9fa", "position": "fixed",
           "left": "-280px", "top": "0px", "bottom": "0px", "box-shadow": "2px 0px 5px rgba(0,0,0,0.1)",
           "transition": "left 0.4s ease-in-out", "overflow-y": "auto", "z-index": "999"}
)

# Main monitoring dashboard layout (original 4 graphs)
monitoring_dashboard_layout = html.Div([
    html.Div([
        dcc.Graph(id="graph-data-collection", style={"height": "40vh"})
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '5px',
              "border": "1px solid #0d6efd", "border-radius": "5px", "margin-bottom":"10px"}),
    html.Div([
        dcc.Graph(id="graph-preprocessing", style={"height": "40vh"})
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '5px',
              "border": "1px solid #0d6efd", "border-radius": "5px", "margin-bottom":"10px"}),
    html.Div([
        dcc.Graph(id="graph-feature-extraction", style={"height": "40vh"}) # This graph is now shared
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '5px',
              "border": "1px solid #0d6efd", "border-radius": "5px"}),
    html.Div([
        dcc.Graph(id="graph-forecasting-diagnosis", style={"height": "40vh"}), # This graph is shared by Forecast/FD Run
        # Removed the message div from here
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '5px',
              "border": "1px solid #0d6efd", "border-radius": "5px"}),
])

# Main Content
content = html.Div(
    children=[
        html.Div(style={"height": "60px"}),
        dcc.Tabs(id="app-tabs", value='tab-monitoring', children=[
            dcc.Tab(label='Monitoring Dashboard', value='tab-monitoring', children=[
                monitoring_dashboard_layout
            ]),
        ]),
        dcc.Interval(id="interval-update", interval=1000, n_intervals=0)
    ],
    id="main-content",
    style={"margin-left": "0px", "padding": "10px", "background-color": "#e9ecef",
           "transition": "margin-left 0.4s ease-in-out", "padding-top": "60px"}
)

# App Layout
app.layout = html.Div([title_bar, sidebar, content])

# --- MQTT Callbacks ---
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0: print("Connected to MQTT Broker!"); client.subscribe(MQTT_TOPIC)
    else: print(f"Failed to connect, return code {rc}\n")

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        timestamp = time.strftime('%H:%M:%S')
        sensor_data["time"].append(timestamp)
        for key in all_sensor_keys:
            value = payload.get(key)
            if value == "" or value is None: sensor_data[key].append(None)
            else:
                try: sensor_data[key].append(float(value))
                except (ValueError, TypeError): sensor_data[key].append(None)
    except Exception as e: print(f"Error processing message: {e} with payload: {msg.payload.decode()}")

try: mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
except AttributeError: mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect; mqtt_client.on_message = on_message
try: mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
except Exception as e: print(f"MQTT Connection Error: {e}")
mqtt_client.loop_start()

# --- Callbacks for Cascading Dropdowns ---
def update_sensor_options(type_value):
    if not type_value: return []
    return sensor_groupings.get(type_value, [])
def reset_sensor_value(_): return None
app.callback(Output('data-collection-sensor-dropdown', 'options'), Input('data-collection-type-dropdown', 'value'))(update_sensor_options)
app.callback(Output('data-collection-sensor-dropdown', 'value'), Input('data-collection-sensor-dropdown', 'options'))(reset_sensor_value)
app.callback(Output('preprocess-sensor-dropdown', 'options'), Input('preprocess-sensorgroup-dropdown', 'value'))(update_sensor_options)
app.callback(Output('preprocess-sensor-dropdown', 'value'), Input('preprocess-sensor-dropdown', 'options'))(reset_sensor_value)
app.callback(Output('fd-sensor-sidebar-dropdown', 'options'), Input('fd-sensorgroup-sidebar-dropdown', 'value'))(update_sensor_options)
app.callback(Output('fd-sensor-sidebar-dropdown', 'value'), Input('fd-sensor-sidebar-dropdown', 'options'))(reset_sensor_value)

# --- Callbacks for mutually exclusive Feature Extraction / Selection dropdowns ---
@app.callback(
    Output('feature-selection-dropdown', 'value', allow_duplicate=True),
    Input('feature-extraction-dropdown', 'value'),
    prevent_initial_call=True # Important to prevent reset on load
)
def reset_selection_on_extraction(extraction_val):
    if extraction_val:
        return None # Reset feature selection
    return dash.no_update

@app.callback(
    Output('feature-extraction-dropdown', 'value', allow_duplicate=True),
    Input('feature-selection-dropdown', 'value'),
    prevent_initial_call=True # Important to prevent reset on load
)
def reset_extraction_on_selection(selection_val):
    if selection_val:
        return None # Reset feature extraction
    return dash.no_update

# --- Other Callbacks ---
@app.callback(
    [Output("sidebar", "style"), Output("main-content", "style")],
    Input("menu-button", "n_clicks"),
    [State("sidebar", "style"), State("main-content", "style")]
)
def toggle_sidebar(n_clicks, sidebar_style, content_style):
    if sidebar_style is None: sidebar_style = {}
    if content_style is None: content_style = {}
    if n_clicks is None: n_clicks = 0 # Initialize if None
    if n_clicks % 2 != 0: # If odd, open sidebar
        sidebar_style["left"] = "0px"
        content_style["margin-left"] = "280px"
    else: # If even (or 0), close sidebar
        sidebar_style["left"] = "-280px"
        content_style["margin-left"] = "0px"
    return sidebar_style, content_style

def create_placeholder_figure(title_text="No Data or Option Selected"):
    layout_args = {"title": title_text, "xaxis": {"title": "Time"}, "yaxis": {"title": "Value"}, "template": "plotly_white", "margin":dict(l=40, r=20, t=40, b=30)}
    return go.Figure(data=[go.Scatter(x=[], y=[])], layout=go.Layout(**layout_args))

# --- Monitoring Tab Graph Update Callbacks ---
@app.callback(Output("graph-data-collection", "figure"), Input("interval-update", "n_intervals"), Input("data-collection-sensor-dropdown", "value"))
def update_graph_data_collection(n_intervals, selected_sensor):
    if not sensor_data["time"] or not selected_sensor or selected_sensor not in sensor_data:
        return create_placeholder_figure(f"Data Collection: {selected_sensor or 'None Selected'}")
    sensor_label = selected_sensor
    for group in sensor_groupings.values():
        for item in group:
            if item['value'] == selected_sensor: sensor_label = item['label']; break
        if sensor_label != selected_sensor: break
    y_data = [val if pd.notnull(val) else None for val in sensor_data[selected_sensor]]
    trace = go.Scatter(x=list(sensor_data["time"]), y=y_data, mode="lines+markers", name=sensor_label, connectgaps=False)
    layout = go.Layout(title=f"Data Collection: {sensor_label}", xaxis_title="Time", yaxis_title=selected_sensor, template="plotly_white", margin=dict(l=40,r=20,t=40,b=30))
    return {"data": [trace], "layout": layout}

@app.callback(Output("graph-preprocessing", "figure"), Input("interval-update", "n_intervals"), Input("preprocess-method-dropdown", "value"), Input("preprocess-sensor-dropdown", "value"))
def update_graph_preprocessing(n_intervals, selected_method, selected_sensor):
    method_label = "None"; y_data_to_plot = []
    if selected_method: method_label = next((item['label'] for item in preprocessing_methods if item['value'] == selected_method), selected_method)
    if not sensor_data["time"] or not selected_sensor or selected_sensor not in sensor_data or not selected_method:
        return create_placeholder_figure(f"Pre-processing ({method_label}): {selected_sensor or 'None Selected'} - Select Method & Sensor")
    sensor_label = selected_sensor
    for group in sensor_groupings.values():
        for item in group:
            if item['value'] == selected_sensor: sensor_label = item['label']; break
        if sensor_label != selected_sensor: break
    raw_data_list = list(sensor_data[selected_sensor]); data_series = pd.Series(raw_data_list, dtype=np.float64); processed_y_data = data_series.copy()
    try:
        # Handle case where all data is NaN before attempting methods that require valid points
        all_nan_for_scaling = data_series.dropna().empty and selected_method in ["scale_min_max", "scale_z_score", "scale_robust"]
        if all_nan_for_scaling:
            processed_y_data = pd.Series(np.nan, index=data_series.index, dtype=np.float64) # Keep all as NaN for scaling if no valid data
        elif selected_method == "fill_ffill_bfill": processed_y_data = data_series.ffill().bfill()
        elif selected_method == "fill_linear_interpolate": processed_y_data = data_series.interpolate(method='linear')
        elif selected_method == "fill_mean": mean_val = data_series.mean(); processed_y_data = data_series.fillna(mean_val if pd.notna(mean_val) else 0)
        elif selected_method == "scale_min_max":
            valid_data = data_series.dropna()
            if not valid_data.empty:
                scaler = MinMaxScaler(); scaled_values = scaler.fit_transform(valid_data.values.reshape(-1,1))
                processed_y_data = pd.Series(np.nan, index=data_series.index, dtype=np.float64); processed_y_data[valid_data.index] = scaled_values.flatten()
            else: processed_y_data = pd.Series(np.nan, index=data_series.index, dtype=np.float64)
        elif selected_method == "scale_z_score":
            valid_data = data_series.dropna()
            if not valid_data.empty:
                std_val = valid_data.std()
                if std_val > 0: mean_val = valid_data.mean(); scaled_values = (valid_data - mean_val) / std_val
                else: scaled_values = pd.Series(0.0, index=valid_data.index) # All same valid values, z-score is 0
                processed_y_data = pd.Series(np.nan, index=data_series.index, dtype=np.float64); processed_y_data[valid_data.index] = scaled_values
            else: processed_y_data = pd.Series(np.nan, index=data_series.index, dtype=np.float64)
        elif selected_method == "scale_robust":
            valid_data_indices = data_series.notna(); valid_data_values = data_series[valid_data_indices].values.reshape(-1, 1)
            if valid_data_values.shape[0] > 0:
                scaled_values = RobustScaler().fit_transform(valid_data_values)
                processed_y_data = pd.Series(np.nan, index=data_series.index, dtype=np.float64); processed_y_data[valid_data_indices] = scaled_values.flatten()
            else: processed_y_data = pd.Series(np.nan, index=data_series.index, dtype=np.float64)
        elif selected_method == "spike_removal_placeholder": processed_y_data = data_series
        y_data_to_plot = processed_y_data.where(pd.notnull(processed_y_data), None).tolist()
    except Exception as e:
        print(f"Error pre-processing {selected_sensor} with {selected_method}: {e}")
        y_data_to_plot = [val if pd.notnull(val) else None for val in sensor_data[selected_sensor]] # Fallback
    trace = go.Scatter(x=list(sensor_data["time"]), y=y_data_to_plot, mode="lines+markers", name=f"{sensor_label} ({method_label})", connectgaps=False)
    layout = go.Layout(title=f"Pre-processing - Method: {method_label}<br>Sensor: {sensor_label}", xaxis_title="Time", yaxis_title="Value (Processed)", template="plotly_white", margin=dict(l=40,r=20,t=60,b=30))
    return {"data": [trace], "layout": layout}

@app.callback(
    Output("graph-feature-extraction", "figure"), # This graph is now shared
    Input("interval-update", "n_intervals"),
    Input("feature-extraction-dropdown", "value"),
    Input("feature-selection-dropdown", "value") # New input
)
def update_graph_features_shared(n_intervals, extraction_value, selection_value):
    # The reset callbacks should ensure only one of extraction_value or selection_value has a value.
    # If somehow both are set (e.g. initial state before reset logic fully settles), prioritize one.
    
    final_selected_value = None
    graph_title_prefix = ""

    if extraction_value:
        final_selected_value = extraction_value
        graph_title_prefix = "Feature Extraction"
    elif selection_value:
        final_selected_value = selection_value
        graph_title_prefix = "Feature Selection"

    if not final_selected_value:
        return create_placeholder_figure("Select Feature Extraction or Selection Method")

    # Placeholder: In a real app, this would plot actual extracted/selected feature results
    # or data related to the method. For now, it might try to plot raw sensor data if the
    # selected method name matches a sensor key (unlikely for method names).
    
    title_text = f"{graph_title_prefix}: {final_selected_value}"
    
    # Attempt to plot raw data if the selected value happens to be a sensor key (placeholder behavior)
    if final_selected_value in sensor_data and sensor_data["time"] and any(d is not None for d in sensor_data[final_selected_value]):
        y_data = [val if pd.notnull(val) else None for val in sensor_data[final_selected_value]]
        trace = go.Scatter(x=list(sensor_data["time"]), y=y_data, mode="lines+markers", name=final_selected_value, connectgaps=False)
        layout = go.Layout(title=title_text, xaxis_title="Time", yaxis_title=final_selected_value, template="plotly_white", margin=dict(l=40,r=20,t=40,b=30))
        return {"data": [trace], "layout": layout}
    else:
        # If not a direct sensor key, show method name and "Not Implemented"
        return create_placeholder_figure(f"{title_text} (Method not implemented or no direct data match)")


@app.callback(
    Output("graph-forecasting-diagnosis", "figure"), # Removed Output for message
    Input("interval-update", "n_intervals"),
    Input("forecasting-dropdown", "value"),
    Input("fd-run-button", "n_clicks"), # For triggering FD
    State("fault-diagnosis-method-sidebar-dropdown", "value"), # Method for FD
    State("fd-sensor-sidebar-dropdown", "value") # Sensor for FD
)
def update_graph_forecasting_and_diagnosis(n_intervals, forecast_method_selected, fd_run_clicks,
                                           fd_method, fd_sensor):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    # Determine the "active" mode for this graph: 'forecasting', 'fault_diagnosis', or 'default_placeholder'
    active_mode = 'default_placeholder'
    fig_title = "Forecasting / Fault Diagnosis" # Generic title for placeholder

    # Check if FD run button was clicked. This takes precedence.
    if triggered_id == "fd-run-button" and fd_run_clicks > 0:
        active_mode = 'fault_diagnosis'
    # If not FD button, check if forecasting dropdown was triggered or if it's an interval update
    # and a forecasting method is currently selected (and no FD run has happened more recently).
    elif forecast_method_selected and (triggered_id == "forecasting-dropdown" or (triggered_id == "interval-update" and (fd_run_clicks == 0 or (ctx.triggered_inputs.get('fd-run-button.n_clicks') is None)))):
        active_mode = 'forecasting'
    # If it's an interval update and FD was the last explicit action, keep showing FD results
    elif triggered_id == "interval-update" and fd_method and fd_sensor and fd_run_clicks > 0:
        active_mode = 'fault_diagnosis_sticky' # A variation to show existing FD results

    # --- Build Figure based on active_mode ---
    if active_mode == 'forecasting':
        fig_title = f"Forecasting: {forecast_method_selected}"
        # Placeholder for actual forecasting logic
        fig = create_placeholder_figure(f"{fig_title} (Not Implemented)")

    elif active_mode == 'fault_diagnosis' or active_mode == 'fault_diagnosis_sticky':
        if not fd_method or not fd_sensor:
            fig = create_placeholder_figure("FD: Select method & sensor, then run.")
            return fig

        if fd_sensor not in sensor_data or not any(pd.notnull(val) for val in sensor_data[fd_sensor]):
            fig = create_placeholder_figure(f"FD: No valid data for {fd_sensor}")
            return fig
        
        current_times = list(sensor_data["time"])
        # Ensure data_series conversion handles potential non-float values if any crept in earlier
        data_values = [val if pd.notnull(val) else np.nan for val in sensor_data[fd_sensor]]
        data_series = pd.Series(data_values, dtype=np.float64)

        sensor_label_fd = next((s_item['label'] for s_group in sensor_groupings.values() for s_item in s_group if s_item['value'] == fd_sensor), fd_sensor)
        method_label_fd = next((item['label'] for item in fault_diagnosis_options_list if item['value'] == fd_method), fd_method)
        
        fig_traces = []
        fig_title_fd = f"{method_label_fd} on {sensor_label_fd}"

        if fd_method == "fd_z_score":
            valid_data = data_series.dropna()
            if valid_data.empty:
                fig_traces.append(go.Scatter(x=current_times, y=data_series.tolist(), mode='lines+markers', name=f'{sensor_label_fd} (No valid data)', line=dict(color='grey'), connectgaps=False))
            else:
                from scipy.stats import zscore # Ensure import
                scores = zscore(valid_data)
                full_scores = pd.Series(np.nan, index=data_series.index, dtype=np.float64)
                full_scores.loc[valid_data.index] = scores # Use .loc for safe assignment
                
                threshold = 3.0
                anomalies_mask = abs(full_scores) > threshold
                anomaly_points_series = data_series[anomalies_mask]

                fig_traces.append(go.Scatter(x=current_times, y=data_series.tolist(), mode='lines+markers', name=f'{sensor_label_fd} Data', line=dict(color='blue'), connectgaps=False))
                if not anomaly_points_series.empty:
                    # Get original indices of anomalies to map to current_times
                    anomaly_time_indices = [i for i, (idx, val) in enumerate(anomalies_mask.items()) if val]
                    anomaly_times_plot = [current_times[i] for i in anomaly_time_indices]
                    anomaly_values_plot = anomaly_points_series.tolist()
                    
                    fig_traces.append(go.Scatter(x=anomaly_times_plot, y=anomaly_values_plot, mode='markers', name=f'Anomaly (Z > {threshold})', marker=dict(color='red', size=10, symbol='x')))
                
            fig = go.Figure(data=fig_traces, layout=go.Layout(title=fig_title_fd, xaxis_title="Time", yaxis_title="Value", template="plotly_white", margin=dict(l=40,r=20,t=60,b=30)))
        
        elif fd_method in ["fd_isolation_forest", "fd_autoencoder", "fd_svm_classifier", "fd_rule_based"]:
            fig = go.Figure(data=[go.Scatter(x=current_times, y=data_series.tolist(), mode='lines+markers', name=f'{sensor_label_fd} Data', connectgaps=False)],
                            layout=go.Layout(title=f"{fig_title_fd} (Placeholder)", template="plotly_white", margin=dict(l=40,r=20,t=60,b=30)))
        else:
            fig = create_placeholder_figure(f"FD: {method_label_fd} on {sensor_label_fd} (Unknown)")
        
    else: # default_placeholder
        fig = create_placeholder_figure(fig_title)

    return fig


# ✅ Expose server for deployment
server = app.server

# Local Run
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)