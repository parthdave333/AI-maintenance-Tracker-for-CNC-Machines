import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib  # <--- Use joblib instead of pickle

# Load pre-trained models
cnc_models = {
    "CNC01": joblib.load(open("CNC01_model.pkl", "rb")),
    "CNC02": joblib.load(open("CNC02_model.pkl", "rb")),
    "CNC03": joblib.load(open("CNC03_model.pkl", "rb")),
    "CNC04": joblib.load(open("CNC04_model.pkl", "rb")),
    "CNC05": joblib.load(open("CNC05_model.pkl", "rb")),
    "CNC06": joblib.load(open("CNC06_model.pkl", "rb")),
    "CNC07": joblib.load(open("CNC07_model.pkl", "rb")),
}

# Images for each machining process
process_images = {
    "Transmission Machining": "transmission_image1.jpg",
    "Engine Machining": "engine_image.jpg",
    "Cylinder Head Machining": "cylinder_head_image.jpg",
}

# Threshold values for graphs
THRESHOLDS = {
    "Control Panel Temperature (°C)": 50,
    "Spindle Motor Temperature (°C)": 80,
    "Servo Motor Temperature (°C)": 65,
}

# Apply custom CSS for button styling
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: #E9F1FA;
        font-family: 'Roboto', sans-serif;
    }}
    .stTitle {{
        font-size: 40px;
        color: #00ABE4;
        font-family: 'Montserrat', sans-serif;
        text-align: center;
    }}
    .stHeader {{
        font-size: 24px;
        color: #00ABE4;
        font-family: 'Roboto', sans-serif;
        text-align: center;
        margin-bottom: 20px;
    }}
    .stButton > button {{
        width: 100%;
        height: 60px;
        font-size: 16px;
        background-color: #00ABE4;
        color: white;
        border-radius: 10px;
        margin: 10px;
    }}
    .stButton > button:hover {{
        background-color: #008FC4;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

def main():
    # Sidebar
    st.sidebar.image("mentor_cir.png", caption="Usha Mary", width=170)
    st.sidebar.image("my_image_cir.png", caption="Praful Bhoyar", width=170)
    st.sidebar.image("red_suit_cir.png", caption="Parth Dave", width=170)

    # Header
    header = st.columns([1, 4, 1])
    with header[0]:
        st.image("organization_logo.png", width=100)
    with header[1]:
        st.markdown("<div class='stTitle'>AI Maintenance Tracker</div>", unsafe_allow_html=True)
    with header[2]:
        st.image("FINAL Technetic logo CTC.png", use_container_width=True)

    # Title
    st.markdown("<div class='stHeader'><b>Transmission & Machining Department Overview</b></div>", unsafe_allow_html=True)
    st.image("department.jpg", use_container_width=True)

    # Machining buttons
    button_col1, button_col2, button_col3 = st.columns(3)
    with button_col1:
        if st.button("Transmission Machining"):
            st.session_state["selected_process"] = "Transmission Machining"
    with button_col2:
        if st.button("Engine Machining"):
            st.session_state["selected_process"] = "Engine Machining"
    with button_col3:
        if st.button("Cylinder Head Machining"):
            st.session_state["selected_process"] = "Cylinder Head Machining"

    # Process selection
    if "selected_process" in st.session_state:
        selected_process = st.session_state["selected_process"]
        st.markdown(f"<div class='stHeader'><b>{selected_process}</b></div>", unsafe_allow_html=True)
        st.image(process_images[selected_process], use_container_width=True)

        machining_processes = {
            "Transmission Machining": ["None", "CNC01", "CNC02", "CNC03"],
            "Engine Machining": ["None", "CNC04", "CNC05", "CNC06"],
            "Cylinder Head Machining": ["None", "CNC07"],
        }

        selected_machine = st.selectbox(
            "Select a machine to Monitor", options=machining_processes[selected_process], index=0
        )

        if selected_machine != "None":
            cnc_machine_page(selected_machine)

def cnc_machine_page(machine):
    st.markdown(f"<div class='stHeader'><b>{machine} Monitoring</b></div>", unsafe_allow_html=True)

    dataset_path = f"df_{machine}.xlsx"
    if os.path.exists(dataset_path):
        data = pd.read_excel(dataset_path)

        required_columns = ["Timestamp", "Control Panel Temperature (°C)", "Spindle Motor Temperature (°C)", "Servo Motor Temperature (°C)"]
        if all(col in data.columns for col in required_columns):
            plot_graph_with_threshold(data, "Control Panel Temperature (°C)", "Timestamp")
            plot_graph_with_threshold(data, "Spindle Motor Temperature (°C)", "Timestamp")
            plot_graph_with_threshold(data, "Servo Motor Temperature (°C)", "Timestamp")
        else:
            st.error("Dataset does not contain required columns for plotting.")
    else:
        st.error(f"Dataset for {machine} not found.")

    predict_maintenance_form(machine)

def plot_graph_with_threshold(data, y_col, x_col):
    st.markdown(
        f"""
        <div style="color: green; font-size: 24px; font-weight: bold; text-align: center;">
            {y_col} vs {x_col}
        </div>
        """,
        unsafe_allow_html=True,
    )
    fig, ax = plt.subplots()
    ax.plot(data[x_col], data[y_col], label=y_col, color="green")
    ax.axhline(
        y=THRESHOLDS[y_col],
        color="red",
        linestyle="--",
        label=f"Threshold ({THRESHOLDS[y_col]} °C)",
    )
    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def predict_maintenance_form(machine):
    st.markdown(f"<div class='stHeader'>Custom Parameter Testing for {machine}</div>", unsafe_allow_html=True)

    if "prediction_result" not in st.session_state:
        st.session_state["prediction_result"] = ""

    feature_ranges = {
        "Control Panel Temperature (°C)": (0.0, 65.0),
        "Spindle Motor Temperature (°C)": (0.0, 85.0),
        "Servo Motor Temperature (°C)": (0.0, 80.0),
        "Coolant Temperature (°C)": (0.0, 50.0),
        "Coolant Flow (L/min)": (0.0, 50.0),
        "Coolant Level (%)": (0.0, 100.0),
        "Tool Wear (%)": (0.0, 75.0),
        "Spindle Speed (RPM)": (0.0, 9000.0),
        "Feed Rate (mm/min)": (0.0, 4000.0),
        "Vibration (mm/s)": (0.0, 0.8),
        "Fan Speed (RPM)": (0.0, 2000.0),
        "Power Consumption (kW)": (0.0, 25.0),
        "Cycle Time (mins)": (0.0, 1440.0),
        "Idle Time (mins)": (0.0, 1440.0),
        "Axis Load (X, Y, Z)": (0.0, 85.0),
        "Ambient Temperature (°C)": (0.0, 50.0),
        "Hydraulic Pressure (bar)": (0.0, 60.0),
        "Tool Breakage (Yes/No)": ("No", "Yes"),
        "Status (Running/Stopped)": ("Stopped", "Running"),
    }

    with st.form("input_form"):
        inputs = {}
        cols = st.columns(3)
        for i, (feature, limits) in enumerate(feature_ranges.items()):
            col = cols[i % 3]
            if isinstance(limits[0], str):
                inputs[feature] = col.selectbox(feature, options=limits)
            else:
                inputs[feature] = col.number_input(
                    feature, min_value=limits[0], max_value=limits[1], value=limits[0]
                )

        submit_button = st.form_submit_button("Submit")

        if submit_button:
            input_df = pd.DataFrame([inputs])
            input_df["Tool Breakage (Yes/No)"] = input_df["Tool Breakage (Yes/No)"].map({"No": 0, "Yes": 1})
            input_df["Status (Running/Stopped)"] = input_df["Status (Running/Stopped)"].map({"Stopped": 0, "Running": 1})

            model = cnc_models[machine]
            expected_features = model.feature_names_in_

            for feature in expected_features:
                if feature not in input_df.columns:
                    input_df[feature] = 0

            input_df = input_df[expected_features]

            prediction = model.predict(input_df)[0]
            st.session_state["prediction_result"] = (
                f"{machine} Requires Maintenance" if prediction == 1 else f"{machine} Requires no Maintenance"
            )

    if st.session_state["prediction_result"]:
        color = "red" if "Requires Maintenance" in st.session_state["prediction_result"] else "green"
        st.markdown(
            f"""
            <div style="border: 2px solid {color}; padding: 10px; border-radius: 10px; text-align: center; color: {color}; font-weight: bold; font-size: 20px;">
                {st.session_state["prediction_result"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

if __name__ == "__main__":
    main()
