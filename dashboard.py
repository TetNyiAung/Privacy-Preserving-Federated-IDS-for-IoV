import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import base64
from streamlit_autorefresh import st_autorefresh

# Paths
RESULTS_FILE = "results/results_log.csv"
SUMMARY_FILE = "results/summary.txt"
PLOTS_DIR = "plots"

# Streamlit Config
st.set_page_config(
    page_title="PP-FedIDS Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================
# Header & Introduction
# ====================
st.markdown("""
    <div style='text-align: center; padding-top: 10px;'>
        <h1 style='color: #FF2D00;'>PP-FedIDS Dashboard</h1>
        <h3 style='color: #c9c9c9;'>Privacy-Preserving Federated Learning for Intrusion Detection in IoV</h3>
    </div>
""", unsafe_allow_html=True)

with st.expander("About the System"):
    st.markdown("""
    **PP-FedIDS** is a federated learning system designed to detect malicious behaviors (attacks) in vehicular networks.  
    Instead of uploading sensitive driving data, each client (vehicle) trains locally and only shares model updates.  
    This dashboard shows real-time results and performance metrics of the Intrusion Detection System (IDS).

    **Key Features:**
    - Preserves privacy while detecting model poisoning, label flipping, and noisy updates.
    - Monitors IDS effectiveness round-by-round.
    - Provides visual analysis, logs, and downloadable reports.
    """)

# Auto-refresh every 10 seconds
st_autorefresh(interval=10000, key="refresh")

# ====================
# Load results
# ====================
@st.cache_data
def load_results():
    if os.path.exists(RESULTS_FILE):
        return pd.read_csv(RESULTS_FILE)
    return pd.DataFrame()

df = load_results()
if df.empty:
    st.warning("No results found. Please run the training (server.py and clients) first.")
    st.stop()

# ====================
# Current Metrics
# ====================
st.markdown("## Current IDS Metrics")
st.markdown("The most recent evaluation of IDS effectiveness for detecting malicious clients.")

latest = df.iloc[-1]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{latest['Accuracy']:.2f}")
col2.metric("Precision", f"{latest['Precision']:.2f}")
col3.metric("Recall", f"{latest['Recall']:.2f}")
col4.metric("F1-Score", f"{latest['F1']:.2f}")

# ====================
# Trend Charts
# ====================
st.markdown("---")
st.markdown("## Performance Trends Over Rounds")
st.markdown("Visual analysis of how the IDS evolves over time.")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Accuracy, Precision, Recall, F1 Score")
    st.line_chart(df[["Accuracy", "Precision", "Recall", "F1"]].set_index(df["round"]))

with col2:
    st.markdown("#### Confusion Matrix Metrics")
    st.bar_chart(df[["TP", "FP", "FN", "TN"]].set_index(df["round"]))

# ====================
# Logs Per Round (Fixed Display)
# ====================
st.markdown("---")
st.markdown("## Client Status Per Round")
st.markdown("Displays detailed logs from each federated round, including client behaviors, flags, and IDS results.")

if os.path.exists(SUMMARY_FILE):
    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
        summary_text = f.read()

    # Display with proper alignment using st.code (auto monospace + preserves spacing)
    st.markdown(
        """
        <div style="display: flex; justify-content: center;">
            <div style="width: 90%;">
        """, unsafe_allow_html=True
    )
    st.code(summary_text, language="text")
    st.markdown("</div></div>", unsafe_allow_html=True)

else:
    st.info("Summary log not found. Please complete training.")

# ====================
# Average Metrics
# ====================
st.markdown("---")
st.markdown("## Average Metrics Across Rounds")
st.markdown("Mean values from all training rounds to evaluate IDS performance long-term.")

avg_df = df[["Accuracy", "Precision", "Recall", "F1"]].mean().round(2).to_frame("Mean").T

st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
st.dataframe(
    avg_df.style.set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center')]},
        {'selector': 'td', 'props': [('text-align', 'center')]}
    ]),
    use_container_width=False
)
st.markdown("</div>", unsafe_allow_html=True)

# ====================
# Full Results Log
# ====================
st.markdown("---")
st.markdown("## All Federated Rounds - IDS Logs")
st.markdown("Tabular format of all detection results per round.")

st.dataframe(
    df.style.set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center')]},
        {'selector': 'td', 'props': [('text-align', 'center')]}
    ]),
    use_container_width=True
)

# ====================
# Visual Plots Section
# ====================
st.markdown("---")
st.markdown("## Visual Analysis of IDS Behavior")
st.markdown("These charts are generated using `plot_results.py` to better understand IDS performance.")

if os.path.exists(PLOTS_DIR):
    images = [f for f in os.listdir(PLOTS_DIR) if f.endswith(".png")]
    if images:
        for image in images:
            with open(os.path.join(PLOTS_DIR, image), "rb") as img_file:
                b64_encoded = base64.b64encode(img_file.read()).decode()

            st.markdown(f"""
                <div style="text-align: center; margin-bottom: 30px;">
                    <img src="data:image/png;base64,{b64_encoded}" style="width:60%; border-radius: 8px; border: 1px solid #ccc;" />
                    <p style="font-weight: bold; margin-top: 10px;">{image}</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No plots found. Please run `plot_results.py` to generate visualizations.")
else:
    st.warning("The `plots/` folder was not found.")

# ====================
# Download Section
# ====================
st.markdown("---")
st.markdown("## Download Reports")
st.markdown("Download logs for research, presentations, or offline documentation.")

col1, col2 = st.columns(2)
with col1:
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "rb") as f:
            st.download_button("Download CSV Results", data=f, file_name="results_log.csv", mime="text/csv")

with col2:
    if os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE, "rb") as f:
            st.download_button("Download Summary Report", data=f, file_name="summary.txt", mime="text/plain")

# ====================
# Footer
# ====================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 13px;'>
        &copy; 2025 - Tet Nyi Aung (24844487) | PP-FedIDS | MSc Cybersecurity Final Project
    </div>
""", unsafe_allow_html=True)
