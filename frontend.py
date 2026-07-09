import streamlit as st
import requests
from PIL import Image
import io
import pandas as pd

# Define UI dimensions and base tracking context details
st.set_page_config(page_title="Smart Campus Surveillance", layout="wide")
st.title("🚨 Smart Campus Fire & Smoke Detection Dashboard")

tab1, tab2 = st.tabs(["🎥 Live Camera Stream & Upload", "📊 System Analytics & Logs"])

with tab1:
    st.header("Surveillance Feed & Image Verification")
    
    # Configure input selectors for verification frames
    input_option = st.radio("Select Input Source:", ("Use Webcam", "Upload Image File (.jpg, .jpeg, .png)"))
    img_file = None
    
    if input_option == "Use Webcam":
        img_file = st.camera_input("Capture live tracking validation feed frame")
    else:
        img_file = st.file_uploader("Select input target frame configuration image...", type=["jpg", "jpeg", "png"])
    
    if img_file is not None:
        bytes_data = img_file.getvalue()
        st.image(bytes_data, caption="Selected Frame Instance", width=500)
        
        with st.spinner("Streaming data matrices onto vision infrastructure..."):
            try:
                files = {"file": ("frame.jpg", bytes_data, "image/jpeg")}
                response = requests.post("http://127.0.0.1:8000/predict-frame", files=files)
                
                if response.status_code == 200:
                    res_data = response.json()["data"]
                    st.success(f"Inference complete. Visual structures identified: {res_data['total_objects_detected']}")
                    
                    if res_data["cloud_url"] and res_data["cloud_url"] != "Cloud Storage Connection Loss":
                        # Process spatial coordinates overlay received from cloud targets
                        st.image(res_data["cloud_url"], caption="Processed Frame: Visual Target Coordinates Mapped")
                    
                    st.json(res_data["detections"])
                else:
                    st.error("Target routing service responded with anomalous state operational error code.")
            except Exception as e:
                st.error(f"Failed to resolve pipeline handshake connection with backend API: {e}")

with tab2:
    st.header("📊 Real-time System Analytics & Historical Logs")
    
    if st.button("🔄 Sync & Refresh Analytics Dashboard", key="refresh_btn"):
        with st.spinner("Extracting transactional history from tracking engine collections..."):
            try:
                log_res = requests.get("http://127.0.0.1:8000/fetch-logs")
                if log_res.status_code == 200:
                    logs_data = log_res.json()["data"]
                    
                    if logs_data:
                        # Construct relational schema array matrix matching transaction metrics
                        df = pd.DataFrame(logs_data)
                        df['Readable Time'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%H:%M:%S')
                        
                        # Display high-level key performance metrics indicators
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total System Inferences", len(df))
                        with col2:
                            total_alerts = df['alert_triggered'].sum() if 'alert_triggered' in df.columns else 0
                            st.metric("Total Alarms Dispatched", int(total_alerts), delta=int(total_alerts), delta_color="inverse")
                        with col3:
                            avg_speed = df['inference_time_seconds'].mean()
                            st.metric("Avg Ingestion Speed", f"{avg_speed:.3f} sec")
                        
                        st.markdown("---")
                        
                        # Construct visualization charting components
                        chart_col1, chart_col2 = st.columns(2)
                        
                        with chart_col1:
                            st.subheader("📈 Threat Timeline (Threat Events Detected)")
                            chart_data1 = df.set_index('Readable Time')[['total_objects_detected']].sort_index()
                            st.line_chart(chart_data1)
                            
                        with chart_col2:
                            st.subheader("⏱️ AI Network Ingestion Speed (Latency Track)")
                            chart_data2 = df.set_index('Readable Time')[['inference_time_seconds']].sort_index()
                            st.bar_chart(chart_data2)
                            
                        st.markdown("---")
                        st.subheader("📋 Raw Database Log Streams (Latest 10 Events)")
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("Target collection path is empty. Complete classification tests first.")
            except Exception as e:
                st.error(f"Error encountered while compiling metadata metrics rows: {e}")
                
                