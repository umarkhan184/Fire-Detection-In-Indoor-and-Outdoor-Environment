import streamlit as st
from yolov8_inference import YOLOv8Inference
import os

# Set page config for better appearance
st.set_page_config(
    page_title="Fire Detection for Indoor & Outdoor Environments",
    page_icon="🔥",
    layout="centered"
)

# Sidebar with project info and fire color
st.sidebar.markdown(
    """
    <div style="background-color:#b71c1c;padding:20px;border-radius:10px">
        <h2 style="color:#fff;text-align:center;">🔥 Fire Detection System</h2>
        <p style="color:#fff;text-align:center;">
            Detect fire in indoor and outdoor environments using AI-powered object detection.<br>
            Upload an image or use your webcam for instant analysis.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Main header with colored background and black font
st.markdown(
    """
    <div style="
        background-color:#ffd54f;
        padding: 36px 10px 36px 10px;
        border-radius: 16px;
        margin-bottom: 30px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.18);
    ">
        <h1 style='text-align: center; color: #111; text-shadow: 2px 2px 8px #fff;'>🔥 Fire Detection for Indoor & Outdoor Environments</h1>
        <h4 style='text-align: center; color: #222;'>AI-Powered Fire Hazard Detection System</h4>
        <p style="text-align:center; color:#333; margin-top:12px; font-size:18px;">
            <em>Upload a photo or use your webcam to detect fire hazards in real-time.<br>
            Enhance safety for homes, offices, and outdoor spaces.</em>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload an indoor or outdoor image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    upload_folder = "static/uploads"
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Run inference
    model_path = r"C:\Users\hp\Desktop\project_fire\Fire_detection_system\best (1).pt"  # Update model path as needed
    output_path = os.path.join(upload_folder, "result_" + uploaded_file.name)
    yolo = YOLOv8Inference(model_path)
    results = yolo.infer(file_path)
    yolo.save_results(results, output_path)

    st.success("Fire detection complete! See the result below.")
    st.image(output_path, caption="Detected Fire Regions", use_container_width=True)
    # Download button
    with open(output_path, "rb") as img_file:
        st.download_button(
            label="Download Result Image",
            data=img_file,
            file_name="detected_" + uploaded_file.name,
            mime="image/jpeg",
            key="download_pred"
        )

st.markdown("---")