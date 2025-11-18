import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Dental Caries Detection",
    page_icon="🦷",
    layout="wide"
)

# --- Helper Function: Get Color by Confidence ---
# OpenCV uses BGR format (Blue, Green, Red)
def get_box_color(confidence):
    if confidence >= 0.80:
        return (0, 255, 0)      # Green
    elif confidence >= 0.60:
        return (0, 255, 255)    # Yellow
    elif confidence >= 0.40:
        return (0, 140, 255)    # Orange
    else:
        return (0, 0, 255)      # Red

# --- Model Loading ---
# Load the trained YOLOv8 model
# Use @st.cache_resource to load the model only once
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # Assumes 'best.pt' is in the same directory
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Please make sure 'best.pt' is in the folder.")
    st.stop()

# --- Application Title ---
st.title("Dental Caries (Cavity) Detection")
st.write("Upload a dental X-ray. The detection boxes will change color based on the model's confidence level.")

# --- Sidebar Settings ---
st.sidebar.header("Settings")
# Slider to filter detections. Default is 25%.
conf_threshold_percent = st.sidebar.slider("Confidence Threshold (%)", 0, 100, 25)
conf_threshold = conf_threshold_percent / 100.0

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Read the uploaded image
    image_data = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_data))
    
    # 2. Convert PIL Image to a NumPy array (RGB)
    img_array = np.array(image)
    
    # 3. Convert RGB to BGR for OpenCV processing
    # If the image is grayscale, convert to BGR so we can draw colored boxes
    if len(img_array.shape) == 2: 
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    elif img_array.shape[2] == 4: # RGBA
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    else:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    st.subheader("Analyzing Image...")

    # 4. Run the model on the image
    # We set a low internal threshold (0.1) to get most boxes, then we filter manually below
    results = model(img_bgr, conf=0.1)

    # 5. Manual Plotting Logic
    # Create a copy of the image to draw annotations on
    annotated_img = img_bgr.copy()
    detections_count = 0
    
    # Loop through all detected boxes
    for box in results[0].boxes:
        # Get confidence score (float)
        score = box.conf.item()
        
        # Check if the score meets the user's slider threshold
        if score >= conf_threshold:
            detections_count += 1
            
            # Get box coordinates (x1, y1, x2, y2) as integers
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Determine color based on confidence score
            color = get_box_color(score)
            
            # Draw the rectangle
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text (e.g., "Caries 85%")
            label_text = f"Caries {score:.0%}"
            
            # Calculate text size for the background box
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Draw filled rectangle for text background (for readability)
            cv2.rectangle(annotated_img, (x1, y1 - 20), (x1 + w, y1), color, -1)
            
            # Draw the text (white color)
            cv2.putText(annotated_img, label_text, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 6. Display the images
    # Create columns for side-by-side display
    col1, col2, _ = st.columns([3, 3, 2]) 
    
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)
    with col2:
        # Convert BGR (OpenCV) back to RGB (Streamlit format)
        result_image_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        st.image(result_image_rgb, caption=f"Detection Results (Threshold: {conf_threshold_percent}%)", use_column_width=True)
        
    # 7. Display detection details
    st.subheader("Detection Details:")
    if detections_count == 0:
        st.write(f"No caries detected above {conf_threshold_percent}% confidence.")
    else:
        st.success(f"Found {detections_count} potential areas.")
        
        # Color Legend
        st.markdown("### Confidence Color Legend")
        st.markdown(
            """
            * <span style='color:green'>**Green:**</span> 80% - 100% (High Confidence)
            * <span style='color:#CCCC00'>**Yellow:**</span> 60% - 80%
            * <span style='color:orange'>**Orange:**</span> 40% - 60%
            * <span style='color:red'>**Red:**</span> Below 40% (Low Confidence)
            """, 
            unsafe_allow_html=True
        )
