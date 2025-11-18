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



# --- Model Loading ---

# Load the trained YOLOv8 model

# Use @st.cache_resource to load the model only once

@st.cache_resource

def load_model():

    model = YOLO("best.pt")  # Assumes 'best.pt' is in the same directory

    return model



model = load_model()



# --- Application Title ---

st.title("Dental Caries (Cavity) Detection")

st.write("Upload a dental X-ray, and the YOLOv8 model will try to detect caries.")



# --- File Uploader ---

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])



if uploaded_file is not None:

    # 1. Read the uploaded image

    image_data = uploaded_file.getvalue()

    image = Image.open(io.BytesIO(image_data))

    

    # 2. Convert PIL Image to an OpenCV format (NumPy array)

    img_cv = np.array(image)

    

    # 3. Ensure image is BGR (OpenCV format)

    if len(img_cv.shape) == 2: # Grayscale

        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)

    elif img_cv.shape[2] == 4: # RGBA

        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2BGR)



    st.subheader("Analyzing Image...")



    # 4. Run the model on the image

    results = model(img_cv)



    # 5. Plot the results on the image

    # 'results[0].plot()' returns a NumPy array (BGR) with bounding boxes

    result_image_plotted = results[0].plot()



    # 6. Display the images (occupying 60% of the screen)

    # We define 3 columns with relative widths [30%, 30%, 40%]

    # The 3rd column (_) is a placeholder for the empty space.

    col1, col2, _ = st.columns([3, 3, 4]) 

    

    with col1:

        st.image(image, caption="Original Image", use_column_width=True)

    with col2:

        # Convert BGR (from OpenCV) to RGB (for Streamlit)

        result_image_rgb = cv2.cvtColor(result_image_plotted, cv2.COLOR_BGR2RGB)

        st.image(result_image_rgb, caption="Detection Results", use_column_width=True)

        

    # 7. (Optional) Display detection details

    st.subheader("Detection Details:")

    if len(results[0].boxes) == 0:

        st.write("No caries detected in this image.")

    else:

        for box in results[0].boxes:

            conf = box.conf[0] * 100

            st.write(f"- Detected object with {conf:.2f}% confidence.")
