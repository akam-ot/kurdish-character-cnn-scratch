import streamlit as st
import cv2
import numpy as np
import pickle
import os
from PIL import Image
import matplotlib.pyplot as plt

# Import model classes
from model_classes import CNNModel
from sample_data import get_available_samples, get_random_sample, get_sample_info

# Configuration
MODEL_PATH = "kurdish_cnn_model.pkl"
IMAGE_SIZE = 32

CLASS_NAMES = [f"{i:02d}" for i in range(1, 36)]

# Kurdish character mapping from class numbers to actual characters
KURDISH_CHAR_MAP = {
    "01": "ئ", "02": "ا", "03": "ب", "04": "پ", "05": "ت", "06": "ج", "07": "چ",
    "08": "ح", "09": "خ", "10": "د", "11": "ر", "12": "ڕ", "13": "ز", "14": "ژ",
    "15": "س", "16": "ش", "17": "ع", "18": "غ", "19": "ف", "20": "ڤ", "21": "ق",
    "22": "ک", "23": "ك", "24": "گ", "25": "ل", "26": "ڵ", "27": "م", "28": "ن",
    "29": "هـ", "30": "ە", "31": "و", "32": "ۆ", "33": "وو", "34": "ی", "35": "ێ",
}


def get_display_name(class_name):
    """Convert class name to display name using actual Kurdish characters."""
    return KURDISH_CHAR_MAP.get(class_name, f"Class {class_name}")


@st.cache_resource
def load_model():
    """Load the trained CNN model."""
    try:
        model = CNNModel(num_classes=len(CLASS_NAMES))
        model.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def preprocess_image(image):
    """Preprocess uploaded image for prediction."""
    try:
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = cv2.equalizeHist(img)
        img = img.astype(np.float32) / 255.0
        img = img.reshape(1, 1, IMAGE_SIZE, IMAGE_SIZE)

        return img
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None


def predict_character(model, image):
    """Make prediction on preprocessed image."""
    try:
        probs = model.forward(image)
        predicted_class = np.argmax(probs[0])
        confidence = probs[0][predicted_class]
        return predicted_class, confidence, probs[0]
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None


def main():
    st.set_page_config(
        page_title="Kurdish Character Recognition", 
        page_icon="🔤", 
        layout="wide"
    )

    st.title("🔤 Kurdish Character Recognition")
    st.markdown("*AI-powered handwritten Kurdish character recognition*")
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Failed to load the model.")
        return

    # Main interface
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📤 Input")
        
        # Choose input method
        method = st.radio(
            "Choose your input method:",
            ["🎯 Try Sample Images", "📁 Upload Your Own"], 
            horizontal=True
        )

        image = None
        image_source = ""

        if method == "📁 Upload Your Own":
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=["png", "jpg", "jpeg", "bmp"],
                help="Upload a clear image of handwritten Kurdish character"
            )
            if uploaded_file:
                image = Image.open(uploaded_file)
                image_source = "Your uploaded image"

        else:  # Sample Images
            available_samples = get_available_samples()
            
            if available_samples:
                options = []
                char_map = {}
                
                for class_name, _, file_path in available_samples:
                    char = KURDISH_CHAR_MAP.get(class_name, class_name)
                    display = f"{char} (Class {class_name})"
                    options.append(display)
                    char_map[display] = (char, class_name, file_path)

                options = sorted(options, key=lambda x: char_map[x][1])
                
                selected = st.selectbox(
                    "Select a Kurdish character:",
                    ["-- Choose a character --"] + options
                )
                
                if selected != "-- Choose a character --":
                    char, class_name, file_path = char_map[selected]
                    image = Image.open(file_path)
                    image_source = f"Sample: {char}"
            else:
                st.warning("No sample images found.")

        # Display selected image
        if image:
            st.image(image, caption=image_source, width=300)
            
            # Analysis button
            if st.button("🔍 Analyze Character", type="primary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    processed_img = preprocess_image(image)
                    
                    if processed_img is not None:
                        predicted_class, confidence, all_probs = predict_character(model, processed_img)
                        
                        if predicted_class is not None:
                            with col2:
                                st.header("🎯 Results")
                                
                                # Main prediction
                                predicted_char = get_display_name(CLASS_NAMES[predicted_class])
                                
                                st.markdown(f"### Predicted: **{predicted_char}**")
                                
                                # Confidence with color coding
                                if confidence > 0.8:
                                    st.success(f"🎯 {confidence:.1%} confidence (Very High)")
                                elif confidence > 0.6:
                                    st.info(f"✅ {confidence:.1%} confidence (Good)")
                                elif confidence > 0.4:
                                    st.warning(f"⚠️ {confidence:.1%} confidence (Moderate)")
                                else:
                                    st.error(f"❌ {confidence:.1%} confidence (Low)")
                                
                                # Show preprocessed image
                                with st.expander("🔍 See processed image"):
                                    processed_display = (processed_img[0, 0] * 255).astype(np.uint8)
                                    st.image(processed_display, caption="How AI sees it", width=150)
                                
                                # Top 5 predictions
                                st.markdown("**Top 5 predictions:**")
                                top_indices = np.argsort(all_probs)[-5:][::-1]
                                
                                for i, idx in enumerate(top_indices):
                                    char = get_display_name(CLASS_NAMES[idx])
                                    prob = all_probs[idx]
                                    st.write(f"{i+1}. **{char}** — {prob:.1%}")

    # Balanced sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **Custom CNN Model**  
        Built from scratch for Kurdish character recognition
        
        **35 Character Classes**  
        Supports handwritten Kurdish letters
        
        **Instructions:**
        1. Choose sample or upload image
        2. Click analyze for AI prediction
        3. View results and confidence
        """)
        
        with st.expander("📋 Supported Characters"):
            # Display characters in rows
            chars = [get_display_name(name) for name in CLASS_NAMES]
            for i in range(0, len(chars), 7):
                st.write("  ".join(chars[i:i+7]))


if __name__ == "__main__":
    main()
