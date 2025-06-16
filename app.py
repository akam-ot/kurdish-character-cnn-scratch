import streamlit as st
import cv2
import numpy as np
import pickle
import os
from PIL import Image
import matplotlib.pyplot as plt

# Import model classes
from model_classes import CNNModel

# Configuration
MODEL_PATH = "kurdish_cnn_model.pkl"
IMAGE_SIZE = 32

# Class names exactly as they appear in the dataset (01, 02, ..., 35)
CLASS_NAMES = [f"{i:02d}" for i in range(1, 36)]

KURDISH_CHAR_MAP = {
"01": "ئ", "02": "ا", "03": "ب", "04": "پ", "05": "ت",
"06": "ج", "07": "چ", "08": "ح", "09": "خ", "10": "د",
"11": "ر", "12": "ڕ", "13": "ز", "14": "ژ", "15": "س",
"16": "ش", "17": "ع", "18": "غ", "19": "ف", "20": "ڤ",
"21": "ق", "22": "ک", "23": "گ", "24": "ل", "25": "ڵ",
"26": "م", "27": "ن", "28": "و", "29": "ۆ", "30": "ه",
"31": "هـ", "32": "ی", "33": "ێ", "34": "ە", "35": "ى"
}


def get_display_name(class_name):
    """Convert class name to display name. Customize this if you know the character mappings."""
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
    """Preprocess uploaded image for prediction - EXACTLY matching training preprocessing."""
    try:
        # Convert PIL image to numpy array
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image

        # Convert to grayscale if needed (training uses cv2.IMREAD_GRAYSCALE)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 1. Resize to IMAGE_SIZE x IMAGE_SIZE
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        # 2. Apply histogram equalization
        img = cv2.equalizeHist(img)

        # 3. Convert to float32 and normalize to [0,1]
        img = img.astype(np.float32) / 255.0

        # 4. Reshape to match training format: (1, IMAGE_SIZE, IMAGE_SIZE)
        img = img.reshape(1, IMAGE_SIZE, IMAGE_SIZE)

        # 5. Add batch dimension for model input: (1, 1, IMAGE_SIZE, IMAGE_SIZE)
        img = img.reshape(1, 1, IMAGE_SIZE, IMAGE_SIZE)

        return img
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None


def predict_character(model, image):
    """Make prediction on preprocessed image."""
    try:
        # Get prediction probabilities
        probs = model.forward(image)

        # Get predicted class
        predicted_class = np.argmax(probs[0])
        confidence = probs[0][predicted_class]

        return predicted_class, confidence, probs[0]
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None


def main():
    st.set_page_config(
        page_title="Kurdish Character Recognition", page_icon="🔤", layout="wide"
    )

    st.title("🔤 Kurdish Handwritten Character Recognition")
    st.markdown(
        "Upload an image of a handwritten Kurdish character and get AI predictions!"
    )

    # Load model
    model = load_model()
    if model is None:
        st.error("Failed to load the model. Please check if the model file exists.")
        return

    # Create columns for layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📤 Upload Image")

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            help="Upload a clear image of a handwritten Kurdish character",
        )

        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Preprocess button
            if st.button("🔍 Analyze Character", type="primary"):
                with st.spinner("Processing image..."):
                    # Preprocess image
                    processed_img = preprocess_image(image)

                    if processed_img is not None:
                        # Make prediction
                        predicted_class, confidence, all_probs = predict_character(
                            model, processed_img
                        )

                        if predicted_class is not None:
                            with col2:
                                st.header("🎯 Prediction Results")

                                # Main prediction
                                st.success(
                                    f"**Predicted Character: {get_display_name(CLASS_NAMES[predicted_class])}**"
                                )
                                st.info(f"**Confidence: {confidence:.2%}**")

                                # Show preprocessed image
                                st.subheader("Preprocessed Image")
                                processed_display = (processed_img[0, 0] * 255).astype(
                                    np.uint8
                                )
                                st.image(
                                    processed_display,
                                    caption="Processed for AI",
                                    width=200,
                                )

                                # Top 5 predictions
                                st.subheader("Top 5 Predictions")
                                top_5_indices = np.argsort(all_probs)[-5:][::-1]

                                for i, idx in enumerate(top_5_indices):
                                    char = get_display_name(CLASS_NAMES[idx])
                                    prob = all_probs[idx]

                                    # Create progress bar for visualization
                                    st.write(f"{i+1}. **{char}** - {prob:.2%}")
                                    st.progress(prob)

                                # Confidence interpretation
                                st.subheader("📊 Confidence Guide")
                                if confidence > 0.8:
                                    st.success(
                                        "🎯 Very High Confidence - Excellent prediction!"
                                    )
                                elif confidence > 0.6:
                                    st.info("✅ Good Confidence - Reliable prediction")
                                elif confidence > 0.4:
                                    st.warning(
                                        "⚠️ Moderate Confidence - Consider image quality"
                                    )
                                else:
                                    st.error("❌ Low Confidence - Try a clearer image")

    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About This App")
        st.markdown(
            """
        This app uses a **Convolutional Neural Network (CNN) built from scratch** 
        to recognize handwritten Kurdish characters.
        
        ### 📋 Instructions:
        1. Upload a clear image of a handwritten Kurdish character
        2. Click "Analyze Character" to get predictions
        3. View the results and confidence scores
        
        ### 🎯 Model Info:
        - **Architecture**: Custom CNN from scratch
        - **Classes**: 35 Kurdish characters
        - **Framework**: Pure NumPy implementation
        - **Training**: Kurdish Handwritten Character Database
        
        ### 💡 Tips for Best Results:
        - Use clear, well-lit images
        - Ensure the character fills most of the image
        - Avoid blurry or distorted images
        - Black ink on white/light background works best
        """
        )

        st.header("🔤 Supported Characters")
        st.write("The model can recognize these 35 Kurdish character classes:")

        # Display characters in a grid
        chars_per_row = 7
        for i in range(0, len(CLASS_NAMES), chars_per_row):
            row_chars = [
                get_display_name(name) for name in CLASS_NAMES[i : i + chars_per_row]
            ]
            st.write(" | ".join(row_chars))


if __name__ == "__main__":
    main()
