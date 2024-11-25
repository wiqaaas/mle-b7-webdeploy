import streamlit as st
import pandas as pd
from PIL import Image
from dummy_model import tabular_predict, text_predict, image_predict

st.title("Interactive ML App with Tabular, Text, and Image Inputs")

# Sidebar Navigation
option = st.sidebar.selectbox(
    "Choose an input type", 
    ["Tabular Data", "Text Data", "Image Data"]
)

# Tabular Input Section
if option == "Tabular Data":
    st.header("Tabular Data Input")
    st.write("Enter numeric inputs for the tabular prediction model.")

    # Accepting inputs for a simple tabular model
    num_features = st.number_input("Number of Features", min_value=1, max_value=10, value=3)
    input_data = []
    for i in range(num_features):
        value = st.number_input(f"Feature {i+1}", value=0.0)
        input_data.append(value)
    
    if st.button("Predict Tabular Data"):
        prediction = tabular_predict(input_data)
        st.write(f"Predicted Value: {prediction:.2f}")

# Text Input Section
elif option == "Text Data":
    st.header("Text Data Input")
    st.write("Enter a text snippet for sentiment analysis.")

    input_text = st.text_area("Enter Text", "")
    
    if st.button("Predict Text Sentiment"):
        sentiment = text_predict(input_text)
        st.write(f"Predicted Sentiment: {sentiment}")

# Image Input Section
elif option == "Image Data":
    st.header("Image Data Input")
    st.write("Upload an image for classification.")

    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Classify Image"):
            prediction = image_predict(image)
            st.write(f"Predicted Class: {prediction}")
