import streamlit as st
import requests

st.title("Sentiment Analysis with Streamlit")

# Define your FastAPI model endpoint
MODEL_ENDPOINT = "http://0.0.0.0:8000/predict/"

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg","bmp"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")

    # Perform inference when the user clicks a button
    if st.button("Predict"):
        try:
            # Send image to FastAPI model for inference
            response = requests.post(MODEL_ENDPOINT, files={"file": uploaded_file})
            sentiment = response.json()["sentiment"]

            # Display sentiment analysis result
            st.success(sentiment)
        except Exception as e:
            st.error(f"An error occurred: {e}")
