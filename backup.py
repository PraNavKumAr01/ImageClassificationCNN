from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, RedirectResponse
import io
from PIL import Image
import tensorflow as tf
import os
import requests
# import boto3

app = FastAPI()

# Create a connection to Amazon S3
# s3 = boto3.client('s3')

# Update this URL with your S3 object URL
# model_url = 'https://imagesentiment.s3.ap-south-1.amazonaws.com/sentiment.h5'
model_url = 'https://drive.google.com/uc?export=download&id=1tprxXY4S6SXsG_ZHhR9RI-JNmQCdza35'

@app.get('/', include_in_schema=False)
def index():
    return RedirectResponse("/predict/", status_code=308)

# Define a function to preprocess the image
def preprocess_image(image):
    resized_image = tf.image.resize(image, (256,256))
    scaled_image = resized_image / 255.0 
    img = tf.expand_dims(scaled_image, 0) 
    return img

# Define the prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Download the model bytes from S3
        # model_response = s3.get_object(Bucket='imagesentiment', Key=model_url)
        model_response = requests.get(model_url)
        # model_bytes = model_response['Body'].read()
        model_bytes = model_response.content

        # Load the model using TensorFlow
        model = tf.keras.models.load_model(io.BytesIO(model_bytes))

        # preproccesing the image
        image = Image.open(io.BytesIO(await file.read()))
        preprocessed_image = preprocess_image(image)

        prediction = model.predict(preprocessed_image)
        
        sentiment = "Cheer up bruda, why do you look sad?" if prediction > 0.5 else "Yayy you look happy!"
        
        return JSONResponse(content={"sentiment": sentiment})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)