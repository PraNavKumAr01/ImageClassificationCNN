from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware import Middleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import io
from PIL import Image
import tensorflow as tf
import os

def convert_get_to_post_middleware(request: Request, call_next):
    if request.method == "GET" and request.url.path == "/predict/":
        request.method = "POST"
    response = call_next(request)
    return response

middleware = [
    Middleware(TrustedHostMiddleware),
    Middleware(convert_get_to_post_middleware)
]

app = FastAPI(middleware=middleware)

@app.get('/', include_in_schema=False)
def index():
    return RedirectResponse("/predict/", status_code=308)

# Load the pre-trained model
model = tf.keras.models.load_model(os.path.join('models', 'sentiment.h5'))

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
        image = Image.open(io.BytesIO(await file.read()))
        preprocessed_image = preprocess_image(image)

        prediction = model.predict(preprocessed_image)
        
        sentiment = "Cheer up bruda, why do you look sad?" if prediction > 0.5 else "Yayy you look happy!"
        
        return JSONResponse(content={"sentiment": sentiment})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)