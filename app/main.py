from fastapi import FastAPI, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from io import BytesIO

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Load model
model = load_model("models/best_model.h5")
categories = ["No Tumor", "Pituitary Tumor", "Meningioma Tumor", "Glioma Tumor"]


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    # Read image file
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB").resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    predictions = model.predict(image_array)
    pred_class = categories[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    result = f"Prediction: {pred_class} ({confidence:.2f}% confidence)"
    return templates.TemplateResponse(
        "result.html", {"request": request, "result": result}
    )


# If running locally, uncomment the following lines
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
