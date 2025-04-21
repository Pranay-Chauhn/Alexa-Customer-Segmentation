from fastapi import FastAPI, File, UploadFile

from app.schemas import Review
from app.predict import segmentation, predict_file

app = FastAPI()


@app.get('/')
def home():
    return {'message': 'API is Running ...'}


@app.post('/predict')
async def predict(review: Review):
    data = [review.review]
    result = segmentation(data)
    return result


@app.post('/predict_file')
async def predict_f(file: UploadFile = File(...)):
    result = await predict_file(file)
    return result
