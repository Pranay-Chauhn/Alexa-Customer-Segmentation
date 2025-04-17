from fastapi import FastAPI
from app.schemas import Review
from app.predict import segmentation

app = FastAPI()


@app.get('/')
def home():
    return {'message': 'API is Running ...'}


@app.post('/predict')
def predict(review: Review):
    data = [review.review]
    result = segmentation(data)
    return result
