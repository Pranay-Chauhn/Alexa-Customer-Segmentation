from fastapi import FastAPI
from app.predict import segmentation

app = FastAPI()


@app.get('/')
def home():
    return {'message': 'API is Running ...'}


@app.post('/predict')
def predict(review: Review):
    data = review.to_list()
    result = segmentation(data)[0]
    return result
