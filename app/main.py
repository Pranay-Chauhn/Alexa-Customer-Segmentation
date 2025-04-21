from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import Review
from app.predict import segmentation, predict_file

app = FastAPI()

# Allow CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to ["http://localhost:5500"] etc.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/')
def home():
    return {'message': 'API is Running fine ...'}


@app.post('/predict')
async def predict(review: Review):
    data = [review.review]
    result = segmentation(data)
    return result


@app.post('/predict_file')
async def predict_f(file: UploadFile = File(...)):
    result = await predict_file(file)
    return result
