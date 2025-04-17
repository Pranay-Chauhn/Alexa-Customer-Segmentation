import joblib
from app.model import model

vectorizer = joblib.load("app/artifacts/preprocessing/vectorizer.pkl")
scaler = joblib.load("app/artifacts/preprocessing/scaler.pkl")


def segmentation(data):
    test_transformed = vectorizer.transform(data).toarray()
    test_scaled = scaler.transform(test_transformed)
    test_predict = model.predict(test_scaled)
    return {'prediction': int(test_predict[0])}
