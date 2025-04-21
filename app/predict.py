import joblib
import pandas as pd
from fastapi import HTTPException
from app.preprocessing import vectorizer, scaler, preprocess_text
from app.model import model
from io import StringIO


def segmentation(data):
    test_transformed = vectorizer.transform(data).toarray()
    test_scaled = scaler.transform(test_transformed)
    test_predict = model.predict(test_scaled)
    return {'prediction': int(test_predict[0])}


async def predict_file(file):
    contents = await file.read()
    string_data = contents.decode('utf-8')
    df = pd.read_csv(StringIO(string_data))
    print("DataFrame head:", df.head())
    print("Column names:", df.columns.tolist())

    if "review" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain a 'review' column.")

    # Preprocessing data
    corpus = [preprocess_text(text) for text in df['review'].astype(str)]
    if not corpus:
        raise HTTPException(status_code=400, detail="No valid reviews found.")
    print("Corpus length:", len(corpus))

    # Vectorizer and Scaling
    x = vectorizer.transform(corpus).toarray()
    x_scaled = scaler.transform(x)

    # Predict
    predictions = model.predict(x_scaled)
    df['prediction'] = predictions
    return {'predictions': df[['review', 'prediction']].to_dict(orient='records')}
