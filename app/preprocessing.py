import joblib
import re
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords

stemmer = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))

# Loading Artifacts
vectorizer = joblib.load("app/artifacts/preprocessing/vectorizer.pkl")
scaler = joblib.load("app/artifacts/preprocessing/scaler.pkl")


def preprocess_text(text):
    review = re.sub('[^a-zA-Z]', '', text)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
    return ' '.join(review)
