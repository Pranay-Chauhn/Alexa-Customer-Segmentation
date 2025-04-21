# Alexa Sentiment Detector 

A powerful FastAPI-based NLP web app to classify Alexa product reviews into positive or negative sentiment. The application supports both **single review prediction** and **batch sentiment analysis** via file upload. Designed with performance, scalability, and user-friendliness in mind.

---

##  Project Highlights

-  **Class Imbalance Solved with SMOTE**  
  The dataset was heavily imbalanced toward positive reviews. To address this, I applied **SMOTE** (Synthetic Minority Over-sampling Technique), which significantly improved recall for Class 0 (negative sentiment).  
   **Achieved Recall (Class 0)**:
  - From **~27% (XGBoost)** → **76% (Logistic Regression + SMOTE)**

-  **Model Evaluation Focused on Recall**  
  Given the imbalance, **recall** was prioritized to reduce false negatives and capture maximum negative feedback — crucial for customer satisfaction use cases.  
   Best recall: **76.47%**, Best F1 Score: **53.79%** for Class 0 (Negative)

-  **Multiple Models Evaluated**  
  Models trained and tested:
  - Logistic Regression
  - SVM (Linear)
  - XGBoost  
  With and without SMOTE applied.

---

##  Final Evaluation Metrics

| Model                        | Accuracy | Precision (C0) | Recall (C0) | F1 Score (C0) | F1 Score (Weighted) |
|-----------------------------|----------|----------------|-------------|---------------|----------------------|
| Logistic Regression         | 92.06%   | 50.94%         | 52.94%      | 51.92%        | 92.13%               |
| SVM (Linear)                | 93.33%   | 60.98%         | 49.02%      | 54.35%        | 93.00%               |
| XGBoost                     | 93.02%   | 66.67%         | 27.45%      | 38.89%        | 91.65%               |
| **Logistic Regression + SMOTE** | **89.37%**   | 41.49%         | **76.47%**      | **53.79%**        | 90.74%               |
| SVM (Linear) + SMOTE        | 88.25%   | 36.47%         | 60.78%      | 45.59%        | 89.54%               |
| XGBoost + SMOTE             | 88.57%   | 36.36%         | 54.90%      | 43.75%        | 89.60%               |

---

##  Tech Stack

- **Backend**: FastAPI (Python 3.10.13)
- **ML Models**: Scikit-learn (Logistic Regression, SVM), XGBoost
- **Data Preprocessing**: CountVectorizer, SMOTE
- **Frontend**: HTML + CSS + Chart.js for pie graph
- **Deployment**: Uvicorn local server

---

##  Features

-  Predict sentiment for a **single review** input via text box
-  **Bulk prediction** using `.csv` upload with interactive result table
-  Live **pie chart visualization** of sentiment results with distinct colors
-  Downloadable `.csv` output file after prediction
-  Clean and mobile-friendly UI

---

##  Folder Structure

```
alexa-sentiment-detector/
│
├── app/
│   ├── main.py             # FastAPI application
│   ├── model.pkl           # Trained model
│   ├── vectorizer.pkl      # CountVectorizer
│   └── utils.py            # Preprocessing helpers
│
├── static/
│   └── styles.css
├── templates/
│   └── index.html
├── .gitignore
├── README.md
└── requirements.txt
```

---

##  How to Run Locally

```bash
# Step 1: Clone the repo
git clone https://github.com/yourusername/alexa-sentiment-detector.git
cd alexa-sentiment-detector

# Step 2: Install requirements
pip install -r requirements.txt

# Step 3: Launch FastAPI app
uvicorn app.main:app --reload

# Open in browser
http://127.0.0.1:8000
```

---

##  Sample CSV Format

```csv
review
"Alexa doesn’t understand my accent."
"This is amazing and very responsive!"
```

---

## Future Improvements

- Add deep learning model support (LSTM/Transformer)
- Deploy using Docker
- CI/CD Integration
- Add feedback collection from users

---

If you found this helpful, don't forget to star the repo and follow me!

```
