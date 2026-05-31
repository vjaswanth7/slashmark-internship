# Sentiment Analysis Project

A 3-class sentiment analysis model for **positive / negative / neutral** text classification.

## What it includes
- Text preprocessing: cleanup, lowercase, stopword removal, lemmatization/stemming fallback
- Feature extraction with **TF-IDF**
- Classical ML models: **Logistic Regression** and **Linear SVM**
- Evaluation: **accuracy, precision, recall, F1**, confusion matrix
- Model export with **joblib**

## Files
- `sentiment_analysis_model.py` — training and prediction script
- `demo_sentiment_data.csv` — small demo dataset with 3 labels
- `sentiment_model.joblib` — saved model after training

## Run
```bash
pip install pandas scikit-learn joblib nltk
python sentiment_analysis_model.py --data demo_sentiment_data.csv --text "The food was amazing"
```

## To use your own data
Create a CSV/TSV file with columns:
- `text`
- `label`

Allowed labels:
- `positive`
- `negative`
- `neutral`

## Notes
- The demo dataset is for quick testing.
- For a stronger academic project, replace the demo CSV with a larger labeled dataset from Twitter, IMDb-style reviews, product reviews, or restaurant reviews.
