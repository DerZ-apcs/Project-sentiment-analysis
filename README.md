# Sentiment Analysis - IMDB Movie Reviews

This project builds a sentiment classifier (Positive/Negative) for movie reviews. The pipeline includes text preprocessing, feature extraction (BoW/TF-IDF), and model training with Logistic Regression and Multinomial Naive Bayes. A Streamlit web UI is included for demo.

## Main components

- Training notebook: Sentiment_Analysis.ipynb
- Web UI: app.py (Streamlit)
- Data: data/Dataset.csv, data/cleaned_imdb_data.csv
- Models and vectorizers: pkl/*.pkl

## Requirements

- Python 3.9+ (recommended 3.10)
- Packages listed in requirements.txt

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train models

Open the notebook and run from top to bottom to create files in pkl/.

```bash
jupyter notebook Sentiment_Analysis.ipynb
```

If you want a quick checklist, run these cells in order:

1. Import libraries
2. Load data from data/Dataset.csv
3. Preprocess and save cleaned data
4. Feature extraction (BoW/TF-IDF)
5. Train models and save .pkl files

After training, pkl/ should contain:

- lr_bow_model.pkl
- bow_vectorizer.pkl
- lr_tfidf_model.pkl
- tfidf_vectorizer.pkl
- nb_bow_model.pkl
- nb_tfidf_model.pkl

## Run the web app (Streamlit)

```bash
streamlit run app.py
```

Open your browser at http://localhost:8501

## Deploy to the web (Streamlit Community Cloud)

1. Push the code to a GitHub repo.
2. Go to https://share.streamlit.io (or Streamlit Community Cloud).
3. Select the repo and set the entry file to app.py.
4. Streamlit will install from requirements.txt and build automatically.

If you want to use existing data/models, make sure pkl/ is included in the repo. Otherwise, train the models first and then deploy.

## Project structure

```
.
├── app.py
├── README.md
├── requirements.txt
├── Sentiment_Analysis.ipynb
├── data/
│   ├── cleaned_imdb_data.csv
│   └── Dataset.csv
└── pkl/
```

## Notes

- app.py requires the model files in pkl/ for inference.
- If models are missing, run the notebook to generate them.

