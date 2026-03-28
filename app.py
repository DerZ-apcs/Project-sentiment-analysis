import os
import joblib
import streamlit as st
import pandas as pd
from typing import Tuple, Optional, Any

class Predictor:
    def __init__(self, name, model, vectorizer):
        self.name = name
        self.model = model
        self.vectorizer = vectorizer

    def predict(self, text: str) -> Tuple[str, Optional[float]]:
        processed_text = [text]
        vectorized_text = self.vectorizer.transform(processed_text)
        pred = self.model.predict(vectorized_text)[0]
        label = "Positive" if pred == 1 else "Negative"
        confidence = None
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(vectorized_text)[0]
            confidence = float(max(probs))
        return label, confidence

@st.cache_resource 
def load_models():
    registry = {}
    paths = {
        "Logistic Regression + BoW": ("pkl/lr_bow_model.pkl", "pkl/bow_vectorizer.pkl"),
        "Logistic Regression + TF-IDF": ("pkl/lr_tfidf_model.pkl", "pkl/tfidf_vectorizer.pkl"),
        "Naive Bayes + BoW": ("pkl/nb_bow_model.pkl", "pkl/bow_vectorizer.pkl"),
        "Naive Bayes + TF-IDF": ("pkl/nb_tfidf_model.pkl", "pkl/tfidf_vectorizer.pkl"),
    }
    for name, (m_path, v_path) in paths.items():
        if os.path.exists(m_path) and os.path.exists(v_path):
            # Dùng joblib.load và truyền trực tiếp đường dẫn
            model = joblib.load(m_path)
            vec = joblib.load(v_path)
            registry[name] = Predictor(name, model, vec)
        else:
            # Thêm cảnh báo nhỏ này ra terminal (Colab) để bạn dễ kiểm tra nếu lỡ gõ sai tên thư mục
            print(f"Cảnh báo: Không tìm thấy file cho {name} tại {m_path} hoặc {v_path}")
            
    return registry

def main():
    st.title("🎬 Movie Review Sentiment Analysis")
    st.markdown("Input a movie review below to see if it is Positive or Negative.")
    registry = load_models()
    if not registry:
        st.error("No models found. Please run the training cells first.")
        return
    model_choice = st.selectbox("Select Model Configuration", list(registry.keys()))
    predictor = registry[model_choice]
    user_input = st.text_area("Review Text:", "I absolutely loved this movie!")
    if st.button("Analyze Sentiment"):
        if user_input.strip():
            label, conf = predictor.predict(user_input)
            color = "green" if label == "Positive" else "red"
            st.markdown(f"### Result: :{color}[{label}]")
            if conf: st.write(f"**Confidence:** {conf:.2%}")
        else: st.warning("Please enter some text.")

if __name__ == '__main__':
    main()
