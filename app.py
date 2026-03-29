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


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-1: #0f172a;
            --bg-2: #1e1b4b;
            --card: rgba(255, 255, 255, 0.08);
            --card-border: rgba(255, 255, 255, 0.16);
            --accent: #f59e0b;
            --accent-2: #22d3ee;
            --text: #f8fafc;
            --muted: #cbd5f5;
        }

        html, body, [data-testid="stAppViewContainer"] {
            background: radial-gradient(1200px 600px at 20% -10%, #3b0764 0%, transparent 55%),
                        radial-gradient(1000px 800px at 110% 10%, #0ea5e9 0%, transparent 50%),
                        linear-gradient(160deg, var(--bg-1) 0%, var(--bg-2) 60%);
            color: var(--text);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        .app-hero {
            padding: 22px 24px;
            border-radius: 18px;
            background: linear-gradient(140deg, rgba(15, 23, 42, 0.6), rgba(30, 27, 75, 0.8));
            border: 1px solid var(--card-border);
            box-shadow: 0 20px 60px rgba(15, 23, 42, 0.45);
            position: relative;
            overflow: hidden;
        }

        .app-hero::after {
            content: "";
            position: absolute;
            inset: -80px 30% auto auto;
            width: 220px;
            height: 220px;
            background: radial-gradient(circle, rgba(245, 158, 11, 0.35), transparent 65%);
            filter: blur(10px);
            animation: floatGlow 6s ease-in-out infinite;
        }

        .hero-title {
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: 0.5px;
            margin: 0 0 6px 0;
        }

        .hero-subtitle {
            color: var(--muted);
            font-size: 1rem;
            margin: 0;
        }

        .card {
            background: var(--card);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 16px 18px;
            box-shadow: 0 16px 35px rgba(15, 23, 42, 0.35);
        }

        .result-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 14px;
            border-radius: 999px;
            font-weight: 600;
            letter-spacing: 0.3px;
            background: rgba(34, 211, 238, 0.12);
            border: 1px solid rgba(34, 211, 238, 0.4);
        }

        .result-pill.negative {
            background: rgba(248, 113, 113, 0.15);
            border-color: rgba(248, 113, 113, 0.5);
        }

        .stButton > button {
            background: linear-gradient(120deg, var(--accent), #fb7185);
            color: #0f172a;
            border: none;
            border-radius: 999px;
            padding: 0.7rem 1.6rem;
            font-weight: 700;
            box-shadow: 0 12px 24px rgba(245, 158, 11, 0.3);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .stButton > button:hover {
            transform: translateY(-2px) scale(1.01);
            box-shadow: 0 16px 30px rgba(245, 158, 11, 0.45);
        }

        .stTextArea textarea {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(148, 163, 184, 0.4);
            color: var(--text);
            border-radius: 12px;
        }

        .stSelectbox [data-baseweb="select"] > div {
            background: rgba(15, 23, 42, 0.7);
            border-radius: 10px;
            border: 1px solid rgba(148, 163, 184, 0.4);
            color: var(--text);
        }

        @keyframes floatGlow {
            0%, 100% { transform: translateY(0px); opacity: 0.9; }
            50% { transform: translateY(18px); opacity: 0.6; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def main():
    st.set_page_config(
        page_title="Movie Review Sentiment",
        page_icon="🎬",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    inject_theme()

    st.markdown(
        """
        <section class="app-hero">
            <h1 class="hero-title">Movie Review Sentiment Analysis</h1>
            <p class="hero-subtitle">
                AIO 2026 • Group CONQ011 • Predict Positive/Negative in seconds.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card">
            <strong>How it works</strong><br/>
            Paste a review, then compare predictions from four model setups.
        </div>
        """,
        unsafe_allow_html=True,
    )
    registry = load_models()
    if not registry:
        st.error("No models found. Please run the training cells first.")
        return
    model_names = list(registry.keys())
    predictors = [registry[name] for name in model_names]
    user_input = st.text_area(
        "Review Text:",
        "I absolutely loved this movie!",
        height=150,
        placeholder="Type or paste a movie review here...",
    )

    action = st.button("Analyze Sentiment", use_container_width=True)

    if action:
        if user_input.strip():
            results = {}
            for predictor in predictors:
                results[predictor.name] = predictor.predict(user_input)
            st.session_state["last_results"] = results
        else:
            st.warning("Please enter some text.")

    if "last_results" in st.session_state:
        results = st.session_state["last_results"]
        cols = st.columns(2)
        for idx, name in enumerate(model_names):
            label, conf = results.get(name, ("N/A", None))
            pill_class = "result-pill" if label == "Positive" else "result-pill negative"
            emoji = "✅" if label == "Positive" else "⚠️"
            conf_text = f"{conf:.2%}" if conf is not None else "N/A"
            with cols[idx % 2]:
                st.markdown(
                    f"""
                    <div class="card">
                        <div style="margin-bottom: 8px; font-weight: 600;">{name}</div>
                        <div class="{pill_class}">{emoji} Result: {label}</div>
                        <div style="margin-top: 10px; color: var(--muted);">
                            Confidence: {conf_text}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

if __name__ == '__main__':
    main()
