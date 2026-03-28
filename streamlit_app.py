import os
import pickle
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def artifact_path(*parts: str) -> str:
    return os.path.join(BASE_DIR, *parts)


@dataclass
class Predictor:
    name: str
    model: Any
    vectorizer: Optional[Any] = None

    def predict(self, text: str) -> Tuple[str, Optional[float]]:
        model_input = [text]

        if self.vectorizer is not None:
            model_input = self.vectorizer.transform(model_input)

        pred = self.model.predict(model_input)[0]
        label = normalize_label(pred)

        confidence = None
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(model_input)[0]
            confidence = float(max(probs))

        return label, confidence


def normalize_label(value: Any) -> str:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"positive", "pos", "1", "true"}:
            return "Positive"
        if lowered in {"negative", "neg", "0", "false"}:
            return "Negative"

    if isinstance(value, (int, float)):
        return "Positive" if int(value) == 1 else "Negative"

    return "Positive" if str(value).lower() in {"positive", "pos"} else "Negative"


def load_pickle_if_exists(paths: List[str]) -> Optional[Any]:
    for path in paths:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    return None


def build_registry_from_pairwise_dict(data: Dict[str, Any]) -> Dict[str, Predictor]:
    registry: Dict[str, Predictor] = {}
    for key, value in data.items():
        if isinstance(value, tuple) and len(value) == 2:
            model, vectorizer = value
            registry[key] = Predictor(name=key, model=model, vectorizer=vectorizer)
        elif hasattr(value, "predict"):
            registry[key] = Predictor(name=key, model=value, vectorizer=None)
    return registry


def build_registry(models_obj: Any, vectorizers_obj: Any) -> Dict[str, Predictor]:
    registry: Dict[str, Predictor] = {}

    if isinstance(models_obj, dict):
        direct = build_registry_from_pairwise_dict(models_obj)
        if direct:
            return direct

    if hasattr(models_obj, "predict"):
        registry["Pipeline model from model.pkl"] = Predictor(
            name="Pipeline model from model.pkl", model=models_obj, vectorizer=None
        )
        return registry

    if not isinstance(models_obj, dict):
        return registry

    if not isinstance(vectorizers_obj, dict):
        return registry

    model_alias = {
        "Logistic Regression": ["logistic", "lr", "logreg", "logistic_regression"],
        "Multinomial NB": ["nb", "mnb", "naive_bayes", "multinomial_nb"],
    }

    vectorizer_alias = {
        "TF-IDF": ["tfidf", "tf_idf", "tf-idf"],
        "Bag of Word": ["bow", "bag_of_words", "count", "count_vectorizer"],
    }

    def pick_key(container: Dict[str, Any], aliases: List[str]) -> Optional[str]:
        lowered_map = {str(k).lower(): k for k in container.keys()}
        for alias in aliases:
            if alias in lowered_map:
                return lowered_map[alias]
        return None

    for model_name, model_keys in model_alias.items():
        model_key = pick_key(models_obj, model_keys)
        if model_key is None:
            continue

        for vec_name, vec_keys in vectorizer_alias.items():
            vec_key = pick_key(vectorizers_obj, vec_keys)
            if vec_key is None:
                continue

            display_name = f"{model_name} + {vec_name}"
            registry[display_name] = Predictor(
                name=display_name,
                model=models_obj[model_key],
                vectorizer=vectorizers_obj[vec_key],
            )

    return registry


def get_predictor_registry() -> Dict[str, Predictor]:
    registry: Dict[str, Predictor] = {}

    # Load from pkl/ directory with specific naming
    model_pairs = [
        (
            artifact_path("pkl", "lr_bow_model.pkl"),
            artifact_path("pkl", "bow_vectorizer.pkl"),
            "Logistic Regression + Bag of Words",
        ),
        (
            artifact_path("pkl", "lr_tfidf_model.pkl"),
            artifact_path("pkl", "tfidf_vectorizer.pkl"),
            "Logistic Regression + TF-IDF",
        ),
        (
            artifact_path("pkl", "nb_bow_model.pkl"),
            artifact_path("pkl", "bow_vectorizer.pkl"),
            "Multinomial NB + Bag of Words",
        ),
        (
            artifact_path("pkl", "nb_tfidf_model.pkl"),
            artifact_path("pkl", "tfidf_vectorizer.pkl"),
            "Multinomial NB + TF-IDF",
        ),
    ]

    for model_path, vec_path, display_name in model_pairs:
        if os.path.exists(model_path) and os.path.exists(vec_path):
            try:
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                with open(vec_path, "rb") as f:
                    vectorizer = pickle.load(f)
                registry[display_name] = Predictor(
                    name=display_name, model=model, vectorizer=vectorizer
                )
            except Exception as e:
                st.warning(f"Failed to load {display_name}: {e}")

    # Fallback to legacy single file loading
    if not registry:
        model_candidates = [
            artifact_path("model.pkl"),
            artifact_path("models", "model.pkl"),
            artifact_path("artifacts", "model.pkl"),
        ]
        vectorizer_candidates = [
            artifact_path("vectorize.pkl"),
            artifact_path("vectorizer.pkl"),
            artifact_path("models", "vectorize.pkl"),
            artifact_path("models", "vectorizer.pkl"),
            artifact_path("artifacts", "vectorize.pkl"),
            artifact_path("artifacts", "vectorizer.pkl"),
        ]

        models_obj = load_pickle_if_exists(model_candidates)
        vectorizers_obj = load_pickle_if_exists(vectorizer_candidates)

        if models_obj is not None:
            registry = build_registry(models_obj, vectorizers_obj)

    return registry


def mock_predict() -> str:
    return random.choice(["Positive", "Negative"])


def render_app() -> None:
    st.set_page_config(page_title="Sentiment Demo UI", page_icon="💬", layout="centered")

    st.title("Sentiment Demo UI")
    st.caption("Mock predict nhanh hoặc test model Logistic Regression/Naive Bayes từ thư mục pkl")

    with st.sidebar:
        st.header("Mode")
        mode = st.radio(
            "Choose prediction mode",
            options=["Mock predict (random)", "Use pkl models (LogReg/NB + BoW/TF-IDF)"],
        )

    registry = get_predictor_registry()

    selected_predictor = None
    if mode == "Use pkl models (LogReg/NB + BoW/TF-IDF)":
        if not registry:
            st.warning(
                "Khong tim thay file trong pkl/. App se fallback ve mock predict."
            )
            mode = "Mock predict (random)"
        else:
            selected_name = st.selectbox("Model setup", list(registry.keys()))
            selected_predictor = registry[selected_name]
            st.success(f"Loaded: {selected_name}")

    text = st.text_area(
        "Input text for sentiment prediction",
        placeholder="Example: This product is great and the delivery is fast.",
        height=140,
    )

    if st.button("Predict", use_container_width=True):
        if not text.strip():
            st.error("Please enter some text before predicting.")
            return

        if mode == "Mock predict (random)":
            label = mock_predict()
            st.subheader(f"Prediction: {label}")
            st.info("Currently in mock mode: results are random Positive/Negative.")
            return

        assert selected_predictor is not None
        try:
            label, confidence = selected_predictor.predict(text)
            st.subheader(f"Prediction: {label}")
            if confidence is not None:
                st.write(f"Confidence: {confidence:.2%}")
        except Exception as exc:
            st.exception(exc)
            st.error(
                "Model load duoc nhung predict loi. Kiem tra dung cap model-vectorizer va schema train."
            )

    with st.expander("Expected artifact format"):
        st.markdown(
            """
            Dat file vao cung thu muc app hoac trong models/ hay artifacts/.

            1) Cach don gian nhat:
            - model.pkl la 1 sklearn Pipeline da gom vectorizer + model.

            2) Cach tach roi:
            - model.pkl: dict chua model, vi du keys: logistic, nb
            - vectorize.pkl: dict chua vectorizer, vi du keys: tfidf, bow

            App tu ghep thanh 4 setup:
            - Logistic Regression + TF-IDF
            - Logistic Regression + Bag of Word
            - Multinomial NB + TF-IDF
            - Multinomial NB + Bag of Word
            """
        )


if __name__ == "__main__":
    render_app()
