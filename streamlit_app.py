import os
import pickle
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st


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
    model_candidates = [
        "model.pkl",
        "models/model.pkl",
        "artifacts/model.pkl",
    ]
    vectorizer_candidates = [
        "vectorize.pkl",
        "vectorizer.pkl",
        "models/vectorize.pkl",
        "models/vectorizer.pkl",
        "artifacts/vectorize.pkl",
        "artifacts/vectorizer.pkl",
    ]

    models_obj = load_pickle_if_exists(model_candidates)
    vectorizers_obj = load_pickle_if_exists(vectorizer_candidates)

    if models_obj is None:
        return {}

    return build_registry(models_obj, vectorizers_obj)


def mock_predict() -> str:
    return random.choice(["Positive", "Negative"])


def render_app() -> None:
    st.set_page_config(page_title="Sentiment Demo UI", page_icon="💬", layout="centered")

    st.title("Sentiment Demo UI")
    st.caption("Mock predict nhanh hoặc test model thật từ file pkl")

    with st.sidebar:
        st.header("Mode")
        mode = st.radio(
            "Choose prediction mode",
            options=["Mock predict (random)", "Use model.pkl + vectorize.pkl"],
        )

    registry = get_predictor_registry()

    selected_predictor = None
    if mode == "Use model.pkl + vectorize.pkl":
        if not registry:
            st.warning(
                "Khong tim thay model/vectorizer hop le. App se fallback ve mock predict."
            )
            mode = "Mock predict (random)"
        else:
            selected_name = st.selectbox("Model setup", list(registry.keys()))
            selected_predictor = registry[selected_name]
            st.success(f"Loaded: {selected_name}")

    text = st.text_area(
        "Nhap text de du doan sentiment",
        placeholder="Vi du: San pham nay rat tot va giao hang nhanh.",
        height=140,
    )

    if st.button("Predict", use_container_width=True):
        if not text.strip():
            st.error("Vui long nhap noi dung truoc khi predict.")
            return

        if mode == "Mock predict (random)":
            label = mock_predict()
            st.subheader(f"Prediction: {label}")
            st.info("Dang o che do mock: ket qua random Positive/Negative.")
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
