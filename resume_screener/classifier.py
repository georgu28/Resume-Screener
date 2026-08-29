"""
Resume category classifier.

Loads the pre-trained TF-IDF + calibrated LinearSVC pipeline produced by
``train.py`` and exposes simple predict helpers. Loading a saved model (rather
than training on startup, as the old KNN version did) keeps the app fast.
"""

import logging
import os
import sys

import joblib

from resume_screener.parser import read_pdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(_ROOT, "models", "resume_clf.joblib")


class ResumeClassifier:
    """Predict a resume's job category from its text using a saved sklearn model."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        """
        Load the trained classification pipeline.

        Args:
            model_path (str): Path to the joblib model saved by train.py

        Raises:
            FileNotFoundError: If the model file is missing (run ``python train.py``)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Train it first with: python train.py"
            )
        self.model = joblib.load(model_path)
        logger.info(f"Loaded classifier with {len(self.model.classes_)} categories")

    def predict_text(self, text: str) -> str:
        """Predict the job category for raw resume text."""
        return str(self.model.predict([text])[0])

    def predict_pdf(self, pdf_path: str) -> str:
        """Predict the job category for a resume PDF."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        text = read_pdf(pdf_path)
        if not text.strip():
            logger.warning(f"No text extracted from PDF: {pdf_path}")
            return "Unknown"
        return self.predict_text(text)

    def get_categories(self) -> list:
        """Return all job categories the model can predict."""
        return list(self.model.classes_)

    def get_prediction_probabilities(self, pdf_path: str) -> dict:
        """
        Return {category: probability} for a resume PDF, sorted high to low.
        """
        text = read_pdf(pdf_path)
        proba = self.model.predict_proba([text])[0]
        prob_dict = dict(zip(self.model.classes_, (float(p) for p in proba)))
        return dict(sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python classifier.py <resume.pdf>")
        sys.exit(1)
    clf = ResumeClassifier()
    print(clf.predict_pdf(sys.argv[1]))
