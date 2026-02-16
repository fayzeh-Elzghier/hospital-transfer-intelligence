import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

SEVERITY_ORDER = {"stable": 0, "urgent": 1, "critical": 2}

class ReportAnalyzer:
    def __init__(self):
        self.vec = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
        self.clf_spec = LogisticRegression(max_iter=200)
        self.clf_sev = LogisticRegression(max_iter=200)
        self.fitted = False

    def fit(self, transfers_df: pd.DataFrame):
        X = self.vec.fit_transform(transfers_df["report_text"].astype(str))
        y_spec = transfers_df["true_specialty"].astype(str)
        y_sev = transfers_df["true_severity"].astype(str)
        self.clf_spec.fit(X, y_spec)
        self.clf_sev.fit(X, y_sev)
        self.fitted = True

    def predict(self, report_text: str):
        if not self.fitted:
            raise RuntimeError("Analyzer not fitted. Call fit() first.")
        X = self.vec.transform([report_text])
        spec = self.clf_spec.predict(X)[0]
        sev = self.clf_sev.predict(X)[0]
        # probabilities for explainability
        spec_probs = dict(zip(self.clf_spec.classes_, self.clf_spec.predict_proba(X)[0]))
        sev_probs = dict(zip(self.clf_sev.classes_, self.clf_sev.predict_proba(X)[0]))
        return spec, sev, spec_probs, sev_probs
