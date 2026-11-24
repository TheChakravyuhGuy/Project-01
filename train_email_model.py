"""Train MailGuard on an email-specific corpus stored in dataset/email/{ham,spam}."""
from pathlib import Path
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
from html import unescape
import re
import json
import warnings

import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parent
CORPUS_DIR = ROOT / "dataset" / "email"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_FILE = MODEL_DIR / "spam_model.joblib"
METRICS_FILE = MODEL_DIR / "metrics.json"

class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
    def handle_data(self, data):
        self.parts.append(data)

def html_to_text(html):
    parser = TextExtractor()
    try:
        parser.feed(html)
        return re.sub(r"\s+", " ", unescape(" ".join(parser.parts))).strip()
    except Exception:
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()

def decode_part(part):
    try:
        content = part.get_content()
        if isinstance(content, str):
            return content
    except Exception:
        pass
    try:
        raw = part.get_payload(decode=True)
        if raw:
            charset = part.get_content_charset() or "utf-8"
            return raw.decode(charset, errors="replace")
    except Exception:
        pass
    return ""

def email_to_text(path):
    try:
        with open(path, "rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)
        subject = str(msg.get("subject", "") or "")
        plain, html = [], []
        parts = msg.walk() if msg.is_multipart() else [msg]
        for part in parts:
            if part.get_content_maintype() == "multipart":
                continue
            if part.get_filename():
                continue  # Never train on attachment payloads.
            ctype = part.get_content_type()
            if ctype == "text/plain":
                txt = decode_part(part)
                if txt.strip():
                    plain.append(txt)
            elif ctype == "text/html":
                txt = decode_part(part)
                if txt.strip():
                    html.append(html_to_text(txt))
        body = "\n".join(plain).strip() or "\n".join(html).strip()
        return f"Subject: {subject}\n\n{body}".strip()
    except Exception:
        return ""

def main():
    rows = []
    for label in ("ham", "spam"):
        folder = CORPUS_DIR / label
        if not folder.exists():
            raise SystemExit(
                f"Missing folder: {folder}\n"
                "Create dataset/email/ham and dataset/email/spam, then extract email corpus files into them. "
                "See EMAIL_DATASET_GUIDE.md."
            )
        for path in folder.rglob("*"):
            if path.is_file():
                text = email_to_text(path)
                if len(text.strip()) >= 10:
                    rows.append((label, text))
    if len(rows) < 100 or len({label for label, _ in rows}) < 2:
        raise SystemExit(f"Found only {len(rows)} usable emails. Please add a larger corpus with both ham and spam.")
    texts = [text for _, text in rows]
    labels = [label for label, _ in rows]
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    candidates = {
        "Multinomial Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=2, max_features=100000)),
            ("classifier", MultinomialNB())
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=2, max_features=100000)),
            ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced"))
        ]),
    }
    results = {}
    best_name, best_model, best_f1 = None, None, -1
    for name, candidate in candidates.items():
        candidate.fit(X_train, y_train)
        pred = candidate.predict(X_test)
        item = {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision_spam": round(float(precision_score(y_test, pred, pos_label="spam", zero_division=0)), 4),
            "recall_spam": round(float(recall_score(y_test, pred, pos_label="spam", zero_division=0)), 4),
            "f1_spam": round(float(f1_score(y_test, pred, pos_label="spam", zero_division=0)), 4),
            "confusion_matrix_labels": ["ham", "spam"],
            "confusion_matrix": confusion_matrix(y_test, pred, labels=["ham", "spam"]).tolist(),
            "classification_report": classification_report(y_test, pred, labels=["ham", "spam"], zero_division=0, output_dict=True),
        }
        results[name] = item
        print(f"\n{name}: accuracy={item['accuracy']:.3f}, spam precision={item['precision_spam']:.3f}, "
              f"spam recall={item['recall_spam']:.3f}, spam F1={item['f1_spam']:.3f}")
        print(classification_report(y_test, pred, zero_division=0))
        if item["f1_spam"] > best_f1:
            best_name, best_model, best_f1 = name, candidate, item["f1_spam"]
    # Retrain selected pipeline on all corpus messages for app inference.
    best_model.fit(texts, labels)
    joblib.dump(best_model, MODEL_FILE)
    payload = {
        "selected_model": best_name,
        "dataset": "Email-specific corpus in dataset/email/{ham,spam}",
        "dataset_is_real": True,
        "dataset_type": "email",
        "total_messages": len(rows),
        "spam_messages": labels.count("spam"),
        "ham_messages": labels.count("ham"),
        "train_test_split": "80/20 stratified, random_state=42",
        "results": results,
        "warning": "Metrics are from one held-out split; validate on a separate corpus before making broad claims."
    }
    METRICS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSelected model: {best_name}")
    print(f"Saved model to {MODEL_FILE}")
    print(f"Saved metrics to {METRICS_FILE}")

if __name__ == "__main__":
    main()
