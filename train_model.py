from pathlib import Path
import io
import zipfile
import urllib.request
import warnings

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "model"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
DATASET_FILE = DATA_DIR / "SMSSpamCollection"
MODEL_FILE = MODEL_DIR / "spam_model.joblib"
METRICS_FILE = MODEL_DIR / "metrics.json"

# UCI SMS Spam Collection. This is SMS data, not a corpus of full email messages.
DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

# Small offline fallback so the app can be demonstrated without downloading a dataset.
# It is for a demo only; results are NOT suitable for reporting as meaningful model performance.
FALLBACK_ROWS = [
    ("ham", "Hey, are we still meeting at the library at 4?"),
    ("ham", "Please send me the lecture notes when you get time."),
    ("ham", "Your appointment is confirmed for tomorrow morning."),
    ("ham", "Can you call me after class?"),
    ("ham", "The project presentation is scheduled for Friday."),
    ("ham", "I will reach home in about twenty minutes."),
    ("ham", "Don't forget to submit the assignment today."),
    ("ham", "Thanks for helping me with the programming question."),
    ("ham", "The train is delayed by ten minutes."),
    ("ham", "Are you coming to college tomorrow?"),
    ("ham", "Your monthly statement is now available to view."),
    ("ham", "Let's discuss the project after the lab."),
    ("ham", "I have shared the document with you."),
    ("ham", "Happy birthday! Have a great day."),
    ("ham", "Please bring your notebook to the meeting."),
    ("spam", "Congratulations! You have been selected for our new credit card offer!"),
    ("spam", "Congratulations! You have won a Lottery Claim now!"),
    ("spam", "Congratulations! You have won a free prize. Claim now!"),
    ("spam", "URGENT! You have been selected to receive a cash reward."),
    ("spam", "URGENT! Your bank account froze due to suspicious activity."),
    ("spam", "Win a free gift card today. Click here to claim."),
    ("spam", "Limited time offer! Buy now and save 90 percent."),
    ("spam", "You are our lucky winner. Call this number immediately."),
    ("spam", "Claim your guaranteed bonus now, offer expires today."),
    ("spam", "Free entry in a prize draw. Text WIN to enter."),
    ("spam", "Earn money fast from home with no experience required."),
    ("spam", "Act now to receive your exclusive reward."),
    ("spam", "You have been chosen for a free holiday. Claim today."),
    ("spam", "Get a cheap loan instantly. Apply now."),
    ("spam", "Congratulations winner! Click the link to collect cash."),
    ("spam", "Exclusive deal just for you. Buy now."),
    ("spam", "You have won a brand new phone. Claim your prize."),
    ("spam", "Free offer waiting for you. Reply now to qualify."),
    ("ham", "The professor moved the lecture to room 204."),
    ("ham", "I will email you the notes this evening."),
    ("ham", "Your package has been delivered to reception."),
    ("ham", "Can we reschedule our call for Monday?"),
    ("ham", "The team meeting starts at 2 pm."),
    ("ham", "I have attached the revised report."),
    ("ham", "Please confirm if you received my message."),
    ("ham", "Your password reset request was received."),
    ("ham", "The college fest registration closes next week."),
    ("ham", "Lunch is ready whenever you are."),
    ("spam", "Get a free vacation now. Limited seats available."),
    ("spam", "You qualify for a special cash prize. Claim immediately."),
    ("spam", "Lowest price guaranteed. Order today and get a bonus."),
    ("spam", "Winner alert! Your reward is waiting. Click to claim."),
    ("spam", "Make quick cash with this amazing opportunity."),
    ("spam", "Congratulations, claim your free voucher now."),
    ("spam", "Act fast to unlock your exclusive discount."),
    ("spam", "You have been selected for a special offer."),
    ("spam", "Free membership offer. Sign up now."),
    ("spam", "Claim your bonus today before the offer expires."),
]

def download_dataset():
    if DATASET_FILE.exists() and DATASET_FILE.stat().st_size > 1000:
        return True
    try:
        print("Downloading UCI SMS Spam Collection dataset...")
        request = urllib.request.Request(
            DATASET_URL, headers={"User-Agent": "EmailSpamDetectionCollegeProject/1.0"}
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            archive_bytes = response.read()
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
            member = next((n for n in zf.namelist() if n.endswith("SMSSpamCollection")), None)
            if not member:
                raise ValueError("Dataset file was not found inside the downloaded ZIP.")
            DATASET_FILE.write_bytes(zf.read(member))
        print(f"Dataset saved to: {DATASET_FILE}")
        return True
    except Exception as exc:
        print(f"Dataset download failed: {exc}")
        print("Using a tiny built-in demonstration dataset instead.")
        return False

def load_data():
    downloaded = download_dataset()
    if downloaded:
        df = pd.read_csv(
            DATASET_FILE, sep="\t", header=None, names=["label", "message"],
            encoding="utf-8", on_bad_lines="skip"
        )
        df["label"] = df["label"].astype(str).str.lower().str.strip()
        df = df[df["label"].isin(["ham", "spam"])]
        df["message"] = df["message"].astype(str).str.strip()
        df = df[df["message"].str.len() > 0]
        df = df.drop_duplicates(subset=["message"])
        return df, True
    return pd.DataFrame(FALLBACK_ROWS, columns=["label", "message"]), False

def main():
    df, real_dataset = load_data()
    if df["label"].nunique() < 2 or len(df) < 10:
        raise RuntimeError("Dataset must contain at least 10 messages across spam and ham labels.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    candidates = {
        "Multinomial Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=1)),
            ("classifier", MultinomialNB())
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=1)),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ]),
    }

    results = {}
    best_name, best_model, best_f1 = None, None, -1
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, preds)), 4),
            "precision_spam": round(float(precision_score(y_test, preds, pos_label="spam", zero_division=0)), 4),
            "recall_spam": round(float(recall_score(y_test, preds, pos_label="spam", zero_division=0)), 4),
            "f1_spam": round(float(f1_score(y_test, preds, pos_label="spam", zero_division=0)), 4),
            "confusion_matrix_labels": ["ham", "spam"],
            "confusion_matrix": confusion_matrix(y_test, preds, labels=["ham", "spam"]).tolist(),
            "classification_report": classification_report(y_test, preds, labels=["ham", "spam"], zero_division=0, output_dict=True),
        }
        results[name] = metrics
        print(f"\n{name}")
        print(f"Accuracy: {metrics['accuracy']:.3f} | Spam precision: {metrics['precision_spam']:.3f} | "
              f"Spam recall: {metrics['recall_spam']:.3f} | Spam F1: {metrics['f1_spam']:.3f}")
        print(classification_report(y_test, preds, zero_division=0))
        if metrics["f1_spam"] > best_f1:
            best_name, best_model, best_f1 = name, model, metrics["f1_spam"]

    # Refit the selected pipeline on all available data for app predictions.
    best_model.fit(df["message"], df["label"])
    joblib.dump(best_model, MODEL_FILE)
    payload = {
        "selected_model": best_name,
        "dataset": "UCI SMS Spam Collection" if real_dataset else "Built-in tiny demo dataset (not for meaningful evaluation)",
        "dataset_is_real": real_dataset,
        "total_messages": int(len(df)),
        "spam_messages": int((df["label"] == "spam").sum()),
        "ham_messages": int((df["label"] == "ham").sum()),
        "train_test_split": "80/20 stratified, random_state=42",
        "results": results,
        "warning": None if real_dataset else "Offline fallback dataset is tiny and illustrative; download the UCI dataset before reporting performance."
    }
    METRICS_FILE.write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSelected model: {best_name}")
    print(f"Saved model: {MODEL_FILE}")
    print(f"Saved metrics: {METRICS_FILE}")
    if not real_dataset:
        warnings.warn("Tiny demo dataset used. Do not report these metrics as representative performance.")

if __name__ == "__main__":
    main()
