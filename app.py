from pathlib import Path
from datetime import datetime
from email.utils import parseaddr
from html import unescape
from html.parser import HTMLParser
import base64
import json
import os
import re

import joblib
from flask import Flask, render_template, request, redirect, url_for, session, flash
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# LOCAL TESTING ONLY — never enable this in production
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "model" / "spam_model.joblib"
METRICS_FILE = ROOT / "model" / "metrics.json"
CLIENT_SECRETS_FILE = ROOT / "credentials.json"
TOKEN_FILE = ROOT / "token.json"

# Gmail API is read-only: no message modify, send, trash, or delete endpoints are called.
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
REDIRECT_URI = "http://127.0.0.1:5000/oauth2callback"

app = Flask(__name__)
app.secret_key = os.environ.get("MAILGUARD_SESSION_SECRET") or os.urandom(32)
# The random fallback is fine for a local demo; it invalidates sessions after restart.
# Set MAILGUARD_SESSION_SECRET to a stable secret for any persistent deployment.

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    MAX_CONTENT_LENGTH=32 * 1024,
)

history = []
model = joblib.load(MODEL_FILE) if MODEL_FILE.exists() else None
try:
    metrics = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    metrics = {}

class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip_depth = 0
    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip_depth += 1
        elif tag in ("p", "div", "br", "li", "tr"):
            self.parts.append(" ")
    def handle_endtag(self, tag):
        if tag in ("script", "style") and self.skip_depth:
            self.skip_depth -= 1
        elif tag in ("p", "div", "li", "tr"):
            self.parts.append(" ")
    def handle_data(self, data):
        if not self.skip_depth:
            self.parts.append(data)

def html_to_text(raw_html):
    parser = TextExtractor()
    try:
        parser.feed(raw_html)
        return re.sub(r"\s+", " ", unescape(" ".join(parser.parts))).strip()
    except Exception:
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", raw_html)).strip()

def decode_part(data):
    if not data:
        return ""
    try:
        raw = base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return ""

def collect_mime_parts(payload, plain_parts, html_parts):
    mime_type = payload.get("mimeType", "")
    body = payload.get("body", {})
    if mime_type == "text/plain" and body.get("data"):
        plain_parts.append(decode_part(body["data"]))
    elif mime_type == "text/html" and body.get("data"):
        html_parts.append(decode_part(body["data"]))
    for child in payload.get("parts", []) or []:
        collect_mime_parts(child, plain_parts, html_parts)

def header_value(headers, name):
    for item in headers or []:
        if item.get("name", "").lower() == name.lower():
            return item.get("value", "")
    return ""

def message_to_record(message):
    payload = message.get("payload", {})
    headers = payload.get("headers", [])
    plain_parts, html_parts = [], []
    collect_mime_parts(payload, plain_parts, html_parts)
    body = "\n".join(p for p in plain_parts if p.strip()).strip()
    if not body and html_parts:
        body = "\n".join(html_to_text(p) for p in html_parts if p.strip()).strip()
    subject = header_value(headers, "Subject") or "(No subject)"
    sender_raw = header_value(headers, "From") or "(Unknown sender)"
    sender_name, sender_email = parseaddr(sender_raw)
    sender = f"{sender_name} <{sender_email}>" if sender_name and sender_email else sender_raw
    date = header_value(headers, "Date")
    snippet = message.get("snippet", "")
    text_for_model = f"Subject: {subject}\n\n{body or snippet}".strip()
    prediction = "UNKNOWN"
    spam_score = None
    if model is not None and text_for_model:
        prediction = str(model.predict([text_for_model])[0]).lower()
        prediction = "SPAM" if prediction == "spam" else "LIKELY LEGITIMATE"
        try:
            classes = list(model.classes_)
            probs = model.predict_proba([text_for_model])[0]
            if "spam" in classes:
                spam_score = round(float(probs[classes.index("spam")]) * 100, 1)
        except (AttributeError, ValueError, IndexError):
            pass
    return {
        "id": message.get("id", ""),
        "thread_id": message.get("threadId", ""),
        "subject": subject,
        "sender": sender,
        "date": date,
        "snippet": snippet,
        "body_preview": (body or snippet)[:700],
        "prediction": prediction,
        "spam_score": spam_score,
    }

def load_saved_credentials():
    if not TOKEN_FILE.exists():
        return None
    try:
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
        return creds if creds and creds.valid else None
    except Exception:
        return None

def get_gmail_service():
    creds = load_saved_credentials()
    if not creds:
        return None
    return build("gmail", "v1", credentials=creds, cache_discovery=False)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    message = ""
    error = None
    if request.method == "POST":
        message = request.form.get("message", "").strip()
        if not message:
            error = "Please enter a message before checking it."
        elif len(message) > 5000:
            error = "Please keep the message under 5,000 characters."
        elif model is None:
            error = "Train a model first by running: python train_model.py"
        else:
            prediction = str(model.predict([message])[0]).lower()
            result = "SPAM" if prediction == "spam" else "LIKELY LEGITIMATE"
            score = None
            try:
                probs = model.predict_proba([message])[0]
                classes = list(model.classes_)
                if "spam" in classes:
                    score = round(float(probs[classes.index("spam")]) * 100, 1)
            except (AttributeError, ValueError, IndexError):
                pass
            history.insert(0, {"message": message[:160], "result": result,
                               "time": datetime.now().strftime("%H:%M:%S")})
            del history[10:]
            return render_template("index.html", result=result, message=message, error=error,
                                   score=score, metrics=metrics, history=history)
    return render_template("index.html", result=result, message=message, error=error,
                           score=None, metrics=metrics, history=history)

@app.route("/connect-gmail")
def connect_gmail():
    if not CLIENT_SECRETS_FILE.exists():
        flash("credentials.json is missing. Follow the Google Cloud setup in README.md.", "error")
        return redirect(url_for("gmail_dashboard"))
    try:
        flow = Flow.from_client_secrets_file(
            str(CLIENT_SECRETS_FILE),
            scopes=SCOPES,
            redirect_uri=REDIRECT_URI,
            autogenerate_code_verifier=True,
        )
        authorization_url, state = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
        )
        session["oauth_state"] = state
        session["oauth_code_verifier"] = flow.code_verifier
        return redirect(authorization_url)
    except Exception as exc:
        app.logger.exception("Could not start Google OAuth flow")
        flash(f"Could not start Google authorization: {exc}", "error")
        return redirect(url_for("gmail_dashboard"))

@app.route("/oauth2callback")
def oauth2callback():
    if request.args.get("error"):
        flash("Google authorization was cancelled or denied.", "error")
        return redirect(url_for("gmail_dashboard"))
    expected_state = session.get("oauth_state")
    received_state = request.args.get("state")
    if not expected_state or not received_state or expected_state != received_state:
        flash("OAuth state validation failed. Please try connecting again.", "error")
        return redirect(url_for("gmail_dashboard"))
    if not CLIENT_SECRETS_FILE.exists():
        flash("credentials.json is missing.", "error")
        return redirect(url_for("gmail_dashboard"))
    try:
        flow = Flow.from_client_secrets_file(
            str(CLIENT_SECRETS_FILE),
            scopes=SCOPES,
            state=expected_state,
            redirect_uri=REDIRECT_URI,
            code_verifier=session.get("oauth_code_verifier"),
        )
        flow.fetch_token(authorization_response=request.url)
        creds = flow.credentials
        # Local-only token file. It is excluded by .gitignore and must never be committed.
        TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
        session.pop("oauth_state", None)
        session.pop("oauth_code_verifier", None)
        flash("Gmail connected successfully in read-only mode.", "success")
    except Exception as exc:
        app.logger.exception("OAuth callback failed")
        flash(f"Google authorization failed: {exc}", "error")
    return redirect(url_for("gmail_dashboard"))

@app.route("/gmail")
def gmail_dashboard():
    connected = load_saved_credentials() is not None
    messages = []
    error = None
    mailbox_email = None
    if connected:
        try:
            service = get_gmail_service()
            profile = service.users().getProfile(userId="me").execute()
            mailbox_email = profile.get("emailAddress")
            # Read only the newest 30 messages in Inbox. This API call does not change Gmail.
            listing = service.users().messages().list(
                userId="me", labelIds=["INBOX"], maxResults=30
            ).execute()
            refs = listing.get("messages", [])
            for ref in refs:
                full = service.users().messages().get(
                    userId="me", id=ref["id"], format="full"
                ).execute()
                messages.append(message_to_record(full))
        except HttpError as exc:
            app.logger.exception("Gmail API request failed")
            error = "Gmail API request failed. Check that Gmail API is enabled and OAuth is configured."
        except Exception:
            app.logger.exception("Could not load Gmail")
            error = "Could not load Gmail messages. Try disconnecting and connecting again."
    spam_messages = [m for m in messages if m["prediction"] == "SPAM"]
    ham_messages = [m for m in messages if m["prediction"] == "LIKELY LEGITIMATE"]
    unknown_messages = [m for m in messages if m["prediction"] == "UNKNOWN"]
    return render_template(
        "gmail.html", connected=connected, mailbox_email=mailbox_email,
        messages=messages, spam_messages=spam_messages, ham_messages=ham_messages,
        unknown_messages=unknown_messages, error=error, metrics=metrics,
    )

@app.route("/disconnect-gmail", methods=["POST"])
def disconnect_gmail():
    # Removes this app's local token only. Does not alter or delete Gmail messages.
    try:
        TOKEN_FILE.unlink(missing_ok=True)
    except OSError:
        app.logger.exception("Could not remove local token file")
    session.pop("oauth_state", None)
    flash("Gmail disconnected from this local app. No Gmail messages were changed.", "success")
    return redirect(url_for("gmail_dashboard"))

@app.errorhandler(413)
def too_large(_error):
    return render_template("index.html", result=None, message="", error="Request too large.",
                           score=None, metrics=metrics, history=history), 413

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )