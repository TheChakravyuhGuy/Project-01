# MailGuard — Gmail Read-Only Spam Analysis

A local Flask college project that uses a scikit-learn text classifier to predict whether message text resembles spam. The Gmail dashboard connects through Google's Gmail API with **read-only OAuth permission** and displays predictions without changing Gmail.

## What it does
- Manual text classifier at `/`
- Gmail dashboard at `/gmail`
- Connects using Google OAuth 2.0
- Reads up to 30 newest messages in Inbox
- Parses message subject and text body; falls back to Gmail snippet if body text is unavailable
- Displays predicted spam and likely legitimate lists
- Does not send, delete, move, star, label, or modify messages
- Local token and client credentials are excluded by `.gitignore`

## Important limitations
The original starter model was trained on the UCI SMS Spam Collection, not full email. That is a prototype only and should not be presented as validated email filtering. For real email evaluation, use an email-specific corpus and follow `EMAIL_DATASET_GUIDE.md`; the included `train_email_model.py` can train from `dataset/email/ham/` and `dataset/email/spam/`. The model can make mistakes; do not rely on it as a security product.

## Windows setup
Open this folder in VS Code. In PowerShell terminal:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
python app.py
```

Open http://127.0.0.1:5000. The Gmail dashboard is at http://127.0.0.1:5000/gmail.

If the virtual environment already exists, activate it and install the new requirements. Re-run `python train_model.py` to make sure model artifacts exist.

## Google Cloud OAuth setup (local testing)

1. Open https://console.cloud.google.com/ and create/select a project.
2. Go to **APIs & Services → Library**, search for **Gmail API**, and click **Enable**.
3. Open **Google Auth Platform** (or OAuth consent screen).
4. Configure app name `MailGuard`, support email, audience `External` for a personal Gmail account, and contact email.
5. Under **Audience**, add your own Gmail address as a **Test user** while the app is in Testing.
6. Go to **Clients** / **Credentials → Create OAuth client ID**. Select **Web application**.
7. Add this exact Authorized redirect URI:
   `http://127.0.0.1:5000/oauth2callback`
8. Create the client and download the JSON file. Put it in the same folder as `app.py`, named exactly:
   `credentials.json`
9. Restart `python app.py`, visit `http://127.0.0.1:5000/gmail`, then click **Connect Gmail securely**.
10. Sign into the test account and approve the requested Gmail read-only access. Google may show an unverified-app warning during testing; only proceed for your own test project if you trust the local code and understand the requested permission.
11. After authorization, MailGuard should show the newest 30 Inbox messages split by model prediction.

OAuth UI labels can change. Official docs:
- Gmail API Python quickstart: https://developers.google.com/workspace/gmail/api/quickstart/python
- Gmail scopes: https://developers.google.com/workspace/gmail/api/auth/scopes
- OAuth consent setup: https://developers.google.com/workspace/guides/configure-oauth-consent

## Troubleshooting
- `redirect_uri_mismatch`: make sure the URI in Google Cloud is exactly `http://127.0.0.1:5000/oauth2callback`.
- `access_denied` / app blocked: ensure your Gmail address is listed under Audience → Test users and app audience is External for personal Gmail.
- `credentials.json is missing`: put the downloaded JSON file beside `app.py` and rename it `credentials.json`.
- `access blocked / app not verified`: during development, use only your own account added as a test user. Public use of restricted Gmail scopes can require Google verification and additional requirements.
- `Model missing`: run `python train_model.py` first.
- `invalid_grant` or old token: click Disconnect on `/gmail`, or stop the server and delete local `token.json`, then connect again.

## Security
- Do not commit `credentials.json`, `token.json`, `.env`, private datasets, or real email contents.
- Do not paste client secrets or tokens into chat or GitHub.
- The app reads email content into the local Flask process to classify it. Treat the machine and local account as trusted; protect access to the computer.
- This project uses a local development server bound to `127.0.0.1`; do not expose it publicly.
- The code requests only `gmail.readonly`; it does not call Gmail modification endpoints.
- If you no longer want the app to access Gmail, click Disconnect in MailGuard and revoke access in your Google Account's third-party connections/security settings.

## Data handling
The current code fetches up to 30 latest messages from the Inbox each time `/gmail` loads. Message contents are processed in memory and rendered in the local browser. No email body is deliberately written to a local database or log. Recent predictions are not persisted.

## GitHub checklist
Before publishing, ensure `.gitignore` is active and the GitHub Desktop changes list does not include `credentials.json`, `token.json`, `.venv`, private email data, or local tokens. If a secret was committed, deleting the file in a later commit is not enough; revoke/rotate the credential and remove it from Git history.
