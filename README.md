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