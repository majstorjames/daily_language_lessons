# Daily Bilingual Quiz (ES + PT-BR) — GitHub Actions

This repo sends a *daily* translation quiz in **Spanish** and **Portuguese (Brazil)**, then
checks for replies every 10 minutes and emails graded feedback with explanations.
It adapts difficulty/themes based on your past answers, building a lightweight “curriculum.”

## What’s inside
- `quiz_email_es.py` / `quiz_email_pt.py`: entry points for Spanish and Portuguese.
- `core_quiz.py`: shared logic (generation, grading, Gmail-safe HTML email, IMAP thread fetch,
  adaptive history, CI-friendly persistence).
- `templates/`: HTML templates for nice Gmail rendering.
- GitHub Actions
  - `.github/workflows/daily_quiz.yml`: sends both ES and PT **daily**.
  - `.github/workflows/check_answers.yml`: checks for replies every **10 minutes** and sends graded answers.

## Required GitHub Actions Secrets
Set these in **Settings → Secrets and variables → Actions**:

- `OPENAI_API_KEY`
- `IMAP_USER` (e.g., `you@gmail.com`) — inbox we read from
- `IMAP_PASS` — **App Password** (Gmail: enable 2FA then create App password)
- `SMTP_USER` (often same as IMAP_USER)
- `SMTP_PASS` — SMTP app password
- `RECIPIENTS` — comma-separated list (e.g., `learner@example.com`)

Optional (have defaults):
- `SENDER_NAME` (defaults: “Spanish/Portuguese Quiz Bot” per script)
- `SMTP_HOST` (default `smtp.gmail.com`), `SMTP_PORT` (default `587`)
- `IMAP_HOST` (default `imap.gmail.com`), `IMAP_PORT` (default `993`)
- `IMAP_MAILBOX` (default `[Gmail]/All Mail`, falls back to `INBOX`)
- `MODEL_NAME` (default `gpt-4o-mini`)

## Scheduling
- Daily quiz: **09:00 America/New_York** (set via cron in `daily_quiz.yml`).
- Answer checker: runs **every 10 minutes**.

Adjust crons in the workflow files as you like.

## Local test
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY=sk-...
export IMAP_USER=you@gmail.com IMAP_PASS=app_pass
export SMTP_USER=you@gmail.com SMTP_PASS=app_pass
export RECIPIENTS="someone@example.com"

# Send today’s quizzes
python quiz_email_es.py --mode quiz
python quiz_email_pt.py --mode quiz

# After replying from the recipient mailbox, run:
python quiz_email_es.py --mode answers
python quiz_email_pt.py --mode answers
```

## Data persistence & adaptivity
The scripts write/update lightweight history under `data/` (tracked in git).
On GitHub Actions, the workflow is configured with permission to **commit** the updated history
so progress persists across runs. The system adapts theme/difficulty and provides a short daily
lesson before the quiz based on your weakest areas so far.
