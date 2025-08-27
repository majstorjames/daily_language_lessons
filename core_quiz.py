
# core_quiz.py
import os, ssl, smtplib, imaplib, email, uuid, json, re, subprocess
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from collections import Counter

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

# ---------------- Debug helper ----------------
def _debug(msg: str):
    """Print debug lines when QUIZ_DEBUG is set (any non-empty)."""
    if os.environ.get("QUIZ_DEBUG"):
        print("[DEBUG]", msg)

# ---------------- OpenAI ----------------
def oa_client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required")
    _debug("Initialized OpenAI client")
    return OpenAI(api_key=key)

def model_name() -> str:
    name = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    _debug(f"Using model: {name}")
    return name

# ---------------- History persistence (CI-friendly) ----------------
def history_path(lang_code: str) -> str:
    os.makedirs("data", exist_ok=True)
    return os.path.join("data", f"{lang_code}_history.json")

def load_history(lang_code: str) -> Dict:
    p = history_path(lang_code)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            hist = json.load(f)
            _debug(f"Loaded history for {lang_code}: {len(hist.get('sessions',[]))} sessions")
            return hist
    _debug(f"No history for {lang_code}; starting fresh")
    return {"sessions": []}

def save_history(lang_code: str, hist: Dict):
    p = history_path(lang_code)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False, indent=2)
    _debug(f"Saved history for {lang_code} ({len(hist.get('sessions',[]))} sessions)")

def git_commit_if_ci(message: str):
    if os.environ.get("GITHUB_ACTIONS") == "true":
        try:
            subprocess.run(["git", "config", "--global", "user.name", "quiz-bot"], check=True)
            subprocess.run(["git", "config", "--global", "user.email", "quiz-bot@users.noreply.github.com"], check=True)
            subprocess.run(["git", "add", "data"], check=True)
            subprocess.run(["git", "diff", "--cached", "--quiet"], check=False)
            subprocess.run(["git", "commit", "-m", message], check=False)
            subprocess.run(["git", "push"], check=False)
            _debug("Committed history to repo (CI)")
        except Exception as e:
            print("[WARN] git commit skipped:", e)

# ---------------- Minimal template renderer ----------------
def render_template(path: str, context: Dict) -> str:
    """
    Tiny mustache-like template engine supporting:
      - {{var}} (escaped)
      - {{{var}}} (raw, unescaped)
      - {% for item in items %}...{% endfor %} with {{item.key}} / {{{item.key}}}
    """
    from html import escape
    import re

    _debug(f"Rendering template: {path}")
    html = open(path, "r", encoding="utf-8").read()

    # Loops
    loop_pat = re.compile(r"{% for (\w+) in (\w+) %}(.+?){% endfor %}", re.DOTALL)
    while True:
        m = loop_pat.search(html)
        if not m:
            break
        var, arr, body = m.group(1), m.group(2), m.group(3)
        rendered_chunks = []
        seq = context.get(arr, [])
        if isinstance(seq, list):
            for elem in seq:
                chunk = body
                if isinstance(elem, dict):
                    # nested raw {{{item.key}}}
                    for k, v in elem.items():
                        chunk = re.sub(
                            r"{{{\s*%s\.%s\s*}}}" % (re.escape(var), re.escape(k)),
                            lambda _: str(v),
                            chunk,
                        )
                    # nested escaped {{item.key}}
                    for k, v in elem.items():
                        chunk = re.sub(
                            r"{{\s*%s\.%s\s*}}" % (re.escape(var), re.escape(k)),
                            lambda _: escape(str(v)),
                            chunk,
                        )
                rendered_chunks.append(chunk)
        html = html[:m.start()] + "".join(rendered_chunks) + html[m.end():]

    # RAW first: {{{var}}}
    raw_top = re.compile(r"{{{\s*(\w+)\s*}}}")
    html = raw_top.sub(lambda m: str(context.get(m.group(1), "")), html)

    # Escaped: {{var}}
    esc_top = re.compile(r"{{\s*(\w+)\s*}}")
    html = esc_top.sub(lambda m: escape(str(context.get(m.group(1), ""))), html)

    return html

# ---------------- Email (HTML + plain) ----------------
def send_email(subject: str, html_body: str, plain_body: str, *, message_id=None, in_reply_to=None, references=None, sender_default="Quiz Bot", language="en"):
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ["SMTP_USER"]
    pwd = os.environ["SMTP_PASS"]

    # Choose recipients by language bucket env vars
    if language == 'es':
        recipients = [x.strip() for x in os.environ["ES_RECIPIENTS"].split(",")]
    elif language == 'pt':
        recipients = [x.strip() for x in os.environ["PT_RECIPIENTS"].split(",")]
    else:
        recipients = [x.strip() for x in os.environ["RECIPIENTS"].split(",")]

    sender_name = os.environ.get("SENDER_NAME", sender_default)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = formataddr((sender_name, user))
    msg["Reply-To"] = user
    msg["To"] = ", ".join(recipients)
    if message_id:  msg["Message-ID"] = message_id
    if in_reply_to: msg["In-Reply-To"] = in_reply_to
    if references:  msg["References"] = references

    part1 = MIMEText(plain_body or "", "plain", "utf-8")
    part2 = MIMEText(html_body or "", "html", "utf-8")
    msg.attach(part1); msg.attach(part2)

    _debug(f"Sending email to {len(recipients)} recipient(s); subject={subject!r}; lang={language}")
    ctx = ssl.create_default_context()
    with smtplib.SMTP(host, port) as s:
        s.starttls(context=ctx)
        s.login(user, pwd)
        s.sendmail(user, recipients, msg.as_string())
    _debug("Email sent")

# ---------------- IMAP helpers (Gmail thread id strategy) ----------------
def imap_connect():
    host = os.environ.get("IMAP_HOST", "imap.gmail.com")
    port = int(os.environ.get("IMAP_PORT", "993"))
    user = os.environ.get("IMAP_USER")
    pwd  = os.environ.get("IMAP_PASS")
    if not user or not pwd:
        raise RuntimeError("IMAP_USER and IMAP_PASS are required for reply-threading.")
    _debug(f"Connecting IMAP {host}:{port} as {user}")
    M = imaplib.IMAP4_SSL(host, port)
    M.login(user, pwd)
    return M

def _select_all_mail(M):
    mailbox = os.environ.get("IMAP_MAILBOX", "[Gmail]/All Mail")
    try:
        typ, _ = M.select(mailbox)
        if typ == 'OK':
            _debug(f"Selected mailbox: {mailbox}")
            return mailbox
    except Exception as e:
        _debug(f"Failed to select {mailbox}: {e}")
    M.select("INBOX"); _debug("Selected INBOX")
    return "INBOX"

def _imap_quote(s: str) -> bytes:
    s = s.replace("\\","\\\\").replace('"','\\"')
    return f'"{s}"'.encode("utf-8")

def _uid_search_header_exact(M, field: str, value: str):
    field_b = field.encode("ascii", "ignore")
    value_b = _imap_quote(value)
    typ, data = M.uid('SEARCH', b'HEADER', field_b, value_b)
    _debug(f"HEADER {field}={value!r} -> {typ} {data and len(data[0].split())}")
    if typ != 'OK' or not data: return []
    return data[0].split()

def _fetch_by_uid(M, uid: bytes):
    typ, msg_data = M.uid('FETCH', uid, '(RFC822)')
    _debug(f"FETCH RFC822 uid={uid} -> {typ}")
    if typ != 'OK' or not msg_data or not msg_data[0]: return None
    return email.message_from_bytes(msg_data[0][1])

def _fetch_thrid(M, uid: bytes) -> Optional[str]:
    typ, msg_data = M.uid('FETCH', uid, '(X-GM-THRID)')
    _debug(f"FETCH X-GM-THRID uid={uid} -> {typ}")
    if typ != 'OK' or not msg_data: return None
    parts = []
    for tup in msg_data:
        if isinstance(tup, tuple):
            for p in tup:
                if isinstance(p, (bytes, bytearray)): parts.append(p)
        elif isinstance(tup, (bytes, bytearray)):
            parts.append(tup)
    blob = b' '.join(parts)
    m = re.search(rb'X-GM-THRID\s+(\d+)', blob)
    return m.group(1).decode() if m else None

def get_text_from_email(msg: email.message.Message) -> str:
    text = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True) or b""
                text += payload.decode(part.get_content_charset() or "utf-8", errors="replace")
    else:
        payload = msg.get_payload(decode=True) or b""
        text = payload.decode(msg.get_content_charset() or "utf-8", errors="replace")
    cleaned = []
    for line in text.splitlines():
        if line.strip().startswith(">"): continue
        if re.match(r"^On .* wrote:$", line.strip()): break
        cleaned.append(line)
    return "\n".join(cleaned).strip()

# ---------- Stronger reply discovery ----------
def _id_variants(message_id: str) -> List[str]:
    s = (message_id or "").strip()
    if not s:
        return []
    s = re.sub(r"\s+", "", s)
    no_br = s.strip("<>")
    variants = [s, f"<{no_br}>", no_br]
    variants += [v.lower() for v in variants]
    seen = set(); out = []
    for v in variants:
        if v not in seen:
            out.append(v); seen.add(v)
    return out

def _uid_search_header_any(M, field: str, values: List[str]) -> List[bytes]:
    hits: List[bytes] = []
    for val in values:
        try:
            field_b = field.encode("ascii", "ignore")
            value_b = _imap_quote(val)
            typ, data = M.uid("SEARCH", b"HEADER", field_b, value_b)
            _debug(f"HEADER {field} {val!r} -> {typ} {data and len(data[0].split())}")
            if typ == "OK" and data and data[0]:
                hits = data[0].split()
                if hits:
                    return hits
        except Exception as e:
            _debug(f"HEADER search error for {field}={val!r}: {e}")
    return hits

def _uid_search_xgm_raw_msgid(M, message_id: str) -> List[bytes]:
    uids: List[bytes] = []
    for val in _id_variants(message_id):
        token = val.strip("<>")
        try:
            q = f'rfc822msgid:{token}'
            typ, data = M.uid("SEARCH", b"X-GM-RAW", _imap_quote(q))
            _debug(f'X-GM-RAW "{q}" -> {typ} {data and len(data[0].split())}')
            if typ == "OK" and data and data[0]:
                uids = data[0].split()
                if uids:
                    return uids
        except Exception as e:
            _debug(f"X-GM-RAW error for {token!r}: {e}")
    return uids

def find_latest_reply(quiz_message_id: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (reply_message_id, reply_plain_text) if a reply is found in the thread."""
    M = imap_connect()
    try:
        _select_all_mail(M)

        # 1) Find the original quiz message by any means
        base_uids: List[bytes] = []
        if quiz_message_id:
            variants = _id_variants(quiz_message_id)
            base_uids = _uid_search_header_any(M, "Message-ID", variants)
            if not base_uids:
                base_uids = _uid_search_header_any(M, "Message-Id", variants)
            if not base_uids:
                base_uids = _uid_search_xgm_raw_msgid(M, quiz_message_id)

        thrid: Optional[str] = None
        if base_uids:
            thrid = _fetch_thrid(M, base_uids[-1])
            _debug(f"Thread id: {thrid}")

        # 2) If we have thread id, take the latest message (except the quiz itself)
        if thrid:
            typ, data = M.uid("SEARCH", None, b"X-GM-THRID", thrid.encode())
            _debug(f"SEARCH X-GM-THRID -> {typ}")
            if typ == "OK" and data and data[0]:
                t_uids = data[0].split()
                if t_uids:
                    candidate = t_uids[-1]
                    if base_uids and candidate == base_uids[-1] and len(t_uids) > 1:
                        candidate = t_uids[-2]
                    msg = _fetch_by_uid(M, candidate)
                    if msg:
                        _debug("Reply found via thread id")
                        return msg.get("Message-ID"), get_text_from_email(msg)

        # 3) Fallback: look for In-Reply-To / References directly
        if quiz_message_id:
            variants = _id_variants(quiz_message_id)
            uids = _uid_search_header_any(M, "In-Reply-To", variants)
            if not uids:
                uids = _uid_search_header_any(M, "References", variants)
            if uids:
                msg = _fetch_by_uid(M, uids[-1])
                if msg:
                    _debug("Reply found via header fallback")
                    return msg.get("Message-ID"), get_text_from_email(msg)

        _debug("No reply found")
        return None, None
    finally:
        try:
            M.logout()
        except Exception:
            pass

# ---------------- HTML utilities ----------------
def _strip_code_fences(s: str) -> str:
    """Remove Markdown code fences like ``` or ```html."""
    s = s.strip()
    s = re.sub(r"^\s*```[a-zA-Z0-9_-]*\s*", "", s)
    s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()

def _ensure_html_list(s: str) -> str:
    """If not HTML, wrap lines into <ul><li>..</li></ul>."""
    if "<" in s and ">" in s:
        return s
    lines = [ln.strip(" -*•\t").strip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return "<ul></ul>"
    return "<ul>" + "".join(f"<li>{ln}</li>" for ln in lines) + "</ul>"

# ---------------- Labels & helpers for templating ----------------
def ui_labels(lang: str) -> Dict[str, str]:
    """All strings the HTML templates expect, per language."""
    if lang == "pt":
        return {
            # quiz
            "quiz_title": "Quiz diário de Português",
            "theme_label": "Tema",
            "mini_lesson_label": "Mini-lição",
            "translate_into_label": "Traduza para Inglês:",
            "reply_hint": 'Responda a este e-mail com suas traduções numeradas (ex.: "1. ...").',
            "quiz_footer": "Enviado automaticamente pelo seu quiz bot. Continue aprendendo! 💪",
            # answers
            "answers_title": "Respostas do Quiz",
            "summary_label": "Resumo",
            "feedback_label": "Feedback",
            "growth_hint": "Foque nos pontos em revisão. Amanhã vamos reforçar essas áreas.",
            "results_label": "Resultados",
            "your_answer_label": "Sua resposta",
            "correct_label": "Resposta correta",
            "next_steps_hint": "Continuaremos adaptando o quiz de acordo com seu desempenho. 🚀",
            "answers_footer": "Enviado automaticamente pelo seu quiz bot. 🎯",
        }
    elif lang == "es":
        return {
            "quiz_title": "Quiz diario de Español",
            "theme_label": "Tema",
            "mini_lesson_label": "Mini-lección",
            "translate_into_label": "Traduce al Español:",
            "reply_hint": 'Responde a este correo con tus traducciones numeradas (p. ej., "1. ...").',
            "quiz_footer": "Enviado automáticamente por tu bot de quiz. ¡Sigue aprendiendo! 💪",
            "answers_title": "Respuestas del Quiz",
            "summary_label": "Resumen",
            "feedback_label": "Retroalimentación",
            "growth_hint": "Enfócate en los puntos a reforzar. Mañana practicaremos esas áreas.",
            "results_label": "Resultados",
            "your_answer_label": "Tu respuesta",
            "correct_label": "Respuesta correcta",
            "next_steps_hint": "Iremos adaptando el quiz a tu progreso. 🚀",
            "answers_footer": "Enviado automáticamente por tu bot de quiz. 🎯",
        }
    # default EN
    return {
        "quiz_title": "Daily Language Quiz",
        "theme_label": "Theme",
        "mini_lesson_label": "Mini-lesson",
        "translate_into_label": "Translate into:",
        "reply_hint": 'Reply with your numbered translations (e.g., "1. ...").',
        "quiz_footer": "Sent automatically by your quiz bot. Keep learning! 💪",
        "answers_title": "Quiz Answers",
        "summary_label": "Summary",
        "feedback_label": "Feedback",
        "growth_hint": "Focus on items needing review. We’ll reinforce them tomorrow.",
        "results_label": "Results",
        "your_answer_label": "Your answer",
        "correct_label": "Correct answer",
        "next_steps_hint": "We’ll keep adapting the quiz to your performance. 🚀",
        "answers_footer": "Sent automatically by your quiz bot. 🎯",
    }


def _status_label_and_class(status: str, lang: str) -> Tuple[str, str]:
    s = (status or "").lower()
    if s.startswith("correct"):
        return ("Correto" if lang=="pt" else "Correcto" if lang=="es" else "Correct", "ok")
    if s.startswith("almost"):
        return ("Quase lá" if lang=="pt" else "Casi" if lang=="es" else "Almost", "almost")
    return ("Precisa de revisão" if lang=="pt" else "Necesita revisión" if lang=="es" else "Needs work", "needs")


# ---------------- Processed-replies ledger (CI-friendly) ----------------
def _processed_path(lang_code: str) -> str:
    os.makedirs("data", exist_ok=True)
    return os.path.join("data", f"{lang_code}_processed_replies.json")

def _load_processed(lang_code: str) -> Dict:
    p = _processed_path(lang_code)
    if os.path.exists(p):
        try:
            return json.load(open(p, "r", encoding="utf-8"))
        except Exception:
            pass
    return {"keys": []}

def _save_processed(lang_code: str, data: Dict):
    json.dump(data, open(_processed_path(lang_code), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def has_processed_reply(lang_code: str, key: str) -> bool:
    """Return True if this quiz|reply pair was already graded & sent."""
    data = _load_processed(lang_code)
    return key in data.get("keys", [])

def record_processed_reply(lang_code: str, key: str, max_keep: int = 500):
    """Remember we graded/sent this quiz|reply, keeping the file small."""
    data = _load_processed(lang_code)
    if key not in data["keys"]:
        data["keys"].append(key)
        if len(data["keys"]) > max_keep:
            data["keys"] = data["keys"][-max_keep:]
        _save_processed(lang_code, data)
    _debug(f"Recorded processed reply key={key!r}")
    
# ---------------- Adaptive generation & grading ----------------
def compute_theme_and_lesson(lang_code: str, target_lang_name: str) -> Tuple[str, str]:
    """
    Pick a theme from history and return (theme, mini_lesson_html).
    Prompt controls mini-lesson language:
      - 'es': English mini-lesson
      - 'pt': Portuguese mini-lesson
    """
    hist = load_history(lang_code)
    topic_counts = {}
    for s in hist["sessions"][-30:]:
        for it in s.get("items", []):
            for t in it.get("topics", []):
                if it.get("status") == "Needs work":
                    topic_counts[t] = topic_counts.get(t, 0) + 1

    if not hist["sessions"]:
        theme = "Saudações e apresentações" if lang_code == "pt" else "Greetings & Introductions"
    elif topic_counts:
        theme = max(topic_counts, key=topic_counts.get)
    else:
        theme = "Ações do dia a dia" if lang_code == "pt" else "Everyday actions"

    c = oa_client()
    if lang_code == "pt":
        prompt = (
            f"Escreva uma mini-lição curta, para iniciantes, em Português, sobre o tema '{theme}'. "
            "Devolva um único snippet HTML VÁLIDO (ex.: <ul><li>…</li></ul>, com <strong> opcional). "
            "Sem markdown, sem blocos de código, sem crases. 3–5 tópicos objetivos."
        )
    else:
        prompt = (
            f"Write a short, beginner-friendly mini-lesson in English about the theme '{theme}'. "
            "Return a single, VALID HTML snippet only (e.g., <ul><li>…</li></ul> with optional <strong>). "
            "Do NOT include markdown or code fences. Keep it to 3–5 concise bullets."
        )

    resp = c.chat.completions.create(
        model=model_name(),
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You output a single valid HTML snippet (no markdown, no code fences)."},
            {"role": "user", "content": prompt},
        ],
    )

    html = (resp.choices[0].message.content or "").strip()
    html = _strip_code_fences(html)
    html = _ensure_html_list(html)
    _debug(f"Mini-lesson generated ({len(html)} chars)")
    return theme, html

def generate_items(lang_code: str, target_lang_name: str, schema: Dict, beginner: bool) -> List[Dict]:
    """
    Generate 10 items tailored to theme & difficulty, deduped against history.
    Prompt-only direction:
      - 'es': English → Spanish (explanations in EN)
      - 'pt': Portuguese → English (explanations in PT)
    For 'pt' we store the Portuguese sentence in 'en' so the quiz template can display it.
    """
    hist = load_history(lang_code)

    seen_pairs = set()
    for s in hist.get("sessions", []):
        for it in s.get("items", []):
            en_prev = (it.get("en") or "").lower()
            gold_prev = (it.get("gold") or "").lower()
            if en_prev and gold_prev:
                seen_pairs.add((en_prev, gold_prev))

    theme, lesson_html = compute_theme_and_lesson(lang_code, target_lang_name)

    c = oa_client()
    examples_to_avoid = [{"en": en, "target": tgt} for (en, tgt) in list(seen_pairs)[-100:]]
    difficulty = "very easy" if beginner and not hist.get("sessions") else "progressively harder but still beginner-friendly"

    if lang_code == "pt":
        instruction = f"""
Você está gerando um quiz de tradução de Português → Inglês.
- 10 itens, frases naturais do dia a dia em português (4–12 palavras).
- Principalmente presente; inclua algumas no passado e no futuro.
- Registro neutro; evite gírias/idiomas.
- Tema: {theme}.
- Evite colisões com itens anteriores (fornecidos).
Retorne um array JSON com 10 objetos:
  "pt": frase em português,
  "target": tradução correta em inglês,
  "topics": array com 2–4 etiquetas curtas,
  "explanation_pt": explicação curta (1 frase) em português.
Apenas o JSON.
Dificuldade: {difficulty}.
""".strip()
    else:
        instruction = f"""
You are generating an English → Spanish translation quiz.
- 10 items, natural everyday English (4–12 words).
- Bias toward present; include a few past/future.
- Neutral register; no idioms.
- Theme: {theme}.
- Avoid collisions with previous items (provided).
Return a JSON array of 10 objects:
  "en": English sentence,
  "target": correct Spanish translation,
  "topics": 2–4 short tags,
  "explanation_en": one-sentence explanation in English.
Only the JSON array.
Difficulty: {difficulty}.
""".strip()

    resp = c.chat.completions.create(
        model=model_name(),
        temperature=0.2,
        messages=[
            {"role": "system", "content": "Return strictly valid JSON (array or {items:[...]}); no commentary."},
            {"role": "user", "content": json.dumps({"avoid": examples_to_avoid, "instruction": instruction}, ensure_ascii=False)},
        ],
    )
    text = (resp.choices[0].message.content or "").strip()
    _debug(f"Raw items response length: {len(text)}")

    # ---- Robust JSON parsing ----
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1].strip()

    items = None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "items" in obj:
            items = obj["items"]
        elif isinstance(obj, list):
            items = obj
    except Exception as e:
        _debug(f"Primary JSON parse failed: {e}")
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            items = json.loads(m.group(0))

    if not isinstance(items, list):
        raise RuntimeError("Model did not return a valid JSON list of items.")

    # ---- Normalize output & drop bad entries/dupes ----
    normalized: List[Dict] = []
    for it in items:
        if not isinstance(it, dict):
            continue

        if lang_code == "pt":
            pt_src = it.get("pt")
            tgt = it.get("target")
            topics = it.get("topics") or []
            expl_pt = it.get("explanation_pt") or ""

            if not isinstance(pt_src, str) or not isinstance(tgt, str):
                continue

            key = (pt_src.lower().strip(), tgt.lower().strip())
            if key in seen_pairs:
                continue

            if not isinstance(topics, list):
                topics = [str(topics)]

            normalized.append({
                "en": pt_src.strip(),          # Store PT source in 'en' so template shows it
                "target": tgt.strip(),         # English answer
                "topics": [str(t).strip() for t in topics],
                "explanation_pt": str(expl_pt).strip(),
            })

        else:
            en_src = it.get("en")
            tgt = it.get("target") or it.get("es")
            topics = it.get("topics") or []
            expl_en = it.get("explanation_en") or ""

            if not isinstance(en_src, str) or not isinstance(tgt, str):
                continue

            key = (en_src.lower().strip(), tgt.lower().strip())
            if key in seen_pairs:
                continue

            if not isinstance(topics, list):
                topics = [str(topics)]

            normalized.append({
                "en": en_src.strip(),
                "target": tgt.strip(),         # Spanish answer
                "topics": [str(t).strip() for t in topics],
                "explanation_en": str(expl_en).strip(),
            })

    if len(normalized) == 0:
        raise RuntimeError("No valid quiz items produced by the model.")

    _debug(f"Prepared {len(normalized)} normalized items")
    return theme, lesson_html, normalized[:10]

def grade_answers(lang_code: str, target_lang_name: str, gold_items: List[Dict], user_answers: Dict[int,str]) -> Tuple[str, List[Dict]]:
    """
    Grade answers with OpenAI and update history.

    For every answer (Correct / Almost / Needs work) we include:
      - status: internal key
      - status_label: localized badge (Correto / Quase lá / Precisa de revisão, etc.)
      - status_class: for CSS
      - expl: localized, concrete explanation (even when correct).
    """
    c = oa_client()

    packed = []
    for i, it in enumerate(gold_items, start=1):
        packed.append({
            "index": i,
            "prompt": it["en"],        # PT flow: Portuguese sentence; ES flow: English sentence
            "gold": it["target"],
            "user": user_answers.get(i, "") or "",
            "topics": it.get("topics", []),
        })

    if lang_code == "pt":
        explain_instr = (
            "Explique EM PORTUGUÊS, claro e curto. "
            "Sempre diga POR QUE está correto ou incorreto. "
            "- Se correto: confirme que transmite o sentido certo e cite 1 ponto gramatical positivo. "
            "- Se quase: diga o que falta/muda e como ajustar. "
            "- Se precisa de revisão: aponte o erro principal (tempo, ordem, escolha de palavra, etc.) e dê 1 dica. "
            "Use 1–2 frases."
        )
    else:
        explain_instr = (
            "Explain IN ENGLISH, clearly and concisely. "
            "Always say WHY it's correct or incorrect. "
            "- If correct: confirm meaning is right and highlight one positive grammar/usage point. "
            "- If almost: say exactly what's off and how to fix. "
            "- If needs work: point out the main error and give one fix tip. "
            "Use 1–2 sentences."
        )

    grader_prompt = {
        "instructions": (
            "Grade each translation. Use status one of: Correct / Almost / Needs work. "
            "Return strict JSON array with {index, status, expl}. "
            f"For 'expl', {explain_instr}"
        ),
        "items": packed,
    }

    resp = c.chat.completions.create(
        model=model_name(),
        temperature=0,
        messages=[
            {"role":"system","content":"Return only valid JSON. No markdown, no commentary."},
            {"role":"user","content": json.dumps(grader_prompt, ensure_ascii=False)},
        ],
    )
    text = (resp.choices[0].message.content or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1].strip()

    try:
        judgments = json.loads(text)
    except Exception:
        m = re.search(r"\[.*\]", text, re.DOTALL)
        judgments = json.loads(m.group(0)) if m else []
    _debug(f"Received {len(judgments)} judgments")

    rows = []
    for j in judgments:
        idx = int(j.get("index", 0))
        if not (1 <= idx <= len(packed)):
            continue
        base = packed[idx - 1]
        status = j.get("status", "Needs work")
        expl = j.get("expl", "").strip()
        status_label, status_class = _status_label_and_class(status, lang_code)
        rows.append({
            "index": idx,
            "en": base["prompt"],
            "gold": base["gold"],
            "user": base["user"],
            "status": status,
            "status_label": status_label,
            "status_class": status_class,
            "expl": expl,
        })

    # Update history
    hist = load_history(lang_code)
    session = {"date": datetime.now(timezone.utc).isoformat(), "items": []}
    for r, base in zip(rows, packed):
        session["items"].append({
            "en": base["prompt"],
            "gold": base["gold"],
            "user": r["user"],
            "status": r["status"],
            "topics": base.get("topics", []),
        })
    hist["sessions"].append(session)
    save_history(lang_code, hist)
    git_commit_if_ci(f"Update {lang_code} history {datetime.now(timezone.utc).strftime('%Y-%m-%d')}")
    _debug("History updated with grading results")

    # Plain text fallback (localized)
    labels = ui_labels(lang_code)
    if lang_code == "pt":
        head = "Feedback:"
        you  = "Sua resposta"
        corr = "Correta"
    elif lang_code == "es":
        head = "Comentarios:"
        you  = "Tu respuesta"
        corr = "Correcta"
    else:
        head = "Feedback:"
        you  = "Yours"
        corr = "Gold"

    plain_lines = [head, ""]
    for r in rows:
        plain_lines.append(f"{r['index']}. {r['en']}")
        plain_lines.append(f"   {you}: {r['user']}")
        plain_lines.append(f"   {corr}: {r['gold']}")
        plain_lines.append(f"   {r['status_label']}: {r['expl']}")
        plain_lines.append("")
    plain_text = "\n".join(plain_lines)

    return plain_text, rows

# ---------------- Core renderers for QUIZ & ANSWERS ----------------
def render_quiz_email_core(lang: str, date_str: str, theme: str, lesson_html: str, items: List[Dict]) -> Tuple[str, str, str]:
    """
    Build subject + HTML + plain for the quiz email using the shared template.
    """
    L = ui_labels(lang)
    subject = f"{L['quiz_title']} — {date_str}"
    html = render_template("templates/email_quiz.html", {
        "subject": subject,
        "header_title": subject,
        "theme_label": L["theme_label"],
        "theme": theme,
        "mini_lesson_label": L["mini_lesson_label"],
        "mini_lesson_html": lesson_html,
        "translate_into_label": L["translate_into_label"],
        "reply_hint": L["reply_hint"],
        "footer_text": L["quiz_footer"],
        "target_lang_name": "Inglês" if lang=="pt" else "Spanish",
        "items": [{"en": it["en"]} for it in items],
    })
    # plain
    plain = f"""{subject}

{L['mini_lesson_label']} (veja o HTML).""" if lang=="pt" else f"""{subject}

{L['mini_lesson_label']} (see HTML)."""
    plain += "\n\n" + L["translate_into_label"] + "\n" + \
             "\n".join([f"{i+1}. {it['en']}" for i, it in enumerate(items)]) + \
             ("\n\n" + L["reply_hint"])
    return subject, html, plain

def render_answers_email_core(lang: str, date_str: str, rows: List[Dict]) -> Tuple[str, str, str]:
    """
    Build subject + HTML + plain for the answers email using the shared template.
    Ensures rows contain status_label/status_class and fills the summary box.
    """
    L = ui_labels(lang)

    # Ensure labels/classes + build summary
    fixed_rows = []
    for r in rows:
        lbl = r.get("status_label")
        cls = r.get("status_class")
        if not (lbl and cls):
            lbl, cls = _status_label_and_class(r.get("status",""), lang)
        fixed_rows.append({**r, "status_label": lbl, "status_class": cls})

    c = Counter([r["status_class"] for r in fixed_rows])
    summary_tag = (
        f"Corretas: {c.get('ok',0)} · Quase: {c.get('almost',0)} · Revisão: {c.get('needs',0)}" if lang=="pt" else
        f"Correctas: {c.get('ok',0)} · Casi: {c.get('almost',0)} · Revisión: {c.get('needs',0)}" if lang=="es" else
        f"Correct: {c.get('ok',0)} · Almost: {c.get('almost',0)} · Review: {c.get('needs',0)}"
    )

    subject = f"{L['answers_title']} — {date_str}"
    html = render_template("templates/email_answers.html", {
        "subject": subject,
        "header_title": subject,
        "summary_label": L["summary_label"],
        "summary_tag": summary_tag,
        "feedback_label": L["feedback_label"],
        "growth_hint": L["growth_hint"],
        "results_label": L["results_label"],
        "your_answer_label": L["your_answer_label"],
        "correct_label": L["correct_label"],
        "next_steps_hint": L["next_steps_hint"],
        "footer_text": L["answers_footer"],
        "rows": fixed_rows,
    })

    # plain
    if lang=="pt":
        head="Feedback (resumo): "; you="Sua"; corr="Correta"
    elif lang=="es":
        head="Resumen: "; you="Tu"; corr="Correcta"
    else:
        head="Summary: "; you="Your"; corr="Correct"

    plain = head + summary_tag + "\n\n" + "\n".join(
        [f"{r['index']}. {r['en']}\n   {you}: {r['user']}\n   {corr}: {r['gold']}\n   {r['status_label']}: {r['expl']}\n"
         for r in fixed_rows]
    )
    return subject, html, plain