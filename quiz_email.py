#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, uuid
from datetime import datetime, timezone
from typing import Tuple

from core_quiz import (
    send_email,
    generate_items,
    grade_answers,
    find_latest_reply,
    render_quiz_email_core,
    render_answers_email_core,
    has_processed_reply,
    record_processed_reply
)

# ---------------- Message-ID -----------------
def gen_message_id(sender_email: str) -> str:
    domain = sender_email.split("@")[-1]
    return f"<{uuid.uuid4().hex}.{int(datetime.now(timezone.utc).timestamp())}@{domain}>"

# ---------------- Helpers --------------------
def cache_paths(lang_code: str, date_str: str) -> Tuple[str, str]:
    os.makedirs("data", exist_ok=True)
    quiz_path = f"data/{lang_code}_quiz_{date_str}.json"
    meta_path = f"data/{lang_code}_meta_{date_str}.json"
    return quiz_path, meta_path

def load_or_make_items(lang_code: str, target_lang_name: str, date_str: str):
    quiz_path, _ = cache_paths(lang_code, date_str)
    if os.path.exists(quiz_path):
        with open(quiz_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
            return obj["theme"], obj["lesson_html"], obj["items"]
    theme, lesson_html, items = generate_items(lang_code, target_lang_name, {}, beginner=True)
    with open(quiz_path, "w", encoding="utf-8") as f:
        json.dump({"theme": theme, "lesson_html": lesson_html, "items": items}, f, ensure_ascii=False, indent=2)
    return theme, lesson_html, items

def parse_numbered_answers(text: str):
    import re
    answers = {}
    for line in (text or "").splitlines():
        m = re.match(r'^\s*(\d{1,2})[\.\)\-:]\s*(.+)$', line.strip())
        if m:
            answers[int(m.group(1))] = m.group(2).strip()
    return answers

def target_lang_name_for(lang_code: str) -> str:
    # Names used in UI/body copy
    return {"es": "Spanish", "pt": "Ingles"}.get(lang_code, "Spanish")

# ---------------- Main -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["quiz", "answers"], required=True)
    ap.add_argument("--lang", choices=["es", "pt"], default="es", help="Which list to send / grade for")
    args = ap.parse_args()

    lang_code = args.lang
    target_lang_name = target_lang_name_for(lang_code)

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")

    quiz_path, meta_path = cache_paths(lang_code, date_str)

    if args.mode == "quiz":
        # 1) Build content (theme, mini-lesson HTML, 10 items)
        theme, lesson_html, items = load_or_make_items(lang_code, target_lang_name, date_str)

        # 2) Render localized email (labels + summary handled in core)
        subject, html, plain = render_quiz_email_core(
            lang=lang_code,         # <-- changed from ui_lang
            date_str=date_str,
            theme=theme,
            lesson_html=lesson_html,
            items=items,
        )

        # 3) Thread-id + send
        quiz_msg_id = gen_message_id(os.environ["SMTP_USER"])
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"quiz_message_id": quiz_msg_id, "subject": subject}, f, ensure_ascii=False, indent=2)

        send_email(
            subject,
            html,
            plain,
            message_id=quiz_msg_id,
            sender_default=f"{target_lang_name} para mi Amor ❤️",
            language=lang_code,
        )



    else:
        # ANSWERS
        quiz_msg_id = None
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                quiz_msg_id = meta.get("quiz_message_id")

        reply_mid, reply_body = (None, None)
        if quiz_msg_id:
            reply_mid, reply_body = find_latest_reply(quiz_msg_id)

        if reply_mid and reply_body:
            # 1) Load gold items
            _, _, items = load_or_make_items(lang_code, target_lang_name, date_str)
            user_answers = parse_numbered_answers(reply_body)

            # 2) Grade (rows contain status_label/status_class/expl always)
            plain_feedback, rows = grade_answers(lang_code, target_lang_name, items, user_answers)

            # 3) Render pretty HTML (header + summary/labels)
            subject, html, plain_fallback = render_answers_email_core(
                lang=lang_code,      # <-- changed from ui_lang
                date_str=date_str,
                rows=rows,
            )

            if reply_mid and reply_body:
                # Dedupe: only grade/send once per unique (quiz, reply) pair
                dedupe_key = f"{quiz_msg_id}|{reply_mid}" if quiz_msg_id else f"|{reply_mid}"
                if has_processed_reply(lang_code, dedupe_key):
                    print(f"Reply already graded; skipping. key={dedupe_key}")
                    return

            refs = f"{quiz_msg_id} {reply_mid}".strip() if quiz_msg_id else reply_mid
            send_email(
                subject,
                html,
                plain_feedback or plain_fallback,
                in_reply_to=reply_mid,
                references=refs,
                sender_default=f"{target_lang_name} para mi Amor ❤️",
                language=lang_code,
            )
            record_processed_reply(lang_code, dedupe_key)
        else:
            print("No reply found yet.")

if __name__ == "__main__":
    main()