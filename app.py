"""
AI Study Assistant - Flask Backend
-----------------------------------
Uploads notes/PDFs and uses a local Ollama model to generate summaries,
practice questions, MCQs, flashcards, evaluate answers, and power a
"chat with my notes" assistant. All data is stored in a local SQLite DB.

No API key needed -- runs entirely on local AI via Ollama.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from typing import Any

import requests
from flask import Flask, g, jsonify, render_template, request
from pypdf import PdfReader

# --------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "study_assistant.db")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

# Generation defaults
DEFAULT_QUESTION_COUNT = 88
DEFAULT_FLASHCARD_COUNT = 6
DEFAULT_MCQ_COUNT = 5

# Context / token limits
NOTE_CONTEXT_CHARS = 12000        # max note chars sent to the model
EVAL_CONTEXT_CHARS = 8000         # smaller window for answer evaluation
OLLAMA_TIMEOUT_SECONDS = 300

# Sampling -- high enough that output is varied, not repetitive.
MODEL_TEMPERATURE = 0.9
MODEL_TOP_P = 0.9

app = Flask(__name__)


# --------------------------------------------------------------------- #
# Database
# --------------------------------------------------------------------- #
SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    note_id INTEGER NOT NULL,
    summary TEXT NOT NULL,
    topics TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(note_id) REFERENCES notes(id)
);
CREATE TABLE IF NOT EXISTS mcqs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    note_id INTEGER NOT NULL,
    question TEXT NOT NULL,
    option_a TEXT, option_b TEXT, option_c TEXT, option_d TEXT,
    correct_option TEXT NOT NULL,
    explanation TEXT,
    FOREIGN KEY(note_id) REFERENCES notes(id)
);
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    note_id INTEGER NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(note_id) REFERENCES notes(id)
);
CREATE TABLE IF NOT EXISTS quiz_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    note_id INTEGER NOT NULL,
    score INTEGER NOT NULL,
    total INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(note_id) REFERENCES notes(id)
);
"""


def get_db() -> sqlite3.Connection:
    """Return a per-request SQLite connection, opening one if needed."""
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception: Exception | None = None) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------------- #
# Ollama client
# --------------------------------------------------------------------- #
class OllamaError(Exception):
    """Raised when the local Ollama service is unreachable or errors."""


def ask_llm(
    prompt: str,
    system: str | None = None,
    max_tokens: int = 1500,
    json_mode: bool = False,
) -> str:
    """Call the local Ollama chat endpoint and return the model's reply."""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": MODEL_TEMPERATURE,
            "top_p": MODEL_TOP_P,
        },
    }
    if json_mode:
        payload["format"] = "json"

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT_SECONDS
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise OllamaError(
            "Could not connect to Ollama. Is it running? (`ollama serve`)"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise OllamaError(f"Ollama error: {exc}") from exc

    return resp.json().get("message", {}).get("content", "")


# --------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------- #
def parse_json_response(text: str) -> Any:
    """Parse a JSON array or object from a model response, returning it raw."""
    text = text.strip().replace("```json", "").replace("```", "").strip()

    start = text.find("[")
    if start == -1:
        start = text.find("{")
    end = text.rfind("]")
    if end == -1:
        end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("AI did not return valid JSON.") from exc


def as_list(parsed: Any) -> list:
    """Normalize a parsed JSON value into a list of items."""
    if isinstance(parsed, dict):
        # Some models wrap a list inside an object key; unwrap it.
        for value in parsed.values():
            if isinstance(value, list):
                return value
        return [parsed]
    if isinstance(parsed, list):
        return parsed
    return [parsed]


# --------------------------------------------------------------------- #
# Request helpers
# --------------------------------------------------------------------- #
def get_note_or_404(note_id: int):
    """Return the note row, or a Flask 404 response tuple."""
    note = get_db().execute(
        "SELECT * FROM notes WHERE id = ?", (note_id,)
    ).fetchone()
    if note is None:
        return None, (jsonify({"error": "Note not found"}), 404)
    return note, None


def request_count(default: int) -> int:
    """Read an integer 'count' from the JSON body, falling back to default."""
    if not request.is_json:
        return default
    try:
        return int(request.get_json().get("count", default))
    except (TypeError, ValueError):
        return default


def clean_pdf_text(text: str) -> str:
    """Replace HTML-entity artifacts that pypdf sometimes emits for brackets."""
    for bad, good in (("&#40;", "("), ("&#41;", ")"), ("&40;", "("), ("&41;", ")")):
        text = text.replace(bad, good)
    return text


# --------------------------------------------------------------------- #
# Prompt templates
# --------------------------------------------------------------------- #
SUMMARY_PROMPT = """\
Here are the student's study notes:
{content}
Write a clear, well-structured summary of this (in bullet points).
After that, below the "IMPORTANT TOPICS:" heading, provide a list of 5-8 important topics/keywords (comma separated single line).
Format:
SUMMARY:
<summary here>
IMPORTANT TOPICS:
<topic1, topic2, topic3, ...>"""

QUESTIONS_PROMPT = """\
Create {count} exam-style practice questions from these notes (mix of short & long answer type).
Return only a JSON array, no other text. Format:
[{{"question": "...", "type": "short/long", "answer_hint": "..."}}]
NOTES:
{content}"""

FLASHCARDS_SYSTEM = (
    "You are a flashcard generator. You must output a JSON array "
    "containing distinct flashcards."
)
FLASHCARDS_PROMPT = """\
Create exactly {count} flashcards from the notes below.
Each flashcard MUST cover a DIFFERENT concept or term from the notes. Do not repeat the same concept twice.

Return ONLY a JSON array of {count} objects. Do not return a single JSON object.

Format:
[
  {{"term": "Concept 1", "definition": "Explanation 1"}},
  {{"term": "Concept 2", "definition": "Explanation 2"}}
]

NOTES:
{content}"""

MCQ_PROMPT = """\
Create {count} unique multiple-choice questions from these notes.
Return ONLY a valid JSON array. Do NOT include any extra text or code blocks.

Each object MUST have this exact format:
{{"question": "...", "option_a": "...", "option_b": "...", "option_c": "...", "option_d": "...", "correct_option": "a", "explanation": "..."}}

The options (option_a, option_b, etc.) must contain ONLY the answer text. Do NOT append random numbers, IDs, semicolons, or HTML entities (like '40;') to the options.

NOTES:
{content}"""

EVAL_PROMPT = """\
You are a strict but encouraging teacher. Based ONLY on the provided notes, evaluate the student's answer.
    Question: {question}
    Student's Answer: {answer}
    Notes:
    {content}
    Return ONLY a JSON object with this exact format:
    {{"is_correct": true/false, "feedback": "A brief explanation of why it is correct or what is missing."}}"""

CHAT_SYSTEM = "You are a helpful study assistant. Answer only based on the provided notes content."
CHAT_PROMPT = "NOTES:\n{content}\n\nSTUDENT'S QUESTION: {question}"


# --------------------------------------------------------------------- #
# Routes -- pages, stats, notes
# --------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stats", methods=["GET"])
def get_stats():
    db = get_db()
    total_notes = db.execute("SELECT COUNT(*) AS count FROM notes").fetchone()["count"]
    total_quizzes = db.execute(
        "SELECT COUNT(*) AS count FROM quiz_attempts"
    ).fetchone()["count"]
    total_chats = db.execute(
        "SELECT COUNT(*) AS count FROM chat_history"
    ).fetchone()["count"]
    return jsonify(
        {
            "total_notes": total_notes,
            "total_quizzes": total_quizzes,
            "total_chats": total_chats,
        }
    )


@app.route("/api/notes", methods=["GET"])
def list_notes():
    db = get_db()
    rows = db.execute(
        "SELECT id, title, created_at FROM notes ORDER BY id DESC"
    ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/notes/<int:note_id>", methods=["GET"])
def get_note(note_id: int):
    note, error = get_note_or_404(note_id)
    if error:
        return error
    return jsonify(dict(note))


@app.route("/api/notes/<int:note_id>/delete", methods=["POST"])
def delete_note(note_id: int):
    db = get_db()
    note = db.execute(
        "SELECT id FROM notes WHERE id = ?", (note_id,)
    ).fetchone()
    if note is None:
        return jsonify({"error": "Note not found"}), 404

    # Cascade-delete everything attached to this note.
    for table in ("summaries", "mcqs", "chat_history", "quiz_attempts"):
        db.execute(f"DELETE FROM {table} WHERE note_id = ?", (note_id,))
    db.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    db.commit()
    return jsonify({"success": True})


@app.route("/api/upload", methods=["POST"])
def upload_note():
    title = request.form.get("title", "").strip()
    content = request.form.get("text", "").strip()
    file = request.files.get("file")

    if file and file.filename:
        filename = file.filename
        filepath = os.path.join(UPLOAD_DIR, filename)
        file.save(filepath)

        if filename.lower().endswith(".pdf"):
            reader = PdfReader(filepath)
            content = "\n".join(page.extract_text() or "" for page in reader.pages)
            content = clean_pdf_text(content)
        else:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
                content = fh.read()

        if not title:
            title = os.path.splitext(filename)[0]

    if not content:
        return jsonify(
            {"error": "No content found. Please upload a file or paste text."}
        ), 400
    if not title:
        title = f"Note {datetime.now().strftime('%d-%b %H:%M')}"

    db = get_db()
    cur = db.execute(
        "INSERT INTO notes (title, content, created_at) VALUES (?, ?, ?)",
        (title, content, datetime.now().isoformat()),
    )
    db.commit()
    return jsonify({"id": cur.lastrowid, "title": title})


# --------------------------------------------------------------------- #
# Routes -- AI generation
# --------------------------------------------------------------------- #
@app.route("/api/notes/<int:note_id>/summary", methods=["POST"])
def generate_summary(note_id: int):
    note, error = get_note_or_404(note_id)
    if error:
        return error

    prompt = SUMMARY_PROMPT.format(content=note["content"][:NOTE_CONTEXT_CHARS])
    try:
        result = ask_llm(prompt, max_tokens=1200)
    except OllamaError as exc:
        return jsonify({"error": str(exc)}), 503

    summary_part, topics_part = result, ""
    if "IMPORTANT TOPICS:" in result:
        parts = result.split("IMPORTANT TOPICS:")
        summary_part = parts[0].replace("SUMMARY:", "").strip()
        topics_part = parts[1].strip()

    db = get_db()
    db.execute(
        "INSERT INTO summaries (note_id, summary, topics, created_at) VALUES (?, ?, ?, ?)",
        (note_id, summary_part, topics_part, datetime.now().isoformat()),
    )
    db.commit()
    return jsonify({"summary": summary_part, "topics": topics_part})


@app.route("/api/notes/<int:note_id>/questions", methods=["POST"])
def generate_questions(note_id: int):
    note, error = get_note_or_404(note_id)
    if error:
        return error

    count = request_count(DEFAULT_QUESTION_COUNT)
    prompt = QUESTIONS_PROMPT.format(
        count=count, content=note["content"][:NOTE_CONTEXT_CHARS]
    )
    try:
        result = ask_llm(prompt, max_tokens=1500, json_mode=True)
        questions = as_list(parse_json_response(result))
    except (OllamaError, ValueError):
        return jsonify({"error": "Question generation failed. Please try again."}), 500
    return jsonify(questions)


@app.route("/api/notes/<int:note_id>/flashcards", methods=["POST"])
def generate_flashcards(note_id: int):
    note, error = get_note_or_404(note_id)
    if error:
        return error

    count = request_count(DEFAULT_FLASHCARD_COUNT)
    prompt = FLASHCARDS_PROMPT.format(
        count=count, content=note["content"][:NOTE_CONTEXT_CHARS]
    )
    try:
        result = ask_llm(
            prompt, system=FLASHCARDS_SYSTEM, max_tokens=1500, json_mode=True
        )
        flashcards = as_list(parse_json_response(result))
    except (OllamaError, ValueError) as exc:
        print(f"Error generating flashcards: {exc}")
        return jsonify(
            {"error": "Flashcard generation failed. Please try again."}
        ), 500
    return jsonify(flashcards)


@app.route("/api/notes/<int:note_id>/mcq", methods=["POST"])
def generate_mcq(note_id: int):
    note, error = get_note_or_404(note_id)
    if error:
        return error

    count = request_count(DEFAULT_MCQ_COUNT)
    prompt = MCQ_PROMPT.format(
        count=count, content=note["content"][:NOTE_CONTEXT_CHARS]
    )

    try:
        result = ask_llm(prompt, max_tokens=1800, json_mode=True)
        mcqs = as_list(parse_json_response(result))
    except (OllamaError, ValueError):
        return jsonify({"error": "MCQ generation failed. Please try again."}), 500

    db = get_db()
    db.execute("DELETE FROM mcqs WHERE note_id = ?", (note_id,))
    for m in mcqs:
        db.execute(
            "INSERT INTO mcqs "
            "(note_id, question, option_a, option_b, option_c, option_d, "
            "correct_option, explanation) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                note_id,
                m.get("question"),
                m.get("option_a"),
                m.get("option_b"),
                m.get("option_c"),
                m.get("option_d"),
                m.get("correct_option"),
                m.get("explanation", ""),
            ),
        )
    db.commit()
    rows = db.execute(
        "SELECT * FROM mcqs WHERE note_id = ?", (note_id,)
    ).fetchall()
    return jsonify([dict(r) for r in rows])


# --------------------------------------------------------------------- #
# Routes -- quiz & answer evaluation
# --------------------------------------------------------------------- #
@app.route("/api/notes/<int:note_id>/quiz/submit", methods=["POST"])
def submit_quiz(note_id: int):
    body = request.get_json(silent=True) or {}
    answers = body.get("answers", {})

    db = get_db()
    rows = db.execute(
        "SELECT * FROM mcqs WHERE note_id = ?", (note_id,)
    ).fetchall()

    review = []
    score = 0
    for row in rows:
        given = answers.get(str(row["id"]))
        is_correct = given is not None and given == row["correct_option"]
        if is_correct:
            score += 1
        review.append(
            {
                "question": row["question"],
                "your_answer": given,
                "correct_answer": row["correct_option"],
                "is_correct": is_correct,
                "explanation": row["explanation"],
            }
        )

    db.execute(
        "INSERT INTO quiz_attempts (note_id, score, total, created_at) VALUES (?, ?, ?, ?)",
        (note_id, score, len(rows), datetime.now().isoformat()),
    )
    db.commit()
    return jsonify({"score": score, "total": len(rows), "review": review})


@app.route("/api/notes/<int:note_id>/check_answer", methods=["POST"])
def check_answer(note_id: int):
    note, error = get_note_or_404(note_id)
    if error:
        return error

    data = request.get_json(silent=True) or {}
    if "question" not in data or "answer" not in data:
        return jsonify({"error": "Missing question or answer"}), 400

    prompt = EVAL_PROMPT.format(
        question=data["question"],
        answer=data["answer"],
        content=note["content"][:EVAL_CONTEXT_CHARS],
    )
    try:
        result = ask_llm(prompt, max_tokens=500, json_mode=True)
        evaluation = parse_json_response(result)
        if isinstance(evaluation, list):
            evaluation = evaluation[0] if evaluation else {}
    except (OllamaError, ValueError):
        return jsonify({"error": "Evaluation failed. Please try again."}), 500
    return jsonify(evaluation)


# --------------------------------------------------------------------- #
# Routes -- chatbot
# --------------------------------------------------------------------- #
@app.route("/api/notes/<int:note_id>/chat", methods=["POST"])
def chat_with_notes(note_id: int):
    note, error = get_note_or_404(note_id)
    if error:
        return error

    user_question = (request.get_json(silent=True) or {}).get("question", "").strip()
    if not user_question:
        return jsonify({"error": "Question is empty"}), 400

    prompt = CHAT_PROMPT.format(
        content=note["content"][:NOTE_CONTEXT_CHARS], question=user_question
    )
    try:
        answer = ask_llm(prompt, system=CHAT_SYSTEM, max_tokens=800)
    except OllamaError as exc:
        return jsonify({"error": str(exc)}), 503

    db = get_db()
    db.execute(
        "INSERT INTO chat_history (note_id, question, answer, created_at) VALUES (?, ?, ?, ?)",
        (note_id, user_question, answer, datetime.now().isoformat()),
    )
    db.commit()
    return jsonify({"answer": answer})


@app.route("/api/notes/<int:note_id>/chat/history", methods=["GET"])
def chat_history_route(note_id: int):
    db = get_db()
    rows = db.execute(
        "SELECT question, answer, created_at FROM chat_history "
        "WHERE note_id = ? ORDER BY id ASC",
        (note_id,),
    ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/notes/<int:note_id>/chat/clear", methods=["POST"])
def clear_chat_history(note_id: int):
    db = get_db()
    db.execute("DELETE FROM chat_history WHERE note_id = ?", (note_id,))
    db.commit()
    return jsonify({"success": True})


if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)