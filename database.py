"""
database.py — Mood Predictor
Handles all SQLite database operations.
"""

import sqlite3
import datetime

DB_PATH = "mood_history.db"


def init_db():
    """Create the predictions table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  TEXT    NOT NULL,
            user_text  TEXT    NOT NULL,
            language   TEXT    NOT NULL,
            mood       TEXT    NOT NULL,
            confidence REAL    NOT NULL,
            severity   TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_prediction(user_text: str, language: str, mood: str, confidence: float, severity: str):
    """Insert one prediction record into the database."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO predictions
           (timestamp, user_text, language, mood, confidence, severity)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            user_text,
            language,
            mood,
            round(confidence, 4),
            severity,
        ),
    )
    conn.commit()
    conn.close()


def load_predictions(limit: int = 50):
    """Return the most recent predictions as a list of dicts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def clear_predictions():
    """Delete all records from the predictions table."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
