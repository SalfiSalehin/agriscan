"""A tiny optional log of predictions and user feedback (standard-library sqlite3).

Stored: time, crop mode, status, top answer, confidence, the language and the model's id. Never
stored: the photo, the filename, the IP address or anything that identifies the person.
"""

import sqlite3
import threading
import time
from pathlib import Path

VERDICTS = ("correct", "wrong")

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at INTEGER NOT NULL,
    status TEXT NOT NULL,
    diagnosis TEXT,
    confidence REAL NOT NULL,
    crop_mode TEXT NOT NULL,
    crop TEXT,
    lang TEXT NOT NULL,
    model_sha TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL REFERENCES predictions(id),
    created_at INTEGER NOT NULL,
    verdict TEXT NOT NULL CHECK (verdict IN ('correct', 'wrong')),
    true_class TEXT
);
"""


class Store:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(SCHEMA)

    def record(self, result: dict, lang: str) -> int:
        with self._lock:
            cursor = self._db.execute(
                "INSERT INTO predictions "
                "(created_at, status, diagnosis, confidence, crop_mode, crop, lang, model_sha) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    int(time.time()),
                    result["status"],
                    result["diagnosis"],
                    float(result["confidence"]),
                    result["crop_mode"],
                    result["crop"],
                    lang,
                    result["model"]["sha256"],
                ),
            )
            self._db.commit()
            return int(cursor.lastrowid)

    def add_feedback(self, prediction_id: int, verdict: str, true_class: str | None) -> bool:
        """False if there is no such prediction (or feedback was already given for it)."""
        if verdict not in VERDICTS:
            raise ValueError("verdict must be 'correct' or 'wrong'")
        with self._lock:
            exists = self._db.execute(
                "SELECT 1 FROM predictions WHERE id = ?", (prediction_id,)
            ).fetchone()
            given = self._db.execute(
                "SELECT 1 FROM feedback WHERE prediction_id = ?", (prediction_id,)
            ).fetchone()
            if not exists or given:
                return False
            self._db.execute(
                "INSERT INTO feedback (prediction_id, created_at, verdict, true_class) "
                "VALUES (?, ?, ?, ?)",
                (prediction_id, int(time.time()), verdict, true_class),
            )
            self._db.commit()
            return True

    def counts(self) -> dict:
        with self._lock:
            predictions = self._db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            feedback = self._db.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        return {"predictions": predictions, "feedback": feedback}

    def close(self) -> None:
        with self._lock:
            self._db.close()
