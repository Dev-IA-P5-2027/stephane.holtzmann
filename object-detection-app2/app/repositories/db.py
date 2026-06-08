import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "app.db"


def get_connection() -> sqlite3.Connection:
    """Retourne une connexion SQLite."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Crée les tables si elles n'existent pas."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        original_path TEXT NOT NULL,
        annotated_path TEXT,
        uploaded_at TEXT NOT NULL,
        model_name TEXT NOT NULL,
        confidence_threshold REAL NOT NULL,
        total_detections INTEGER NOT NULL
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_id INTEGER NOT NULL,
        label TEXT NOT NULL,
        confidence REAL NOT NULL,
        x1 INTEGER NOT NULL,
        y1 INTEGER NOT NULL,
        x2 INTEGER NOT NULL,
        y2 INTEGER NOT NULL,
        crop_path TEXT,
        FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
    )
    """)

    conn.commit()
    conn.close()
    