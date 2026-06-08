from app.repositories.db import get_connection


def insert_detection(
    image_id: int,
    label: str,
    confidence: float,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    crop_path: str | None = None,
) -> None:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO detections (
            image_id,
            label,
            confidence,
            x1,
            y1,
            x2,
            y2,
            crop_path
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            image_id,
            label,
            confidence,
            x1,
            y1,
            x2,
            y2,
            crop_path,
        ),
    )

    conn.commit()
    conn.close()


def get_detections_by_image(image_id: int):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT label, confidence, x1, y1, x2, y2, crop_path
        FROM detections
        WHERE image_id = ?
        ORDER BY id ASC
        """,
        (image_id,),
    )

    rows = cursor.fetchall()
    conn.close()
    return rows


def get_all_detected_labels():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT DISTINCT label
        FROM detections
        ORDER BY label ASC
        """
    )

    rows = cursor.fetchall()
    conn.close()

    return [row[0] for row in rows]


def get_all_crops(label: str | None = None):
    conn = get_connection()
    cursor = conn.cursor()

    if label:
        cursor.execute(
            """
            SELECT image_id, label, confidence, crop_path
            FROM detections
            WHERE crop_path IS NOT NULL
              AND label = ?
            ORDER BY id DESC
            """,
            (label,),
        )
    else:
        cursor.execute(
            """
            SELECT image_id, label, confidence, crop_path
            FROM detections
            WHERE crop_path IS NOT NULL
            ORDER BY id DESC
            """
        )

    rows = cursor.fetchall()
    conn.close()
    return rows


def get_detection_stats():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM detections")
    total_detections = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT label, COUNT(*) as total
        FROM detections
        GROUP BY label
        ORDER BY total DESC
        """
    )
    by_label = cursor.fetchall()

    conn.close()

    return {
        "total_detections": total_detections,
        "by_label": by_label,
    }