from datetime import datetime
from app.repositories.db import get_connection


def insert_image(
    filename: str,
    original_path: str,
    model_name: str = "none",
    confidence_threshold: float = 0.0,
    total_detections: int = 0,
    annotated_path: str | None = None,
) -> int:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO images (
            filename,
            original_path,
            annotated_path,
            uploaded_at,
            model_name,
            confidence_threshold,
            total_detections
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            filename,
            original_path,
            annotated_path,
            datetime.now().isoformat(),
            model_name,
            confidence_threshold,
            total_detections,
        ),
    )

    image_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return image_id


def get_all_images():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, filename, original_path, uploaded_at,
               model_name, total_detections, annotated_path
        FROM images
        ORDER BY uploaded_at DESC
        """
    )

    rows = cursor.fetchall()
    conn.close()
    return rows


def get_image_by_id(image_id: int):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, filename, original_path, uploaded_at,
               model_name, total_detections, annotated_path
        FROM images
        WHERE id = ?
        """,
        (image_id,),
    )

    row = cursor.fetchone()
    conn.close()
    return row


def get_images_by_min_detections(min_detections: int):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, filename, original_path, uploaded_at,
               model_name, total_detections, annotated_path
        FROM images
        WHERE total_detections >= ?
        ORDER BY uploaded_at DESC
        """,
        (min_detections,),
    )

    rows = cursor.fetchall()
    conn.close()
    return rows


def get_images_by_model(model_name: str):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, filename, original_path, uploaded_at,
               model_name, total_detections, annotated_path
        FROM images
        WHERE model_name = ?
        ORDER BY uploaded_at DESC
        """,
        (model_name,),
    )

    rows = cursor.fetchall()
    conn.close()
    return rows


def get_images_by_class(class_name: str):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT DISTINCT i.id, i.filename, i.original_path, i.uploaded_at,
                        i.model_name, i.total_detections, i.annotated_path
        FROM images i
        JOIN detections d ON i.id = d.image_id
        WHERE d.class_name = ?
        ORDER BY i.uploaded_at DESC
        """,
        (class_name,),
    )

    rows = cursor.fetchall()
    conn.close()
    return rows