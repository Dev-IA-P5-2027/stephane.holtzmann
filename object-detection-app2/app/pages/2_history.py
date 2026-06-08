import streamlit as st
from pathlib import Path

from app.utils.paths import ensure_directories
from app.repositories.db import init_db
from app.repositories.image_repository import get_all_images, get_images_by_class
from app.repositories.detection_repository import (
    get_detections_by_image,
    get_all_detected_labels,
)


def bootstrap_page():
    ensure_directories()
    init_db()


bootstrap_page()

st.title("Historique des images")

if "selected_image_id" not in st.session_state:
    st.session_state["selected_image_id"] = None

classes = get_all_detected_labels()

col1, col2, col3 = st.columns(3)

with col1:
    sort_order = st.selectbox(
        "Trier par date",
        ["Plus récent", "Plus ancien"]
    )

with col2:
    min_detections = st.slider(
        "Nombre minimum de détections",
        min_value=0,
        max_value=10,
        value=0
    )

with col3:
    selected_class = st.selectbox(
        "Filtrer par objet détecté",
        ["Toutes"] + classes
    )

if selected_class == "Toutes":
    images = get_all_images()
else:
    images = get_images_by_class(selected_class)

if sort_order == "Plus ancien":
    images = sorted(images, key=lambda x: x[3])
else:
    images = sorted(images, key=lambda x: x[3], reverse=True)

images = [img for img in images if img[5] >= min_detections]

if not images:
    st.info("Aucune image trouvée pour ce filtre.")
    st.stop()

st.write(f"**{len(images)} image(s) trouvée(s)**")

for img in images:
    image_id, filename, path, uploaded_at, model, total, annotated_path = img

    col1, col2, col3 = st.columns([1, 2, 3])

    with col1:
        if path and Path(path).exists():
            st.image(path, width=140)
        else:
            st.warning("Image introuvable")

    with col2:
        st.write(f"**ID :** {image_id}")
        st.write(f"**Nom :** {filename}")
        st.write(f"**Date :** {uploaded_at}")
        st.write(f"**Modèle :** {model}")
        st.write(f"**Détections :** {total}")

        if st.button("Voir détail", key=f"detail_{image_id}"):
            if st.session_state["selected_image_id"] == image_id:
                st.session_state["selected_image_id"] = None
            else:
                st.session_state["selected_image_id"] = image_id

    with col3:
        if st.session_state["selected_image_id"] == image_id:
            st.write("### Détail")

            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:
                st.write("**Image originale**")
                if path and Path(path).exists():
                    st.image(path, width=220)
                else:
                    st.warning("Image originale introuvable")

            with detail_col2:
                st.write("**Image annotée**")
                if annotated_path and Path(annotated_path).exists():
                    st.image(annotated_path, width=220)
                else:
                    st.warning("Image annotée introuvable")

            detections = get_detections_by_image(image_id)

            st.write("**Objets détectés**")

            if detections:
                for class_name, confidence, x_min, y_min, x_max, y_max, crop_path in detections:
                    det_col1, det_col2 = st.columns([1, 2])

                    with det_col1:
                        if crop_path and Path(crop_path).exists():
                            st.image(crop_path, width=100)
                        else:
                            st.warning("Crop introuvable")

                    with det_col2:
                        st.write(f"**Classe :** {class_name}")
                        st.write(f"**Confiance :** {confidence:.2f}")
                        st.write(f"**BBox :** ({x_min}, {y_min}, {x_max}, {y_max})")

                    st.markdown("---")
            else:
                st.info("Aucune détection enregistrée.")

    st.markdown("---")