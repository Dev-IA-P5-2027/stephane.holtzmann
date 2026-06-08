import io
import zipfile
import streamlit as st
from pathlib import Path

from app.utils.paths import ensure_directories
from app.repositories.db import init_db
from app.repositories.detection_repository import (
    get_all_crops,
    get_all_detected_labels,
)


def bootstrap_page():
    ensure_directories()
    init_db()


def build_crops_zip(crops):
    """
    Construit un zip en mémoire à partir des crops filtrés.
    Chaque crop est rangé dans un sous-dossier par classe.
    """
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for image_id, class_name, confidence, crop_path in crops:
            if not crop_path:
                continue

            crop_file = Path(crop_path)
            if not crop_file.exists():
                continue

            # nom propre dans le zip
            arcname = f"{class_name}/{crop_file.name}"
            zip_file.write(crop_file, arcname=arcname)

    zip_buffer.seek(0)
    return zip_buffer


bootstrap_page()

st.title("Dataset - Crops")

classes = get_all_detected_labels()

selected_class = st.selectbox(
    "Filtrer par classe",
    ["Toutes"] + classes
)

thumb_size = st.slider(
    "Taille des miniatures",
    min_value=80,
    max_value=250,
    value=160,
    step=10
)

num_cols = st.slider(
    "Nombre de colonnes",
    min_value=2,
    max_value=6,
    value=4,
    step=1
)

if selected_class == "Toutes":
    crops = get_all_crops()
    zip_filename = "dataset_crops_all.zip"
else:
    crops = get_all_crops(selected_class)
    zip_filename = f"dataset_crops_{selected_class}.zip"

st.write(f"**{len(crops)} crop(s) trouvé(s)**")

if not crops:
    st.info("Aucun crop trouvé pour ce filtre.")
    st.stop()

# Bouton export ZIP
zip_buffer = build_crops_zip(crops)

st.download_button(
    label="📦 Exporter les crops en ZIP",
    data=zip_buffer,
    file_name=zip_filename,
    mime="application/zip",
)

st.markdown("---")

cols = st.columns(num_cols)

for idx, (image_id, class_name, confidence, crop_path) in enumerate(crops):
    col = cols[idx % num_cols]

    with col:
        if crop_path and Path(crop_path).exists():
            st.image(crop_path, width=thumb_size)
        else:
            st.warning("Crop introuvable")

        st.caption(f"Classe : {class_name}")
        st.caption(f"Confiance : {confidence:.2f}")
        st.caption(f"Image ID : {image_id}")