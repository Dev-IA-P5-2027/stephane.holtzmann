import streamlit as st
from PIL import Image

from app.services.image_service import process_uploaded_image

st.title("Upload & Détection")

uploaded_file = st.file_uploader(
    "Choisis une image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Image uploadée")
    st.image(image, use_container_width=True)

    if st.button("Sauvegarder et lancer la détection"):
        with st.spinner("Traitement en cours..."):
            result = process_uploaded_image(uploaded_file)

        st.success(f"Image sauvegardée : {result['filename']}")
        st.info(f"ID en base : {result['image_id']}")

        col1, col2 = st.columns(2)

        with col1:
            st.write("### Image originale")
            st.image(result["original_path"], use_container_width=True)

        with col2:
            st.write("### Image annotée")
            st.image(result["annotated_path"], use_container_width=True)

        st.write("### Résumé")
        st.write(f"**Nom :** {result['filename']}")
        st.write(f"**Modèle :** {result['model_name']}")
        st.write(f"**Nombre de détections :** {result['total_detections']}")

        st.write("### Objets détectés")

        if result["detections"]:
            for det in result["detections"]:
                col_crop, col_info = st.columns([1, 3])

                with col_crop:
                    if det["crop_path"]:
                        st.image(det["crop_path"], use_container_width=True)
                    else:
                        st.warning("Crop introuvable")

                with col_info:
                    st.write(f"**Classe :** {det['label']}")
                    st.write(f"**Confiance :** {det['confidence']:.2f}")
                    st.write(
                        f"**BBox :** "
                        f"({det['x1']}, {det['y1']}, {det['x2']}, {det['y2']})"
                    )
                    st.markdown("---")
        else:
            st.info("Aucune détection trouvée.")