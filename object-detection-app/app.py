import streamlit as st
from PIL import Image
from services.detection import detect_objects

st.set_page_config(page_title="Détection d'objets", layout="wide")

st.title("Mini app de détection")
st.write("Upload d'image + détection d'objets avec YOLOv8 + crops + filtre + sélection")

uploaded_file = st.file_uploader(
    "Choisis une image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image source")
        st.image(image, use_container_width=True)

    with st.spinner("Détection en cours..."):
        detected_image, detections, crops = detect_objects(image)

    with col2:
        st.subheader("Image annotée")
        st.image(detected_image, use_container_width=True)

    st.subheader("Détections")
    if detections:
        st.dataframe(detections, use_container_width=True)
    else:
        st.warning("Aucun objet détecté.")

    # ----------------------------
    # Filtre par label
    # ----------------------------
    st.subheader("Filtrage des objets")

    labels = sorted(list(set(d["label"] for d in detections))) if detections else []
    selected_label = st.selectbox(
        "Filtrer par type d'objet",
        ["Tous"] + labels
    )

    filtered_crops = [
        crop for crop in crops
        if selected_label == "Tous" or crop["label"] == selected_label
    ]

    # ----------------------------
    # Affichage des crops
    # ----------------------------
    st.subheader("Crops des objets détectés")

    if filtered_crops:
        cols = st.columns(3)

        for idx, crop_data in enumerate(filtered_crops):
            with cols[idx % 3]:
                st.image(
                    crop_data["image"],
                    caption=(
                        f'{crop_data["label"]} '
                        f'({crop_data["confidence"]:.3f})'
                    ),
                    use_container_width=True
                )

                if st.button(
                    f'Sélectionner {crop_data["label"]} #{crop_data["id"]}',
                    key=f'select_{crop_data["id"]}_{idx}'
                ):
                    st.session_state["selected_crop"] = crop_data
    else:
        st.info("Aucun crop à afficher pour ce filtre.")

    # ----------------------------
    # Objet sélectionné
    # ----------------------------
    if "selected_crop" in st.session_state:
        st.subheader("Objet sélectionné")

        selected = st.session_state["selected_crop"]

        st.image(
            selected["image"],
            caption=(
                f'Objet sélectionné : {selected["label"]} '
                f'({selected["confidence"]:.3f})'
            ),
            width=400
        )

        st.write(
            {
                "id": selected["id"],
                "label": selected["label"],
                "confidence": selected["confidence"]
            }
        )

else:
    st.info("Aucune image chargée pour le moment")