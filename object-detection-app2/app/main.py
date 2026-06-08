import streamlit as st
from app.utils.paths import ensure_directories
from app.repositories.db import init_db

st.set_page_config(page_title="Object Detection App", layout="wide")


def bootstrap_app():
    if "bootstrapped" not in st.session_state:
        ensure_directories()
        init_db()
        st.session_state["bootstrapped"] = True
        st.toast("Initialisation terminée")


bootstrap_app()

st.title("Object Detection App")
st.write("Utilise le menu latéral pour naviguer entre les pages.")