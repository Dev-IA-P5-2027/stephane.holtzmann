# app.py
# Maquette UI "opérateurs" (Streamlit) — Projet classification fraude CB
# 3 pages : Login (avec création de compte) / Chargement CSV / Prédiction (liste + extraction ligne)
# NOTE : prédiction = placeholder (heuristique). Tu pourras brancher ton vrai modèle ensuite.

import io
import csv
import pandas as pd
import streamlit as st

# IMPORTANT: set_page_config doit être appelé UNE SEULE FOIS, tout en haut
st.set_page_config(page_title="Fraude CB • Maquette opérateurs", layout="wide")

# -----------------------------
# Config "dataset propre" attendu
# -----------------------------
EXPECTED_COLS = [
    "transactionId", "step", "type", "amount", "nameOrig",
    "oldbalanceOrg", "newbalanceOrig", "nameDest",
    "oldbalanceDest", "newbalanceDest"
]
OPTIONAL_COLS = ["isFraud"]
TYPE_ALLOWED = {"PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"}

# -----------------------------
# State
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "login"  # login | upload | predict
if "is_logged" not in st.session_state:
    st.session_state.is_logged = False
if "users" not in st.session_state:
    st.session_state.users = {"operateur@demo.fr": "password"}  # mock
if "df" not in st.session_state:
    st.session_state.df = None
if "pred_df" not in st.session_state:
    st.session_state.pred_df = None

# -----------------------------
# Helpers
# -----------------------------
def goto(page: str):
    st.session_state.page = page
    st.rerun()

def sniff_delimiter(text: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=[",", ";"])
        return dialect.delimiter
    except Exception:
        return ";" if text[:4096].count(";") > text[:4096].count(",") else ","

def to_float_fr(x):
    if pd.isna(x):
        return 0.0
    s = str(x).strip().replace(" ", "")
    if s == "":
        return 0.0
    if "," in s and "." in s:
        s = s.replace(",", "")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return 0.0

def validate_schema(df: pd.DataFrame):
    cols = list(df.columns)
    missing = [c for c in EXPECTED_COLS if c not in cols]
    extras = [c for c in cols if c not in EXPECTED_COLS + OPTIONAL_COLS]
    ok = len(missing) == 0
    return ok, missing, extras

def placeholder_predict_row(row: dict):
    t = str(row.get("type", "")).upper().strip()
    amount = to_float_fr(row.get("amount", 0))
    if t in {"TRANSFER", "CASH_OUT"} and amount >= 1000:
        return "FRAUDE", 0.87
    return "PAS FRAUDE", 0.12

def run_batch_predictions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    labels, probas = [], []

    for _, r in out.iterrows():
        lab, p = placeholder_predict_row(r.to_dict())
        labels.append(lab)
        probas.append(p)

    out["decision"] = labels
    out["fiabilite_pct"] = (pd.Series(probas) * 100).round(0).astype(int)

    cols_ui = ["transactionId", "step", "type", "amount", "decision", "fiabilite_pct"]
    cols_ui = [c for c in cols_ui if c in out.columns]
    out_ui = out[cols_ui].copy()

    out_ui = out_ui.sort_values("fiabilite_pct", ascending=False, kind="stable").reset_index(drop=True)
    return out_ui

def topbar():
    c1, c2, c3 = st.columns([2, 3, 2])
    with c1:
        st.markdown("### Détection fraude CB")
        st.caption("Maquette opérateurs • Streamlit")
    with c3:
        if st.session_state.is_logged:
            if st.button("Déconnexion"):
                st.session_state.is_logged = False
                st.session_state.df = None
                st.session_state.pred_df = None
                goto("login")

# -----------------------------
# Pages
# -----------------------------
def page_login():
    topbar()
    st.markdown("## Login")
    st.info("Interface opérateurs. Connexion + création de compte (maquette).")

    with st.container(border=True):
        st.markdown("### Connexion / Nouveau compte")

        tab_login, tab_create = st.tabs(["Connexion", "Créer un compte"])

        with tab_login:
            email = st.text_input("Email professionnel", placeholder="operateur@demo.fr")
            password = st.text_input("Mot de passe", type="password", placeholder="password")
            colA, colB = st.columns([1, 2])

            with colA:
                if st.button("Se connecter", type="primary"):
                    if email in st.session_state.users and st.session_state.users[email] == password:
                        st.session_state.is_logged = True
                        goto("upload")
                    else:
                        st.error("Identifiants invalides.")

            with colB:
                st.caption("Démo : operateur@demo.fr / password")

            st.caption("Mot de passe oublié (optionnel)")

        with tab_create:
            new_email = st.text_input("Email professionnel (nouveau compte)", key="new_email")
            new_pass = st.text_input("Mot de passe", type="password", key="new_pass")
            new_pass2 = st.text_input("Confirmer le mot de passe", type="password", key="new_pass2")

            if st.button("Créer le compte"):
                if not new_email or "@" not in new_email:
                    st.error("Email invalide.")
                elif new_pass != new_pass2:
                    st.error("Les mots de passe ne correspondent pas.")
                elif len(new_pass) < 4:
                    st.error("Mot de passe trop court (min 4).")
                elif new_email in st.session_state.users:
                    st.error("Ce compte existe déjà.")
                else:
                    st.session_state.users[new_email] = new_pass
                    st.success("Compte créé. Tu peux te connecter.")

def page_upload():
    if not st.session_state.is_logged:
        goto("login")

    topbar()
    st.markdown("## Chargement des données")
    st.caption("Dataset déjà nettoyé : upload → vérif schéma → aperçu → continuer.")

    left, right = st.columns([1, 1], gap="large")

    with left:
        with st.container(border=True):
            st.markdown("### Charger un CSV")
            file = st.file_uploader("Fichier CSV (données propres)", type=["csv"])
            st.caption("Tolérance : séparateur , ou ; • décimales FR ok (ex: 9839,64)")

            if file is not None:
                raw = file.getvalue().decode("utf-8", errors="ignore")
                delim = sniff_delimiter(raw)

                try:
                    df = pd.read_csv(io.StringIO(raw), sep=delim, engine="python")
                    st.session_state.df = df
                    st.success(f"Fichier chargé ({len(df)} lignes) • séparateur détecté: '{delim}'")
                except Exception as e:
                    st.session_state.df = None
                    st.error(f"Erreur lecture CSV : {e}")

    with right:
        with st.container(border=True):
            st.markdown("### Validation & aperçu (5 lignes)")
            if st.session_state.df is None:
                st.info("Charge un fichier pour voir la validation et l’aperçu.")
                return

            df = st.session_state.df
            ok, missing, extras = validate_schema(df)

            st.write("**Contrôles**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Colonnes", len(df.columns))
            c2.metric("Lignes", len(df))
            c3.metric("NA (total)", int(df.isna().sum().sum()))

            if ok:
                st.success("✅ Fichier conforme au schéma attendu.")
            else:
                st.error("❌ Fichier non conforme : colonnes manquantes.")

            if missing:
                st.write("**Colonnes manquantes** :", ", ".join(missing))
            if extras:
                st.write("**Colonnes inconnues (non bloquant)** :", ", ".join(extras))

            st.write("**Aperçu (5 premières lignes)**")
            st.dataframe(df.head(5), use_container_width=True, height=220)

            st.divider()
            st.caption("Colonnes attendues : " + ", ".join(EXPECTED_COLS) + " (+ isFraud optionnel)")

            if st.button("Valider & Continuer vers la prédiction", type="primary", disabled=not ok):
                st.session_state.pred_df = None
                goto("predict")

def page_predict():
    if not st.session_state.is_logged:
        goto("login")
    if st.session_state.df is None:
        goto("upload")

    topbar()
    st.markdown("## Prédiction")

    df = st.session_state.df
    ok, _, _ = validate_schema(df)
    if not ok:
        st.error("Le dataset n’est pas conforme. Retourne au chargement.")
        if st.button("Retour au chargement"):
            goto("upload")
        return

    left, right = st.columns([1.35, 1], gap="large")

    with left:
        with st.container(border=True):
            st.markdown("### Résultats — Vue globale (batch)")

            colA, colB, colC = st.columns([1, 1, 2])
            with colA:
                if st.button("Lancer la prédiction sur tout le CSV", type="primary"):
                    st.session_state.pred_df = run_batch_predictions(df)
            with colB:
                if st.button("Réinitialiser"):
                    st.session_state.pred_df = None
            with colC:
                st.caption("Décision + fiabilité (%) par transaction. Tri décroissant.")

            if st.session_state.pred_df is None:
                st.info("Clique sur **Lancer la prédiction** pour afficher la liste.")
            else:
                pred_df = st.session_state.pred_df
                show_only_fraud = st.checkbox("Afficher uniquement FRAUDE", value=False)
                view = pred_df if not show_only_fraud else pred_df[pred_df["decision"] == "FRAUDE"]
                st.dataframe(view, use_container_width=True, height=420)

    with right:
        with st.container(border=True):
            st.markdown("### Extraction d’une transaction (détail)")

            if st.session_state.pred_df is None:
                st.info("Lance d’abord la prédiction batch.")
                return

            pred_df = st.session_state.pred_df
            ids = pred_df["transactionId"].tolist() if "transactionId" in pred_df.columns else []
            if not ids:
                st.warning("Colonne transactionId absente : impossible de sélectionner une ligne.")
                return

            selected_id = st.selectbox("Choisir une transactionId", options=ids)
            row_pred = pred_df[pred_df["transactionId"] == selected_id].iloc[0].to_dict()
            row_raw = df[df["transactionId"] == selected_id].iloc[0].to_dict()

            st.write("**Résultat**")
            st.metric("Décision", row_pred.get("decision", "—"))
            st.metric("Fiabilité (%)", int(row_pred.get("fiabilite_pct", 0)))

            st.divider()
            st.write("**Données transaction**")
            show_keys = [
                "transactionId", "step", "type", "amount", "nameOrig", "nameDest",
                "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"
            ]
            details = {k: row_raw.get(k, "") for k in show_keys if k in row_raw}
            st.json(details)

            st.caption("Sélection = lecture du résultat déjà calculé (pas de recalcul).")

# -----------------------------
# Router
# -----------------------------
if st.session_state.page == "login":
    page_login()
elif st.session_state.page == "upload":
    page_upload()
elif st.session_state.page == "predict":
    page_predict()
else:
    goto("login")
