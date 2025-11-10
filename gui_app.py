# gui_app.py
import streamlit as st
from utils import (
    load_descriptors,
    search_similar_images,
    display_results,
    load_image,
)
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

# ================================
# CONFIGURATION GÉNÉRALE
# ================================

st.set_page_config(page_title="CBIR - Recherche d’Images par le Contenu", layout="wide")
st.title("🖼️ Système CBIR - Recherche d'Images par le Contenu")

dataset_path = os.path.join(os.path.dirname(__file__), "dataset")

# ================================
# PANNEAU LATÉRAL DE PARAMÈTRES
# ================================

st.sidebar.header("⚙️ Paramètres du système")

bins = st.sidebar.selectbox("Nombre de bins :", [8, 16, 32], index=1)
color_space = st.sidebar.selectbox("Espace couleur :", ["RGB", "HSV", "Lab"])
metric = st.sidebar.selectbox(
    "Mesure de distance :",
    ["hist_intersection", "euclidean", "chi2", "correlation"],
    index=0
)
top_n = st.sidebar.slider("Nombre d'images à afficher :", 3, 10, 5)

# Sélection automatique du fichier JSON selon les bins
json_file = f"descriptors_{bins}.json"
json_path = os.path.join(os.path.dirname(__file__), json_file)

if not os.path.exists(json_path):
    st.warning(f"⚠️ Le fichier {json_file} n'existe pas. Lance d'abord l’indexation avec {bins} bins.")
    st.stop()

# Charger les descripteurs
descriptors = load_descriptors(json_path)
st.sidebar.success(f"Fichier chargé : {json_file}")

# ================================
# ZONE DE CHARGEMENT D’IMAGE
# ================================

st.header("📤 Charger une image requête")
uploaded_file = st.file_uploader("Choisis une image depuis ton ordinateur :", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Lire l’image uploadée
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    query_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

    st.image(query_img, caption="Image requête", use_column_width=True)

    # ================================
    # LANCEMENT DE LA RECHERCHE
    # ================================
    st.header("🔎 Résultats de la recherche")

    if st.button("Lancer la recherche"):
        # Sauvegarde temporaire de l’image requête pour compatibilité
        tmp_path = os.path.join("query_tmp.jpg")
        cv2.imwrite(tmp_path, cv2.cvtColor(query_img, cv2.COLOR_RGB2BGR))

        # Lancer la recherche
        results = search_similar_images(
            query_image_path=tmp_path,
            descriptors=descriptors,
            bins_per_channel=bins,
            metric=metric,
            color_space=color_space,
            top_n=top_n
        )

        # Afficher les résultats
        cols = st.columns(top_n)
        for i, (img_name, score) in enumerate(results):
            img_path = os.path.join(dataset_path, img_name)
            img = load_image(img_path, color_space="RGB")
            cols[i].image(img, caption=f"{img_name}\nScore: {score:.3f}", use_container_width=True)

        # Supprimer l’image temporaire
        os.remove(tmp_path)
