# 🖼️ CBIR – Content-Based Image Retrieval

L’objectif est de construire un mini-système de **recherche d’images par le contenu (CBIR)**,  
en utilisant les descripteurs basés sur la **couleur** des images.

---

## 📌 Objectifs du projet

- Extraire des **descripteurs de couleur** (histogrammes/histobines) à partir d’un dataset d’images.
- Implémenter la **recherche d’images similaires** en utilisant différentes distances :
  - Histogram Intersection (Swain & Ballard)
  - Euclidienne
  - Chi²
  - Corrélation
- Comparer les performances selon :
  - L’espace de couleur (RGB, HSV, Lab)
  - Le nombre de bins (8, 16, 32)
  - La mesure de distance utilisée
- Créer une **interface graphique (GUI)** permettant à l’utilisateur :
  - De charger une image requête
  - De configurer les paramètres (bins, espace couleur, distance, etc.)
  - D’afficher les images les plus similaires

---

## 🧠 Structure du projet
```
mini_projet_CBIR/
│
├── dataset/ # Dossier contenant les images à indexer
│
├── utils.py # Fonctions principales : histogrammes, distances, recherche
├── indexation.py # Script pour indexer le dataset (phase 1)
├── evaluate_cbir.py # Évaluation automatique (comparaison distances & espaces)
├── gui_app.py # Interface graphique Streamlit (phase 2)
├── main.py # Point d’entrée du projet : lance le GUI
└── README.md
```
---

## ⚙️ Installation

### 1. Cloner le projet
```bash
git clone https://github.com/RhiateAyoub/CBIR-Corel-1000.git
cd mini_projet_CBIR
```
### 2. Installer les dépendances
```bash
pip install -r requirements.txt
(ou installe manuellement si tu n’as pas encore le fichier)
```
```bash
pip install opencv-python numpy matplotlib streamlit
```
## 🚀 Utilisation
### 🔹 1. Phase d’indexation
Génère les descripteurs de couleur pour le dataset :
```bash
python index_dataset.py
```
Les fichiers .json produits sont enregistrés dans le dossier descriptors/.

### 🔹 2. Phase de recherche (avec GUI)
Lancer l’application graphique :

```bash
python main.py
```
L’interface permet de :

- Charger une image requête

Choisir :

- l’espace de couleur (RGB, HSV, Lab)

- le nombre de bins (8, 16, 32)

- la distance (Histogram Intersection, Euclidienne, Chi², Corrélation)

- Afficher les images les plus similaires à l’écran

### 🔹 3. Évaluation automatique
Comparer la précision moyenne selon les différents paramètres :

```bash
python evaluate_cbir.py
```
Le script affiche :

- la précision moyenne par distance et espace couleur

- enregistre un graphique (evaluation_results.png) pour le rapport


## 🧩 Détails techniques
Langage : Python 3

### Bibliothèques :

- OpenCV → traitement d’images

- NumPy → manipulation numérique

- Matplotlib → visualisation

- Streamlit → interface utilisateur

### Méthode de descripteur :

- Histogrammes par canal (R, G, B / H, S, V / L, a, b)

- Regroupement en bins (np.add.reduceat)

- Normalisation à somme unitaire

### Distances implémentées :

- Intersection d’histogrammes

- Euclidienne

- Chi²

- Corrélation

## 🧪 Tests et observations
- L’augmentation du nombre de bins (de 8 à 32) améliore légèrement la précision mais augmente le temps de calcul.

- L’espace RGB donne des résultats constants.

- L’espace HSV devient meilleur après normalisation photométrique.

## 👨‍💻 Auteur
### Ayoub Rhiate