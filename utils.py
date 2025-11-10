import cv2
import numpy as np
import os
import json
import matplotlib.pyplot as plt

def load_image(path, size=(256, 256), color_space="RGB"):
    """Charge, redimensionne et convertit une image vers l’espace de couleur choisi."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Impossible de lire l'image : {path}")
    img = cv2.resize(img, size)

    if color_space == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif color_space == "HSV":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif color_space == "Lab":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    else:
        raise ValueError(f"Espace de couleur non supporté : {color_space}")
    
    return img



def compute_histogram(img, bins_per_channel=16):
    """
    Calcule l'histogramme réduit (histobine) d'une image RGB.
    Retourne un vecteur de taille 3 * bins_per_channel.
    """
    descriptor = []
    for channel in range(3):  # R, G, B
        hist = cv2.calcHist([img], [channel], None, [256], [0, 256])
        hist = hist.flatten()
        # Normalisation
        hist /= hist.sum()

        # Regroupement en bins
        step = 256 // bins_per_channel
        hist_binned = np.add.reduceat(hist, np.arange(0, 256, step))
        hist_binned /= hist_binned.sum()
        descriptor.extend(hist_binned.tolist())

    return np.array(descriptor, dtype=np.float32)


def index_dataset(dataset_dir="dataset", bins_per_channel=16, output_file="descriptors.json"):
    """
    Parcourt récursivement toutes les images dans dataset/ et ses sous-dossiers,
    calcule leurs descripteurs, puis les sauvegarde dans un fichier JSON.
    """
    descriptors = {}
    supported_ext = (".jpg", ".jpeg", ".png")

    # Parcours récursif de tous les sous-dossiers
    all_images = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(supported_ext):
                all_images.append(os.path.join(root, file))

    print(f"[INFO] Indexation de {len(all_images)} images depuis '{dataset_dir}' ...")

    for i, img_path in enumerate(all_images, 1):
        filename = os.path.relpath(img_path, dataset_dir)  # Chemin relatif
        try:
            img = load_image(img_path)
            desc = compute_histogram(img, bins_per_channel=bins_per_channel)
            descriptors[filename] = desc.tolist()
        except Exception as e:
            print(f"[ERREUR] {filename}: {e}")

        if i % 100 == 0:
            print(f"  → {i} images traitées")

    # Sauvegarde descripteurs dans le fichier JSON
    with open(output_file, "w") as f:
        json.dump(descriptors, f, indent=4)

    print(f"[SUCCÈS] Descripteurs enregistrés dans '{output_file}' ({len(descriptors)} images indexées).")


# ================================
#   PHASE 2 : RECHERCHE D’IMAGES
# ================================

def load_descriptors(file_path="descriptors.json"):
    """Charge le fichier JSON contenant les descripteurs."""
    with open(file_path, "r") as f:
        return json.load(f)


def histogram_intersection(h1, h2):
    """Distance de Swain & Ballard (Histogram Intersection)"""
    return np.sum(np.minimum(h1, h2)) / np.sum(h1)


def euclidean_distance(h1, h2):
    """Distance Euclidienne"""
    return np.linalg.norm(h1 - h2)


def chi_square_distance(h1, h2):
    """Distance Chi²"""
    return 0.5 * np.sum(((h1 - h2) ** 2) / (h1 + h2 + 1e-10))


def correlation_distance(h1, h2):
    """Distance basée sur la corrélation"""
    return np.corrcoef(h1, h2)[0, 1]


def search_similar_images(query_image_path, descriptors, bins_per_channel=16,
                          metric="hist_intersection", color_space="RGB", top_n=5):
    """
    Recherche les images les plus similaires à l'image requête.
    - query_image_path : chemin de l'image requête
    - descriptors : dictionnaire chargé depuis descriptors.json
    - metric : 'hist_intersection', 'euclidean', 'chi2', 'correlation'
    - top_n : nombre d'images similaires à retourner
    """
    # Calcul du descripteur de l'image requête
    query_img = load_image(query_image_path, color_space=color_space)
    query_desc = compute_histogram(query_img, bins_per_channel=bins_per_channel)

    # Sélection de la distance
    if metric == "hist_intersection":
        distance_func = histogram_intersection
        reverse = True  # Plus la valeur est grande, plus c’est similaire
    elif metric == "euclidean":
        distance_func = euclidean_distance
        reverse = False  # Plus la valeur est petite, plus c’est similaire
    elif metric == "chi2":
        distance_func = chi_square_distance
        reverse = False
    elif metric == "correlation":
        distance_func = correlation_distance
        reverse = True
    else:
        raise ValueError(f"Métrique inconnue : {metric}")

    # Calcul de la similarité avec chaque image
    results = []
    for img_name, desc in descriptors.items():
        desc_np = np.array(desc, dtype=np.float32)
        score = distance_func(query_desc, desc_np)
        results.append((img_name, score))

    # Tri selon la mesure choisie
    results.sort(key=lambda x: x[1], reverse=reverse)

    # Retourne les top_n images les plus proches
    return results[:top_n]


def display_results(query_image_path, results, dataset_dir="dataset"):
    """Affiche l’image requête et les top résultats."""
    plt.figure(figsize=(12, 6))

    # Affiche l'image requête
    query_img = load_image(query_image_path)
    plt.subplot(2, 3, 1)
    plt.imshow(query_img)
    plt.title("Image requête")
    plt.axis("off")

    # Affiche les résultats
    for i, (img_name, score) in enumerate(results, 1):
        img_path = os.path.join(dataset_dir, img_name)
        img = load_image(img_path)
        plt.subplot(2, 3, i + 1)
        plt.imshow(img)
        plt.title(f"{img_name}\nScore: {score:.3f}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
