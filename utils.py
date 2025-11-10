# utils.py
import cv2
import numpy as np
import os
import json

def load_image(path, size=(256, 256)):
    """Charge et redimensionne une image en RGB"""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Impossible de lire l'image : {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
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
