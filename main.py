# main.py
from utils import (
    index_dataset,
    load_descriptors,
    search_similar_images,
    display_results
)
import os

if __name__ == "__main__":
    # print("=== Phase 1 : Indexation CBIR ===")
    # index_dataset(dataset_dir="dataset", bins_per_channel=16, output_file="descriptors.json")
    print("=== Phase 2 : Recherche d'images similaires ===")

    # Définir le dossier dataset et le fichier JSON
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset")
    json_file = "descriptors.json"

    # Charger les descripteurs
    descriptors = load_descriptors(json_file)

    # Image requête (tu peux changer le chemin)
    query_image = os.path.join(dataset_path, "elephants", "500.jpg")

    # Rechercher les images similaires
    results = search_similar_images(
        query_image_path=query_image,
        descriptors=descriptors,
        dataset_dir=dataset_path,
        bins_per_channel=16,
        metric="hist_intersection",
        top_n=5
    )

    # Afficher les résultats
    display_results(query_image, results, dataset_dir=dataset_path)
