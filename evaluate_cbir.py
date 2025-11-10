import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_descriptors, search_similar_images

# ===============================
# CONFIGURATION GÉNÉRALE
# ===============================
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")

# Nombre de bins utilisés pour tous les tests
BINS = 16
TOP_N = 10
QUERY_PER_CLASS = 5  # nombre d'images requêtes par classe testée

COLOR_SPACES = ["RGB", "HSV", "Lab"]
DISTANCES = ["hist_intersection", "euclidean", "chi2", "correlation"]

# Récupération des classes (sous-dossiers du dataset)
classes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]

# Liste pour enregistrer les résultats
results_summary = []

# ===============================
# FONCTION UTILITAIRE
# ===============================
def get_class_name(image_path):
    """Retourne le dossier parent (classe) d'une image."""
    return os.path.normpath(image_path).split(os.sep)[-2]


# ===============================
# BOUCLE PRINCIPALE D'ÉVALUATION
# ===============================
for color_space in COLOR_SPACES:
    # Charger le fichier descripteur correspondant à l’espace de couleur
    json_file = f"descriptors_{color_space}_{BINS}.json"
    if not os.path.exists(json_file):
        print(f"⚠️  Fichier {json_file} introuvable. Passe cette configuration.")
        continue

    descriptors = load_descriptors(json_file)
    print(f"\n=== Tests sur {color_space} ({BINS} bins) ===")

    for distance in DISTANCES:
        print(f"\n➡️ Distance : {distance}")
        total_correct = 0
        total_tests = 0

        for cls in classes:
            class_path = os.path.join(DATASET_DIR, cls)
            images = [os.path.join(class_path, f) for f in os.listdir(class_path)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))]

            if len(images) == 0:
                continue

            # Échantillon aléatoire d'images par classe
            sampled = random.sample(images, min(QUERY_PER_CLASS, len(images)))

            for query_path in sampled:
                similar = search_similar_images(
                    query_image_path=query_path,
                    descriptors=descriptors,
                    bins_per_channel=BINS,
                    metric=distance,
                    color_space=color_space,
                    top_n=TOP_N
                )

                # Compter combien des résultats appartiennent à la même classe
                correct = sum(
                    1 for (img_name, _) in similar
                    if get_class_name(img_name) == cls
                )

                total_correct += correct
                total_tests += TOP_N

        # Calcul du taux de précision
        accuracy = (total_correct / total_tests) * 100 if total_tests > 0 else 0
        results_summary.append({
            "ColorSpace": color_space,
            "Distance": distance,
            "Accuracy": round(accuracy, 2)
        })

        print(f"✅ Précision moyenne : {accuracy:.2f}%")


df = pd.DataFrame(results_summary)

# ===============================
# VISUALISATION DES RÉSULTATS
# ===============================
plt.figure(figsize=(10, 6))
for color_space in COLOR_SPACES:
    subset = df[df["ColorSpace"] == color_space]
    plt.plot(subset["Distance"], subset["Accuracy"], marker='o', label=color_space)

plt.title(f"Comparaison des distances selon l’espace de couleur ({BINS} bins)")
plt.xlabel("Mesure de distance")
plt.ylabel("Précision moyenne (%)")
plt.legend(title="Espace couleur")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"evaluation_results_{BINS}bins.png", dpi=300)
plt.show()

print(f"📊 Graphique enregistré sous : evaluation_results_{BINS}bins.png")
