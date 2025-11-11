from utils import index_dataset
import os

# ============== CETTE PARTIE SERT A L'INDEXATION DE LA BASE ==============:
print("=== Phase 1 : Indexation CBIR ===")
dataset_path = os.path.join(os.path.dirname(__file__), "dataset")
for color_space in ["RGB", "HSV", "Lab"]:
    for bins in [8, 16, 32]:
        output = f"descriptors_{color_space}_{bins}.json"
        print(f"\n[Indexation] {color_space}, bins = {bins}")
        index_dataset(
            dataset_dir=dataset_path,
            bins_per_channel=bins,
            color_space=color_space,
            output_file=output
        )
