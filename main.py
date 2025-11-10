# main.py
from utils import index_dataset

if __name__ == "__main__":
    print("=== Phase 1 : Indexation CBIR ===")
    index_dataset(dataset_dir="dataset", bins_per_channel=16, output_file="descriptors.json")
