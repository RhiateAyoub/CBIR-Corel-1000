import subprocess
import sys
import os
from utils import index_dataset

if __name__ == "__main__":
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

    
        
    # print("🚀 Lancement de l'interface CBIR (Streamlit)...\n")

    # # Récupère le chemin absolu de gui_app.py
    # app_path = os.path.join(os.path.dirname(__file__), "gui_app.py")

    # # Vérifie si Streamlit est installé
    # try:
    #     import streamlit
    # except ImportError:
    #     print("⚠️ Streamlit n'est pas installé. Installation en cours...")
    #     subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
    #     import streamlit

    # # Lance l'application Streamlit
    # try:
    #     subprocess.run(["streamlit", "run", app_path], check=True)
    # except KeyboardInterrupt:
    #     print("\n🛑 Application interrompue par l'utilisateur.")
    # except Exception as e:
    #     print(f"❌ Erreur lors du lancement de l'application : {e}")
