import pandas as pd
import os

# Configuration
files = [
    "golds/homophobie_annotations_gold_flat_updated.xlsx",
    "golds/obésité_annotations_gold_flat_updated.xlsx",
    "golds/racisme_annotations_gold_flat_updated.xlsx",
    "golds/religion_annotations_gold_flat_updated.xlsx"
]
output_file = "golds/random_sample_120.xlsx"
sample_size = 30

def main():
    all_samples = []
    
    for file_path in files:
        if not os.path.exists(file_path):
            print(f"Les 4 fichiers ne répondent pas tous à l'appel")
            sys.exit(1)
            
        try:
            # Chargement XLSX
            df = pd.read_excel(file_path)
            
            # Echantillonnage
            sample = df.sample(sample_size)
            
            all_samples.append(sample)
            print(f"Echantillonnage opéré !")
            
        except Exception as e:
            print(f"Erreur avec {file_path}: {e}")

    if all_samples:
        # Concaténation des DataFrames
        final_df = pd.concat(all_samples, ignore_index=True)
        
        # Save dans un new XLSX
        final_df.to_excel(output_file, index=False)
        print(f"\nSuccess! Echantillonage dans {output_file}")
    else:
        print("Erreur")

if __name__ == "__main__":
    main()
