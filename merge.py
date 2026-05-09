import pandas as pd
import glob
import os

folder_path = "/content/EvaluationEMOTYC/golds"

fichiers = glob.glob(os.path.join(folder_path, "*.xlsx"))

dfs = [pd.read_excel(f) for f in fichiers]
resultat = pd.concat(dfs, ignore_index=True)

resultat.to_excel(os.path.join(folder_path, "CyberAdoAgg_gold_global_total.xlsx"), index=False)