import json
from collections import defaultdict
import pandas as pd
import numpy as np

folders = [
    "default_settings_HAR.75_seft",
    "default_settings_HAR.95_seft",
    "default_settings_HAR.99_seft",
]

for folder in folders:
    df_dict = defaultdict(list)
    for i in range(0, 5):
        json_path = f"{folder}/{i}/results.json"
        df_dict["Models"].append(f"Split {i}")
        json_object = json.load(open(json_path, "r"))
        df_dict["Accuracy"].append(json_object["test_acc"])
        df_dict["AUPRC"].append(json_object["test_auprc_macro"])
        df_dict["AUROC"].append(json_object["test_auroc_macro"])
    df_dict["Models"].append("Averages")
    df_dict["Accuracy"].append(np.average(df_dict["Accuracy"]))
    df_dict["AUPRC"].append(np.average(df_dict["AUPRC"]))
    df_dict["AUROC"].append(np.average(df_dict["AUROC"]))
    df_dict["Models"].append("St Dev")
    df_dict["Accuracy"].append(np.std(df_dict["Accuracy"]))
    df_dict["AUPRC"].append(np.std(df_dict["AUPRC"]))
    df_dict["AUROC"].append(np.std(df_dict["AUROC"]))
    df = pd.DataFrame(df_dict)
    print(folder)
    print(df)
    print("\n")
    print("\n")
