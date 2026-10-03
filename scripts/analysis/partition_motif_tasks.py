import json
import glob
import os
import shutil
import tqdm

with open("samples_metadata.json") as fp:
    samples_metadata = json.load(fp)

os.makedirs("samples_by_task", exist_ok=True)

for entry, data in tqdm.tqdm(samples_metadata.items()):
    path = data['path']
    os.makedirs(os.path.join("samples_by_task", data['name']), exist_ok=True)
    shutil.copy(path, os.path.join("samples_by_task", data['name'], entry + ".pdb"))

    possible_folded_paths = glob.glob(
        os.path.join("esmfold", entry, "*.pdb")
    )
    folded_path = None
    for p in possible_folded_paths:
        if "model_name=v_48_020" in p:
            folded_path = p
            break
    os.makedirs(os.path.join("folded_by_task", data['name']), exist_ok=True)
    shutil.copy(folded_path, os.path.join("folded_by_task", data['name'], entry + "_folded.pdb"))
