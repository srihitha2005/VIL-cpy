import os
import tqdm
ROOT = os.path.join("VIL_Dataset","dataset_filtered","train")  

for root, dirs, files in os.walk(ROOT):
    for file in tqdm.tqdm(files):
        if not file.lower().endswith(".png"):
            os.remove(os.path.join(root, file))
