import os
import random
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

# Convert between PIL and tensor
to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

# MixUp function
def mixup_images(img1, img2, alpha=0.5):

    lam = np.random.beta(alpha, alpha)
    mixed = lam * img1 + (1 - lam) * img2
    return mixed.clamp(0, 1)
    

# Config
ROOT = os.path.join("VIL_Dataset","train")  # 🔁 Change to your dataset root
NUM_AUGS = 2              # 🔁 How many MixUps per image
SAVE_FORMAT = "png"       # "png" or "jpg" etc.

# Process each domain and class
for domain in os.listdir(ROOT):
    domain_path = os.path.join(ROOT, domain)
    if not os.path.isdir(domain_path):
        continue

    for cls in os.listdir(domain_path):
        class_path = os.path.join(domain_path, cls)
        if not os.path.isdir(class_path):
            continue

        img_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(img_files) < 2:
            continue

        for i, file1 in tqdm(list(enumerate(img_files)), desc=f"{domain}/{cls}"):
            path1 = os.path.join(class_path, file1)
            try:
                img1 = to_tensor(Image.open(path1).convert("RGB"))
            except:
                continue

            for aug_i in range(NUM_AUGS):
                file2 = random.choice(img_files)
                path2 = os.path.join(class_path, file2)
                try:
                    img2 = to_tensor(Image.open(path2).convert("RGB"))
                except:
                    continue
                try:
                    mixed = mixup_images(img1, img2)
                    mixed_img = to_pil(mixed)

                    base_name = os.path.splitext(file1)[0]
                    new_filename = f"{base_name}_aug_{aug_i}.{SAVE_FORMAT}"
                    new_path = os.path.join(class_path, new_filename)
                    mixed_img.save(new_path)
                except Exception as e:
                    print(f"Skipping due to error: {e}")
                    continue
