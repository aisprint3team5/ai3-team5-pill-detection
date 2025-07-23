import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm

def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.9):
    images = list(Path(images_dir).glob("*.png"))
    random.shuffle(images)

    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    for phase, image_list in zip(["train", "val"], [train_images, val_images]):
        img_output_dir = Path(output_dir) / "images" / phase
        lbl_output_dir = Path(output_dir) / "labels" / phase
        img_output_dir.mkdir(parents=True, exist_ok=True)
        lbl_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[DEV - INFO] Copying {phase} set... ({len(image_list)} images)")
        for img_path in tqdm(image_list):
            label_path = Path(labels_dir) / (img_path.stem + ".txt")

            shutil.copy(img_path, img_output_dir / img_path.name)
            if label_path.exists():
                shutil.copy(label_path, lbl_output_dir / label_path.name)
