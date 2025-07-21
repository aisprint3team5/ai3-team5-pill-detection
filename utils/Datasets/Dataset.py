import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class SampleDataset(Dataset):
    def __init__(self, image_dir, label_dir, class_to_idx, S=7, B=2, C=20, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.class_to_idx = class_to_idx
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        self.S = S
        self.B = B
        self.C = C
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        image_filename = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        # Load label
        label_path = os.path.join(self.label_dir, os.path.splitext(image_filename)[0] + '.txt')
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    cls_id, x_center, y_center, box_w, box_h = map(float, line.strip().split())
                    boxes.append([int(cls_id), x_center, y_center, box_w, box_h])

        boxes = torch.tensor(boxes) if boxes else torch.zeros((0, 5))  # Shape: (num_boxes, 5)

        if self.transform:
            image = self.transform(image)

        return image, boxes
