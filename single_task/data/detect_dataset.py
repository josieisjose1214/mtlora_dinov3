import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class DetectDataset(Dataset):
    """Detection dataset for YOLO format annotations"""
    def __init__(self, root, split='train', transform=None, img_size=512):
        self.root = root
        self.split = split
        self.transform = transform
        self.img_size = img_size

        self.img_dir = os.path.join(root, 'images', split)
        self.label_dir = os.path.join(root, 'labels', split)

        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        print(f"[DetectDataset] {split}: {len(self.img_files)} images")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.rsplit('.', 1)[0] + '.txt')

        img = Image.open(img_path).convert('RGB')

        # Load YOLO format: class_id cx cy bw bh (all normalized to [0,1])
        # Convert directly to absolute coords at target img_size
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])

                        # Convert normalized coords to absolute coords at img_size
                        x1 = (cx - bw / 2) * self.img_size
                        y1 = (cy - bh / 2) * self.img_size
                        x2 = (cx + bw / 2) * self.img_size
                        y2 = (cy + bh / 2) * self.img_size

                        # Clamp
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(self.img_size, x2)
                        y2 = min(self.img_size, y2)

                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2])
                            labels.append(cls_id + 1)  # +1: background=0

        if self.transform:
            img = self.transform(img)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)

        return img, {'boxes': boxes, 'labels': labels}


def build(image_set, args):
    img_size = getattr(args, 'img_size', 512)
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return DetectDataset(args.data_path, split=image_set, transform=transform, img_size=img_size)
