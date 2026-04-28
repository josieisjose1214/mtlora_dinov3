import torch
import torch.nn as nn
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as standard_transforms


class SegmentDataset(Dataset):
    def __init__(self, path, split='train', transform=None):
        self.root = path
        img_root = os.path.join(os.path.join(self.root, split), "images")
        label_root = os.path.join(os.path.join(self.root, split), "class_id")
        self.transform = transform

        self.image_list = []
        self.label_list = []

        imgs = sorted(os.listdir(img_root))
        for f in imgs:
            img_path = os.path.join(img_root, f)
            label_path = os.path.join(label_root, f)
            self.image_list.append(img_path)
            self.label_list.append(label_path)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        label_path = self.label_list[idx]

        img = Image.open(img_path).convert("RGB")
        label = np.array(Image.open(label_path)).astype(np.int64)

        # Transform
        if self.transform is not None:
            img = self.transform(img)

        label = torch.from_numpy(label)

        return img, label


def build(image_set, args):
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_root = args.data_path
    if image_set == 'train':
        return SegmentDataset(data_root, split='train', transform=transform)
    elif image_set == 'val':
        return SegmentDataset(data_root, split='val', transform=transform)