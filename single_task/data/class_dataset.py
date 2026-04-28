import os
import glob
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as standard_transforms


class ClassifyDisease(Dataset):
    def __init__(
        self,
        path,
        split='train',
        transform=None,
    ):
        self.root = path  # ./class_disease/classification_dataset
        self.split_dir = os.path.join(self.root, split)  # train/val
        self.transform = transform

        # 1. 查找所有类别名称 (子文件夹名)
        class_names = [d.name for d in os.scandir(self.split_dir) if d.is_dir()]
        self.class_to_idx = {name: i for i, name in enumerate(sorted(class_names))}
        self.idx_to_class = {i: name for name, i in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)

        # 2. 收集所有图像文件的路径和对应的标签
        self.image_list = []
        self.label_list = []

        for class_name, class_idx in self.class_to_idx.items():
            class_path = os.path.join(self.split_dir, class_name)
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.jfif', '*.webp'):
                for img_path in glob.glob(os.path.join(class_path, ext)):
                    self.image_list.append(img_path)
                    self.label_list.append(class_idx)

        print(f"[ClassifyDisease] {split}: {len(self.image_list)} images, {self.num_classes} classes: {self.class_to_idx}")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        label = self.label_list[idx]

        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(label, dtype=torch.long)
        return img, label


def build(image_set, args):
    transform = standard_transforms.Compose([
        standard_transforms.Resize((256, 256)),
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_root = args.data_path
    dataset = ClassifyDisease(data_root, split=image_set, transform=transform)
    return dataset
