import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import xml.etree.ElementTree as ET
import torchvision.transforms as standard_transforms


class CountDataset(Dataset):
    def __init__(self, data_root, transform=None, train=False, flip=False):
        self.root_path = data_root
        self.train = train
        self.transform = transform
        self.flip = flip
        self.patch_size = 256

        # Get image and annotation paths
        split = "train" if train else "val"
        img_dir = os.path.join(data_root, "images", split)
        ann_dir = os.path.join(data_root, "annotations", split)

        self.img_list = []
        self.ann_list = []

        for img_name in os.listdir(img_dir):
            if img_name.endswith(('.jpg', '.JPG', '.png')):
                img_path = os.path.join(img_dir, img_name)
                ann_name = os.path.splitext(img_name)[0] + '.xml'
                ann_path = os.path.join(ann_dir, ann_name)

                if os.path.exists(ann_path):
                    self.img_list.append(img_path)
                    self.ann_list.append(ann_path)

        self.nSamples = len(self.img_list)

    def parse_xml(self, xml_path):
        """Parse XML annotation to get point coordinates"""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        points = []
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            # Align with legacy LeafTips loader:
            # read bbox as int and round center coordinates.
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            # Calculate center point
            cx = np.round((xmin + xmax) / 2).astype(np.int32)
            cy = np.round((ymin + ymax) / 2).astype(np.int32)
            points.append([cy, cx])  # [y, x] format

        return np.array(points) if len(points) > 0 else np.zeros((0, 2))

    def compute_density(self, points):
        """Compute crowd density"""
        if len(points) == 0:
            return torch.tensor(999.0).reshape(-1)

        points_tensor = torch.from_numpy(points.copy())
        dist = torch.cdist(points_tensor, points_tensor, p=2)
        if points_tensor.shape[0] > 1:
            density = dist.sort(dim=1)[0][:,1].mean().reshape(-1)
        else:
            density = torch.tensor(999.0).reshape(-1)
        return density

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        # Load image and points
        img_path = self.img_list[index]
        ann_path = self.ann_list[index]

        img = cv2.imread(img_path)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        points = self.parse_xml(ann_path).astype(float)

        # Transform
        if self.transform is not None:
            img = self.transform(img)
        img = torch.Tensor(img)

        # Random scale
        if self.train:
            scale_range = [0.8, 1.2]
            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)

            if scale * min_size > self.patch_size:
                img = torch.nn.functional.interpolate(
                    img.unsqueeze(0), scale_factor=scale, mode='bilinear'
                ).squeeze(0)
                points *= scale

        # Random crop
        if self.train:
            img, points = self.random_crop(img, points)

        # Random flip
        if random.random() > 0.5 and self.train and self.flip:
            img = torch.flip(img, dims=[2])
            points[:, 1] = self.patch_size - points[:, 1]

        # Target
        target = {}
        target['points'] = torch.Tensor(points)
        target['labels'] = torch.ones([points.shape[0]]).long()

        if self.train:
            target['density'] = self.compute_density(points)

        if not self.train:
            target['image_path'] = img_path

        return img, target

    def random_crop(self, img, points):
        """Random crop patch"""
        patch_h = patch_w = self.patch_size

        start_h = random.randint(0, img.size(1) - patch_h) if img.size(1) > patch_h else 0
        start_w = random.randint(0, img.size(2) - patch_w) if img.size(2) > patch_w else 0
        end_h = start_h + patch_h
        end_w = start_w + patch_w

        idx = (points[:, 0] >= start_h) & (points[:, 0] <= end_h) & \
              (points[:, 1] >= start_w) & (points[:, 1] <= end_w)

        result_img = img[:, start_h:end_h, start_w:end_w]
        result_points = points[idx]
        result_points[:, 0] -= start_h
        result_points[:, 1] -= start_w

        # Resize
        imgH, imgW = result_img.shape[-2:]
        fH, fW = patch_h/imgH, patch_w/imgW
        result_img = torch.nn.functional.interpolate(
            result_img.unsqueeze(0), (patch_h, patch_w)
        ).squeeze(0)
        result_points[:, 0] *= fH
        result_points[:, 1] *= fW

        return result_img, result_points


def build(image_set, args):
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_root = args.data_path
    if image_set == 'train':
        return CountDataset(data_root, train=True, transform=transform, flip=True)
    elif image_set == 'val':
        return CountDataset(data_root, train=False, transform=transform)
