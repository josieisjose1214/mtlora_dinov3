"""
Script to visualize data loading and check counting points
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from data.count_dataset import CountDataset
import torchvision.transforms as standard_transforms
from PIL import Image

# Denormalize function
class DeNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

denorm = DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Transform
transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset
print("Loading training dataset...")
train_dataset = CountDataset('./count_dataset', train=True, transform=transform, flip=False)
print(f"Training samples: {len(train_dataset)}")

# print("\nLoading validation dataset...")
# val_dataset = CountDataset('./count_dataset', train=False, transform=transform)
# print(f"Validation samples: {len(val_dataset)}")

# Check first few samples
print("\n" + "="*50)
print("Visualizing samples with counting points:")
print("="*50)

num_samples = min(5, len(train_dataset))
fig, axes = plt.subplots(1, num_samples, figsize=(5*num_samples, 5))
if num_samples == 1:
    axes = [axes]

for i in range(num_samples):
    img, target = train_dataset[i]
    points = target['points'].numpy()
    count = len(points)

    # Denormalize image
    img_denorm = denorm(img.clone())
    img_np = (img_denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # Draw points
    img_vis = img_np.copy()
    for point in points:
        y, x = int(point[0]), int(point[1])
        cv2.circle(img_vis, (x, y), 3, (255, 0, 0), -1)

    axes[i].imshow(img_vis)
    axes[i].set_title(f'Sample {i}\nCount: {count}')
    axes[i].axis('off')

    print(f"Sample {i}: Count={count}, Image shape={img.shape}")

plt.tight_layout()
plt.savefig('data_visualization.png', dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to: data_visualization.png")
plt.close()

# Statistics
print("\n" + "="*50)
print("Dataset Statistics:")
print("="*50)

train_counts = []
for i in range(len(train_dataset)):
    _, target = train_dataset[i]
    train_counts.append(len(target['points']))

# val_counts = []
# for i in range(len(val_dataset)):
#     _, target = val_dataset[i]
#     val_counts.append(len(target['points']))

print(f"\nTraining set:")
print(f"  Mean count: {np.mean(train_counts):.2f}")
print(f"  Std count: {np.std(train_counts):.2f}")
print(f"  Min count: {np.min(train_counts)}")
print(f"  Max count: {np.max(train_counts)}")

# print(f"\nValidation set:")
# print(f"  Mean count: {np.mean(val_counts):.2f}")
# print(f"  Std count: {np.std(val_counts):.2f}")
# print(f"  Min count: {np.min(val_counts)}")
# print(f"  Max count: {np.max(val_counts)}")

print("\n" + "="*50)
print("Data check completed!")
print("="*50)
