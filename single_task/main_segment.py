import argparse
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
import json
import matplotlib.pyplot as plt
import numpy as np

from data import build_dataset
from model import SegmentModel


def compute_metrics(preds, labels, num_classes, ignore_index=255):
    """Compute IoU and Accuracy metrics"""
    class_ious = []
    class_accs = []

    for cls in range(num_classes):
        pred_mask = (preds == cls)
        label_mask = (labels == cls)
        valid_mask = (labels != ignore_index)

        # IoU
        intersection = (pred_mask & label_mask & valid_mask).sum().item()
        union = ((pred_mask | label_mask) & valid_mask).sum().item()
        if union > 0:
            class_ious.append(intersection / union)

        # Accuracy
        correct = (pred_mask & label_mask & valid_mask).sum().item()
        total = (label_mask & valid_mask).sum().item()
        if total > 0:
            class_accs.append(correct / total)

    miou = np.mean(class_ious) if len(class_ious) > 0 else 0.0
    macc = np.mean(class_accs) if len(class_accs) > 0 else 0.0

    return class_ious, miou, class_accs, macc


def plot_loss_curves(log_file, output_dir):
    """Plot training loss curves"""
    with open(log_file, 'r') as f:
        logs = json.load(f)

    epochs = [log['epoch'] for log in logs]
    train_loss = [log.get('train_loss', 0) for log in logs]
    val_loss = [log.get('val_loss', None) for log in logs]
    val_miou = [log.get('val_miou', None) for log in logs]

    # Filter out None values
    val_epochs = [e for e, v in zip(epochs, val_loss) if v is not None]
    val_loss = [v for v in val_loss if v is not None]
    val_miou = [v for v in val_miou if v is not None]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, train_loss)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True)

    if val_loss:
        axes[1].plot(val_epochs, val_loss)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Validation Loss')
        axes[1].set_title('Validation Loss')
        axes[1].grid(True)

    if val_miou:
        axes[2].plot(val_epochs, val_miou)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('mIoU')
        axes[2].set_title('Validation mIoU')
        axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curves.png', dpi=150)
    plt.close()


def get_args_parser():
    parser = argparse.ArgumentParser('Segmentation Training', add_help=False)

    # Model parameters
    parser.add_argument('--model_name', default='convnext_small', type=str)
    parser.add_argument('--pretrained_path', default='dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth', type=str)
    parser.add_argument('--num_classes', default=4, type=int)
    parser.add_argument('--class_names', default=['Background', 'Head', 'Stem', 'Leaf'], type=list)

    # Dataset parameters
    parser.add_argument('--dataset_file', default='Segment', type=str)
    parser.add_argument('--data_path', default='./segmentation_dataset', type=str)

    # Training parameters
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr_drop', default=50, type=int)
    parser.add_argument('--eval_freq', default=1, type=int, help='Validation frequency (epochs)')

    # Other parameters
    parser.add_argument('--output_dir', default='./outputs_segment', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--seed', default=42, type=int)

    return parser


def main(args):
    device = torch.device(args.device)

    # Build model
    model = SegmentModel(
        model_name=args.model_name,
        pretrained_path=args.pretrained_path,
        num_classes=args.num_classes
    )
    model.to(device)

    # Build optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # Build dataset
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save training configuration
    config_file = output_dir / 'config.json'
    config = vars(args)
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    print(f"Training configuration saved to {config_file}")

    # Initialize training log
    log_file = output_dir / 'training_log.json'
    training_logs = []

    print("Start training")
    best_miou = 0.0

    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0

        for images, labels in data_loader_train:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        lr_scheduler.step()
        avg_train_loss = train_loss / len(data_loader_train)

        # Validate every eval_freq epochs or on the last epoch
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            # Validation
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for images, labels in data_loader_val:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    preds = outputs.argmax(dim=1)
                    all_preds.append(preds.cpu())
                    all_labels.append(labels.cpu())

            avg_val_loss = val_loss / len(data_loader_val)

            # Compute metrics
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            class_ious, miou, class_accs, macc = compute_metrics(all_preds, all_labels, args.num_classes)

            # Log
            log_entry = {
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_miou': miou,
                'val_macc': macc,
                'class_ious': class_ious,
                'class_accs': class_accs,
                'lr': optimizer.param_groups[0]['lr']
            }
            training_logs.append(log_entry)

            with open(log_file, 'w') as f:
                json.dump(training_logs, f, indent=2)

            # Plot curves every 10 epochs
            if (epoch + 1) % 10 == 0:
                plot_loss_curves(log_file, output_dir)

            print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
            print(f"  mIoU={miou:.4f}, mAcc={macc:.4f}, Best mIoU={best_miou:.4f}")
            for i, (name, iou, acc) in enumerate(zip(args.class_names, class_ious, class_accs)):
                print(f"  {name}: IoU={iou:.4f}, Acc={acc:.4f}")

            # Save best model
            if miou > best_miou:
                best_miou = miou
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                }
                torch.save(checkpoint, output_dir / 'best_checkpoint.pth')
        else:
            # Just log training loss
            log_entry = {
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'lr': optimizer.param_groups[0]['lr']
            }
            training_logs.append(log_entry)
            print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}")

        # Save checkpoint every epoch
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
        }
        torch.save(checkpoint, output_dir / 'checkpoint.pth')

    # Final plot
    plot_loss_curves(log_file, output_dir)
    print(f"Training completed. Best mIoU: {best_miou:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Segmentation training script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
