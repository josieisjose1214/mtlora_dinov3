import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
import json
import matplotlib.pyplot as plt
import numpy as np

from data import build_dataset
from model import ClassifyModel


def compute_balanced_accuracy(all_labels, all_preds, num_classes):
    """Balanced Accuracy: mean of per-class recall"""
    per_class_recall = []
    for c in range(num_classes):
        mask = (all_labels == c)
        if mask.sum() == 0:
            continue
        recall = (all_preds[mask] == c).sum().item() / mask.sum().item()
        per_class_recall.append(recall)
    return np.mean(per_class_recall) if per_class_recall else 0.0, per_class_recall


def compute_mAP(all_labels, all_probs, num_classes):
    """mean Average Precision from softmax probabilities"""
    aps = []
    for c in range(num_classes):
        # binary: is this class the ground truth?
        binary_labels = (all_labels == c).astype(np.float32)
        if binary_labels.sum() == 0:
            continue
        scores = all_probs[:, c]

        # sort by descending score
        sorted_indices = np.argsort(-scores)
        binary_labels = binary_labels[sorted_indices]

        # compute precision-recall and AP
        tp = np.cumsum(binary_labels)
        fp = np.cumsum(1 - binary_labels)
        precision = tp / (tp + fp)
        recall = tp / binary_labels.sum()

        # AP via trapezoidal rule (prepend 0 for recall)
        recall = np.concatenate([[0], recall])
        precision = np.concatenate([[1], precision])
        ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])
        aps.append(ap)

    return np.mean(aps) if aps else 0.0, aps


def plot_loss_curves(log_file, output_dir):
    """Plot training loss curves"""
    with open(log_file, 'r') as f:
        logs = json.load(f)

    epochs = [log['epoch'] for log in logs]
    train_loss = [log.get('train_loss', 0) for log in logs]
    val_loss = [log.get('val_loss', None) for log in logs]
    val_ba = [log.get('val_ba', None) for log in logs]
    val_mAP = [log.get('val_mAP', None) for log in logs]

    val_epochs = [e for e, v in zip(epochs, val_loss) if v is not None]
    val_loss = [v for v in val_loss if v is not None]
    val_ba = [v for v in val_ba if v is not None]
    val_mAP = [v for v in val_mAP if v is not None]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

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

    if val_ba:
        axes[2].plot(val_epochs, val_ba)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Balanced Accuracy')
        axes[2].set_title('Validation BA')
        axes[2].grid(True)

    if val_mAP:
        axes[3].plot(val_epochs, val_mAP)
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('mAP')
        axes[3].set_title('Validation mAP')
        axes[3].grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curves.png', dpi=150)
    plt.close()


def get_args_parser():
    parser = argparse.ArgumentParser('Classification Training', add_help=False)

    # Model parameters
    parser.add_argument('--model_name', default='convnext_small', type=str)
    parser.add_argument('--pretrained_path', default='dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth', type=str)
    parser.add_argument('--num_classes', default=8, type=int)

    # Dataset parameters
    parser.add_argument('--dataset_file', default='Classify', type=str)
    parser.add_argument('--data_path', default='./class_disease/classification_dataset', type=str)

    # Training parameters
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr_drop', default=50, type=int)
    parser.add_argument('--eval_freq', default=1, type=int, help='Validation frequency (epochs)')

    # Other parameters
    parser.add_argument('--output_dir', default='./outputs_classify', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--seed', default=42, type=int)

    return parser


def main(args):
    device = torch.device(args.device)

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Build dataset first to get num_classes
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    # Auto-detect num_classes from dataset
    if hasattr(dataset_train, 'num_classes'):
        args.num_classes = dataset_train.num_classes
        print(f"Auto-detected {args.num_classes} classes: {dataset_train.idx_to_class}")

    # Build model
    model = ClassifyModel(
        model_name=args.model_name,
        pretrained_path=args.pretrained_path,
        num_classes=args.num_classes
    )
    model.to(device)

    # Build optimizer (only trainable params)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # Build dataloader
    data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Loss function
    criterion = nn.CrossEntropyLoss()

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
    best_ba = 0.0

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

        # Validate
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            model.eval()
            val_loss = 0.0
            all_labels = []
            all_preds = []
            all_probs = []

            with torch.no_grad():
                for images, labels in data_loader_val:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    probs = F.softmax(outputs, dim=1)
                    preds = outputs.argmax(dim=1)

                    all_labels.append(labels.cpu())
                    all_preds.append(preds.cpu())
                    all_probs.append(probs.cpu())

            avg_val_loss = val_loss / max(len(data_loader_val), 1)

            all_labels = torch.cat(all_labels).numpy()
            all_preds = torch.cat(all_preds).numpy()
            all_probs = torch.cat(all_probs).numpy()

            # Balanced Accuracy
            ba, per_class_recall = compute_balanced_accuracy(all_labels, all_preds, args.num_classes)

            # mAP
            mAP, per_class_ap = compute_mAP(all_labels, all_probs, args.num_classes)

            # Per-class info
            per_class_info = {}
            cls_idx = 0
            for c in range(args.num_classes):
                if (all_labels == c).sum() == 0:
                    continue
                cls_name = dataset_train.idx_to_class.get(c, str(c))
                per_class_info[cls_name] = {
                    'recall': per_class_recall[cls_idx] if cls_idx < len(per_class_recall) else 0.0,
                    'AP': per_class_ap[cls_idx] if cls_idx < len(per_class_ap) else 0.0,
                }
                cls_idx += 1

            # Log
            log_entry = {
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_ba': ba,
                'val_mAP': mAP,
                'per_class': per_class_info,
                'lr': optimizer.param_groups[0]['lr']
            }
            training_logs.append(log_entry)

            with open(log_file, 'w') as f:
                json.dump(training_logs, f, indent=2)

            if (epoch + 1) % 10 == 0:
                plot_loss_curves(log_file, output_dir)

            print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, "
                  f"Val Loss={avg_val_loss:.4f}, BA={ba:.4f}, mAP={mAP:.4f}, Best BA={best_ba:.4f}")
            for cls_name, info in per_class_info.items():
                print(f"  {cls_name}: Recall={info['recall']:.4f}, AP={info['AP']:.4f}")

            # Save best model (based on BA)
            if ba > best_ba:
                best_ba = ba
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'best_ba': best_ba,
                    'best_mAP': mAP,
                    'class_to_idx': dataset_train.class_to_idx,
                }
                torch.save(checkpoint, output_dir / 'best_checkpoint.pth')
        else:
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
    print(f"Training completed. Best BA: {best_ba:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Classification training script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
