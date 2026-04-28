import argparse
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
import json
import numpy as np

from data import build_dataset
from model import DetectModel


def compute_ap(recalls, precisions):
    """Compute AP from precision-recall curve"""
    recalls = np.concatenate([[0], recalls, [1]])
    precisions = np.concatenate([[0], precisions, [0]])

    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])

    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return ap


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1,y1,x2,y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def evaluate_detection(all_predictions, all_targets, iou_thresh=0.5):
    """Compute AP at given IoU threshold"""
    tp_list = []
    fp_list = []
    scores_list = []
    num_gt = 0

    for preds, targets in zip(all_predictions, all_targets):
        pred_boxes = preds['boxes'].cpu().numpy()
        pred_scores = preds['scores'].cpu().numpy()
        gt_boxes = targets['boxes'].cpu().numpy()

        num_gt += len(gt_boxes)
        matched = np.zeros(len(gt_boxes), dtype=bool)

        for pred_box, score in zip(pred_boxes, pred_scores):
            scores_list.append(score)
            best_iou = 0
            best_idx = -1

            for idx, gt_box in enumerate(gt_boxes):
                if matched[idx]:
                    continue
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou >= iou_thresh and best_idx >= 0:
                tp_list.append(1)
                fp_list.append(0)
                matched[best_idx] = True
            else:
                tp_list.append(0)
                fp_list.append(1)

    if len(scores_list) == 0:
        return 0.0

    indices = np.argsort(scores_list)[::-1]
    tp = np.array(tp_list)[indices]
    fp = np.array(fp_list)[indices]

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recalls = tp_cumsum / max(num_gt, 1)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    return compute_ap(recalls, precisions)


def get_args_parser():
    parser = argparse.ArgumentParser('Detection Training', add_help=False)
    parser.add_argument('--model_name', default='convnext_small', type=str)
    parser.add_argument('--pretrained_path', default='dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth', type=str)
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--dataset_file', default='Detect', type=str)
    parser.add_argument('--data_path', default='./detect_dataset', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr_drop', default=50, type=int)
    parser.add_argument('--eval_freq', default=5, type=int)
    parser.add_argument('--output_dir', default='./outputs_detect', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--seed', default=42, type=int)
    return parser


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets


def main(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Build model
    model = DetectModel(model_name=args.model_name, pretrained_path=args.pretrained_path, num_classes=args.num_classes)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # Dataset
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / 'training_log.json'
    training_logs = []
    best_ap = 0.0

    print("Start training")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for images, targets in data_loader_train:
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            train_loss += losses.item()

        lr_scheduler.step()
        avg_train_loss = train_loss / len(data_loader_train)

        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            model.eval()
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for images, targets in data_loader_val:
                    images = images.to(device)
                    preds = model(images)
                    all_preds.extend(preds)
                    all_targets.extend(targets)

            ap50 = evaluate_detection(all_preds, all_targets, iou_thresh=0.5)
            ap75 = evaluate_detection(all_preds, all_targets, iou_thresh=0.75)
            ap = (ap50 + ap75) / 2

            log_entry = {'epoch': epoch, 'train_loss': avg_train_loss, 'AP50': ap50, 'AP75': ap75, 'AP': ap, 'lr': optimizer.param_groups[0]['lr']}
            training_logs.append(log_entry)

            with open(log_file, 'w') as f:
                json.dump(training_logs, f, indent=2)

            print(f"Epoch {epoch}: Loss={avg_train_loss:.4f}, AP50={ap50:.4f}, AP75={ap75:.4f}, AP={ap:.4f}, Best AP={best_ap:.4f}")

            if ap > best_ap:
                best_ap = ap
                torch.save({'model': model.state_dict(), 'epoch': epoch, 'best_ap': best_ap}, output_dir / 'best_checkpoint.pth')
        else:
            print(f"Epoch {epoch}: Loss={avg_train_loss:.4f}")

        torch.save({'model': model.state_dict(), 'epoch': epoch}, output_dir / 'checkpoint.pth')

    print(f"Training completed. Best AP: {best_ap:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Detection training script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

