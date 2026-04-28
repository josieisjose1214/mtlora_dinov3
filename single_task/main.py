import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
import matplotlib.pyplot as plt

from data import build_dataset
from engine import evaluate, train_one_epoch
from torch.utils.data import DataLoader
import pet_head.misc as utils
from model import build_model


def plot_loss_curves(log_file, output_dir):
    """Plot training loss curves"""
    with open(log_file, 'r') as f:
        logs = json.load(f)

    epochs = [log['epoch'] for log in logs]
    train_loss = [log.get('train_loss', 0) for log in logs]

    # Filter out None/0 values for validation metrics
    val_epochs = [log['epoch'] for log in logs if 'val_mae' in log]
    val_mae = [log['val_mae'] for log in logs if 'val_mae' in log]
    val_rmse = [log['val_rmse'] for log in logs if 'val_rmse' in log]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, train_loss)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True)

    if val_mae:
        axes[1].plot(val_epochs, val_mae)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Validation MAE')
        axes[1].grid(True)

    if val_rmse:
        axes[2].plot(val_epochs, val_rmse)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('RMSE')
        axes[2].set_title('Validation RMSE')
        axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curves.png', dpi=150)
    plt.close()


def get_args_parser():
    parser = argparse.ArgumentParser('Counting Model Training', add_help=False)

    # Model parameters
    parser.add_argument('--model_name', default='convnext_small', type=str,
                       help='Model name: convnext_small, vit_small, vit_base, vit_large')
    parser.add_argument('--pretrained_path', default='dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth', type=str,
                       help='Path to pretrained weights')
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--position_embedding', default='sine', type=str)
    parser.add_argument('--enc_layers', default=4, type=int)
    parser.add_argument('--dec_layers', default=2, type=int)
    parser.add_argument('--dim_feedforward', default=512, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--nheads', default=8, type=int)

    # Loss coefficients
    parser.add_argument('--ce_loss_coef', default=1.0, type=float)
    parser.add_argument('--point_loss_coef', default=5.0, type=float)
    parser.add_argument('--eos_coef', default=0.5, type=float)

    # Matcher parameters
    parser.add_argument('--set_cost_class', default=1.0, type=float)
    parser.add_argument('--set_cost_point', default=0.05, type=float)

    # Dataset parameters
    parser.add_argument('--dataset_file', default='SHA', type=str)
    parser.add_argument('--data_path', default='./data', type=str)

    # Training parameters
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--warmup_epochs', default=5, type=int, help='Learning rate warmup epochs')
    parser.add_argument('--clip_max_norm', default=0.1, type=float)
    parser.add_argument('--eval_freq', default=10, type=int, help='Validation frequency (epochs)')

    # Other parameters
    parser.add_argument('--output_dir', default='./outputs', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)

    return parser


def main(args):
    device = torch.device(args.device)

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Build model and criterion
    model, criterion = build_model(args)
    model.to(device)
    criterion.to(device)

    # Build optimizer
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad]}
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # Learning rate warmup function
    def adjust_learning_rate(optimizer, epoch, warmup_epochs, base_lr):
        if epoch < warmup_epochs:
            lr = base_lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        return None

    # Build dataset
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    # Build dataloader
    data_loader_train = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=utils.collate_fn
    )
    data_loader_val = DataLoader(
        dataset_val, batch_size=1, shuffle=False,
        num_workers=args.num_workers, collate_fn=utils.collate_fn
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save training configuration
    config_file = output_dir / 'config.json'
    if not args.resume:  # Only save config for new training
        config = vars(args)
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        print(f"Training configuration saved to {config_file}")

    # Initialize training log
    log_file = output_dir / 'training_log.json'
    training_logs = []

    # Resume from checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
        # Load training logs if exists
        if log_file.exists():
            with open(log_file, 'r') as f:
                training_logs = json.load(f)

    # Evaluation only
    if args.eval:
        test_stats = evaluate(model, data_loader_val, device)
        print(f"MAE: {test_stats['mae']:.2f}, RMSE: {test_stats['rmse']:.2f}, R²: {test_stats['r2']:.4f}")
        return

    # Training loop
    print("Start training")
    best_mae = float('inf')
    for epoch in range(args.start_epoch, args.epochs):
        # Learning rate warmup
        warmup_lr = adjust_learning_rate(optimizer, epoch, args.warmup_epochs, args.lr)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm
        )

        # Apply learning rate scheduler after warmup
        if epoch >= args.warmup_epochs:
            lr_scheduler.step()

        current_lr = warmup_lr if warmup_lr is not None else optimizer.param_groups[0]['lr']

        # Validate every eval_freq epochs or on the last epoch
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            # Evaluate
            test_stats = evaluate(model, data_loader_val, device, epoch)

            # Log training stats
            log_entry = {
                'epoch': epoch,
                'train_loss': train_stats.get('loss', 0),
                'val_mae': test_stats['mae'],
                'val_rmse': test_stats['rmse'],
                'val_r2': test_stats['r2'],
                'lr': current_lr
            }
            training_logs.append(log_entry)

            # Save logs
            with open(log_file, 'w') as f:
                json.dump(training_logs, f, indent=2)

            # Plot curves every 10 epochs
            if (epoch + 1) % 10 == 0:
                plot_loss_curves(log_file, output_dir)

            # Save best model
            if test_stats['mae'] < best_mae:
                best_mae = test_stats['mae']
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                torch.save(checkpoint, output_dir / 'best_checkpoint.pth')

            print(f"Epoch {epoch}: MAE={test_stats['mae']:.2f}, RMSE={test_stats['rmse']:.2f}, R²={test_stats['r2']:.4f}, Best MAE={best_mae:.2f}")
        else:
            # Just log training loss without validation
            log_entry = {
                'epoch': epoch,
                'train_loss': train_stats.get('loss', 0),
                'lr': current_lr
            }
            training_logs.append(log_entry)
            print(f"Epoch {epoch}: Train Loss={train_stats.get('loss', 0):.4f}, LR={current_lr:.6f}")

        # Save checkpoint every epoch
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args,
        }
        torch.save(checkpoint, output_dir / 'checkpoint.pth')

    # Final plot
    plot_loss_curves(log_file, output_dir)
    print(f"Training completed. Best MAE: {best_mae:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Counting training script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)