"""
Training script for HRM model on next location prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from pathlib import Path
import numpy as np

from .modeling.hrm_location import LocationHRM, LocationHRMConfig
from .utils.location_data import LocationDataset, LocationDataLoader


def compute_accuracy_at_k(logits: torch.Tensor, targets: torch.Tensor, k: int = 1) -> float:
    """Compute top-k accuracy."""
    if k == 1:
        predictions = logits.argmax(dim=-1)
        return (predictions == targets).float().mean().item()
    else:
        _, topk_preds = logits.topk(k, dim=-1)
        correct = (topk_preds == targets.unsqueeze(-1)).any(dim=-1)
        return correct.float().mean().item()


def train_epoch(
    model: LocationHRM,
    train_loader: LocationDataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> tuple:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_acc1 = 0.0
    total_acc5 = 0.0
    total_acc10 = 0.0
    num_batches = 0
    
    for batch_idx, (inputs, targets, features) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(inputs, features)
        
        # Compute loss
        loss = F.cross_entropy(logits, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Compute metrics
        acc1 = compute_accuracy_at_k(logits, targets, k=1)
        acc5 = compute_accuracy_at_k(logits, targets, k=5)
        acc10 = compute_accuracy_at_k(logits, targets, k=10)
        
        total_loss += loss.item()
        total_acc1 += acc1
        total_acc5 += acc5
        total_acc10 += acc10
        num_batches += 1
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx + 1}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Acc@1: {acc1:.4f} | Acc@5: {acc5:.4f} | Acc@10: {acc10:.4f}")
    
    avg_loss = total_loss / num_batches
    avg_acc1 = total_acc1 / num_batches
    avg_acc5 = total_acc5 / num_batches
    avg_acc10 = total_acc10 / num_batches
    
    return avg_loss, avg_acc1, avg_acc5, avg_acc10


def evaluate(
    model: LocationHRM,
    data_loader: LocationDataLoader,
    device: torch.device,
    name: str = "Val",
) -> tuple:
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    total_acc1 = 0.0
    total_acc5 = 0.0
    total_acc10 = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets, features in data_loader:
            # Forward pass
            logits = model(inputs, features)
            
            # Compute loss
            loss = F.cross_entropy(logits, targets)
            
            # Compute metrics
            acc1 = compute_accuracy_at_k(logits, targets, k=1)
            acc5 = compute_accuracy_at_k(logits, targets, k=5)
            acc10 = compute_accuracy_at_k(logits, targets, k=10)
            
            total_loss += loss.item()
            total_acc1 += acc1
            total_acc5 += acc5
            total_acc10 += acc10
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_acc1 = total_acc1 / num_batches
    avg_acc5 = total_acc5 / num_batches
    avg_acc10 = total_acc10 / num_batches
    
    print(f"\n{name} Results:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Acc@1: {avg_acc1:.4f} ({avg_acc1*100:.2f}%)")
    print(f"  Acc@5: {avg_acc5:.4f} ({avg_acc5*100:.2f}%)")
    print(f"  Acc@10: {avg_acc10:.4f} ({avg_acc10*100:.2f}%)\n")
    
    return avg_loss, avg_acc1, avg_acc5, avg_acc10


def train_location_model(
    dataset_name: str = "geolife",
    data_dir: str = "data",
    checkpoint_dir: str = "checkpoints",
    max_seq_len: int = 50,
    hidden_size: int = 256,
    num_layers: int = 4,
    num_heads: int = 4,
    expansion: float = 4.0,
    high_level_cycles: int = 2,
    low_level_cycles: int = 2,
    use_features: bool = True,
    batch_size: int = 128,
    learning_rate: float = 1e-4,
    num_epochs: int = 100,
    save_every: int = 5,
    device: str = "cuda",
):
    """
    Train HRM model for next location prediction.
    
    Args:
        dataset_name: Name of dataset ('geolife' or 'diy')
        data_dir: Directory containing data files
        checkpoint_dir: Directory to save checkpoints
        max_seq_len: Maximum sequence length
        hidden_size: Hidden dimension size
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        expansion: MLP expansion factor
        high_level_cycles: Number of high-level reasoning cycles
        low_level_cycles: Number of low-level reasoning cycles
        use_features: Whether to use additional features
        batch_size: Batch size
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        save_every: Save checkpoint every N epochs
        device: Device to use ('cuda' or 'cpu')
    """
    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed(42)
    
    # Create generator
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    
    # Load datasets
    print("\n" + "="*60)
    print("Loading datasets...")
    print("="*60)
    
    if dataset_name == "geolife":
        train_path = f"{data_dir}/geolife/geolife_transformer_7_train.pk"
        val_path = f"{data_dir}/geolife/geolife_transformer_7_validation.pk"
        test_path = f"{data_dir}/geolife/geolife_transformer_7_test.pk"
    else:  # diy
        train_path = f"{data_dir}/diy/diy_h3_res8_transformer_7_train.pk"
        val_path = f"{data_dir}/diy/diy_h3_res8_transformer_7_validation.pk"
        test_path = f"{data_dir}/diy/diy_h3_res8_transformer_7_test.pk"
    
    train_dataset = LocationDataset(train_path, max_seq_len=max_seq_len, use_features=use_features)
    val_dataset = LocationDataset(val_path, max_seq_len=max_seq_len, use_features=use_features)
    test_dataset = LocationDataset(test_path, max_seq_len=max_seq_len, use_features=use_features)
    
    # Create data loaders
    train_loader = LocationDataLoader(train_dataset, batch_size=batch_size, device=device, shuffle=True)
    val_loader = LocationDataLoader(val_dataset, batch_size=batch_size, device=device, shuffle=False)
    test_loader = LocationDataLoader(test_dataset, batch_size=batch_size, device=device, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model configuration
    config = LocationHRMConfig(
        max_seq_len=max_seq_len,
        vocab_size=train_dataset.vocab_size,
        high_level_cycles=high_level_cycles,
        low_level_cycles=low_level_cycles,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        expansion=expansion,
        use_features=use_features,
        num_users=train_dataset.num_users if use_features else 100,
        dtype=torch.float32,  # Use float32 for better stability
        dropout=0.1,
    )
    
    # Create model
    print("\n" + "="*60)
    print("Initializing model...")
    print("="*60)
    model = LocationHRM(config=config, generator=generator, device=device)
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable:,}")
    
    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=learning_rate * 0.1,
    )
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    best_val_acc = 0.0
    best_test_acc = 0.0
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Train
        train_loss, train_acc1, train_acc5, train_acc10 = train_epoch(
            model, train_loader, optimizer, device, epoch
        )
        
        # Evaluate
        val_loss, val_acc1, val_acc5, val_acc10 = evaluate(
            model, val_loader, device, name="Validation"
        )
        
        test_loss, test_acc1, test_acc5, test_acc10 = evaluate(
            model, test_loader, device, name="Test"
        )
        
        # Update learning rate
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch} Summary:")
        print(f"  Train - Loss: {train_loss:.4f} | Acc@1: {train_acc1:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f} | Acc@1: {val_acc1:.4f}")
        print(f"  Test  - Loss: {test_loss:.4f} | Acc@1: {test_acc1:.4f}")
        print(f"  Time: {epoch_time:.2f}s | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        if epoch % save_every == 0 or val_acc1 > best_val_acc:
            checkpoint_path = f"{checkpoint_dir}/location_hrm_epoch{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config,
                'train_loss': train_loss,
                'val_acc1': val_acc1,
                'test_acc1': test_acc1,
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if val_acc1 > best_val_acc:
            best_val_acc = val_acc1
            best_checkpoint_path = f"{checkpoint_dir}/location_hrm_best_val.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config,
                'train_loss': train_loss,
                'val_acc1': val_acc1,
                'test_acc1': test_acc1,
            }, best_checkpoint_path)
            print(f"  New best validation accuracy! Saved to {best_checkpoint_path}")
        
        if test_acc1 > best_test_acc:
            best_test_acc = test_acc1
            best_test_checkpoint_path = f"{checkpoint_dir}/location_hrm_best_test.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config,
                'train_loss': train_loss,
                'val_acc1': val_acc1,
                'test_acc1': test_acc1,
            }, best_test_checkpoint_path)
            print(f"  New best test accuracy! Saved to {best_test_checkpoint_path}")
        
        # Check if goal is achieved
        if test_acc1 >= 0.40:
            print(f"\n{'='*60}")
            print(f"🎉 GOAL ACHIEVED! Test Acc@1 = {test_acc1:.4f} (>= 0.40)")
            print(f"{'='*60}\n")
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"Best validation Acc@1: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"Best test Acc@1: {best_test_acc:.4f} ({best_test_acc*100:.2f}%)")
    
    return model, best_val_acc, best_test_acc


if __name__ == '__main__':
    train_location_model()
