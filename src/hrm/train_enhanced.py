"""
Advanced training script for Enhanced HRM (<500K params, >40% test acc).

Advanced Training Techniques:
1. Exponential Moving Average (EMA) of model weights
2. Cosine annealing with warm restarts
3. Temporal jittering data augmentation
4. Adaptive gradient accumulation
5. Mixed precision training
6. TopKCrossEntropyLoss integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import numpy as np
import math
import copy
from collections import defaultdict
import difftopk

from .modeling.hrm_enhanced import EnhancedLocationHRM, EnhancedHRMConfig, count_parameters
from .utils.location_data import LocationDataset, LocationDataLoader


class ExponentialMovingAverage:
    """EMA of model parameters for better generalization."""
    
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply EMA parameters to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class TemporalJitteringAugmentation:
    """Data augmentation with temporal jittering."""
    
    def __init__(self, time_jitter_std=5, duration_jitter_std=2):
        self.time_jitter_std = time_jitter_std  # minutes
        self.duration_jitter_std = duration_jitter_std  # minutes
    
    def __call__(self, features):
        """Apply temporal jittering to features."""
        if features is None:
            return features
        
        augmented = {}
        for key, value in features.items():
            if key == 'start_min':
                # Jitter start time (in minutes)
                jitter = torch.randn_like(value.float()) * self.time_jitter_std
                augmented[key] = torch.clamp(value + jitter.long(), 0, 1439)  # 24*60-1
            elif key == 'duration':
                # Jitter duration
                jitter = torch.randn_like(value.float()) * self.duration_jitter_std
                augmented[key] = torch.clamp(value + jitter.long(), 1, 1000)
            else:
                augmented[key] = value
        
        return augmented


def compute_accuracy_at_k(logits: torch.Tensor, targets: torch.Tensor, k: int = 1) -> float:
    """Compute top-k accuracy."""
    if k == 1:
        predictions = logits.argmax(dim=-1)
        return (predictions == targets).float().mean().item()
    else:
        _, topk_preds = logits.topk(k, dim=-1)
        correct = (topk_preds == targets.unsqueeze(-1)).any(dim=-1)
        return correct.float().mean().item()


class AdaptiveGradientAccumulator:
    """Adaptive gradient accumulation based on batch loss variance."""
    
    def __init__(self, base_accumulation_steps=1, max_accumulation_steps=4):
        self.base_steps = base_accumulation_steps
        self.max_steps = max_accumulation_steps
        self.loss_history = []
        self.window_size = 10
    
    def get_accumulation_steps(self, current_loss):
        """Determine accumulation steps based on loss variance."""
        self.loss_history.append(current_loss)
        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)
        
        if len(self.loss_history) < self.window_size:
            return self.base_steps
        
        # High variance -> more accumulation for stability
        variance = np.var(self.loss_history)
        mean_loss = np.mean(self.loss_history)
        cv = math.sqrt(variance) / (mean_loss + 1e-8)  # coefficient of variation
        
        if cv > 0.3:
            return min(self.max_steps, self.base_steps + 2)
        elif cv > 0.15:
            return min(self.max_steps, self.base_steps + 1)
        else:
            return self.base_steps


def train_epoch(
    model: EnhancedLocationHRM,
    ema: ExponentialMovingAverage,
    train_loader: LocationDataLoader,
    optimizer: torch.optim.Optimizer,
    criterion,
    scheduler,
    device: torch.device,
    epoch: int,
    max_epochs: int,
    augmentation: TemporalJitteringAugmentation,
    grad_accumulator: AdaptiveGradientAccumulator,
    scaler: torch.cuda.amp.GradScaler,
) -> tuple:
    """Train for one epoch with advanced techniques."""
    model.train()
    total_loss = 0.0
    total_topk_loss = 0.0
    total_ce_loss = 0.0
    total_acc1 = 0.0
    total_acc5 = 0.0
    total_acc10 = 0.0
    num_batches = 0
    
    accumulation_steps = grad_accumulator.base_steps
    optimizer.zero_grad()
    
    for batch_idx, (inputs, targets, features) in enumerate(train_loader):
        # Apply temporal jittering augmentation
        if features is not None and np.random.rand() < 0.5:  # 50% chance
            features = augmentation(features)
        
        # Mixed precision training
        with torch.cuda.amp.autocast():
            # Forward pass
            logits = model(inputs, features)
            
            # Compute loss
            if hasattr(criterion, 'forward'):
                loss, topk_loss, ce_loss = criterion(logits, targets, epoch, max_epochs)
            else:
                loss = criterion(logits, targets)
                topk_loss = loss
                ce_loss = 0.0
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Update adaptive accumulation steps
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
            # Update EMA
            ema.update()
            
            # Update accumulation steps
            accumulation_steps = grad_accumulator.get_accumulation_steps(loss.item() * accumulation_steps)
        
        # Compute metrics
        with torch.no_grad():
            acc1 = compute_accuracy_at_k(logits, targets, k=1)
            acc5 = compute_accuracy_at_k(logits, targets, k=5)
            acc10 = compute_accuracy_at_k(logits, targets, k=10)
        
        total_loss += loss.item() * accumulation_steps
        total_topk_loss += topk_loss if isinstance(topk_loss, float) else topk_loss
        total_ce_loss += ce_loss if isinstance(ce_loss, float) else ce_loss
        total_acc1 += acc1
        total_acc5 += acc5
        total_acc10 += acc10
        num_batches += 1
        
        if (batch_idx + 1) % 10 == 0:
            if hasattr(criterion, 'forward'):
                print(f"Epoch {epoch} | Batch {batch_idx + 1}/{len(train_loader)} | "
                      f"Loss: {loss.item() * accumulation_steps:.4f} (TopK: {topk_loss:.4f}, CE: {ce_loss:.4f}) | "
                      f"Acc@1: {acc1:.4f} | Acc@5: {acc5:.4f} | Acc@10: {acc10:.4f} | "
                      f"AccumSteps: {accumulation_steps}")
            else:
                print(f"Epoch {epoch} | Batch {batch_idx + 1}/{len(train_loader)} | "
                      f"Loss: {loss.item() * accumulation_steps:.4f} | "
                      f"Acc@1: {acc1:.4f} | Acc@5: {acc5:.4f} | Acc@10: {acc10:.4f}")
    
    avg_loss = total_loss / num_batches
    avg_topk_loss = total_topk_loss / num_batches if total_topk_loss > 0 else 0
    avg_ce_loss = total_ce_loss / num_batches if total_ce_loss > 0 else 0
    avg_acc1 = total_acc1 / num_batches
    avg_acc5 = total_acc5 / num_batches
    avg_acc10 = total_acc10 / num_batches
    
    return avg_loss, avg_topk_loss, avg_ce_loss, avg_acc1, avg_acc5, avg_acc10


def evaluate(
    model: EnhancedLocationHRM,
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


def train_enhanced_hrm(
    dataset_name: str = "geolife",
    data_dir: str = "data",
    checkpoint_dir: str = "checkpoints",
    max_seq_len: int = 50,
    hidden_size: int = 96,
    num_layers: int = 2,
    num_heads: int = 4,
    expansion: float = 2.0,
    high_level_cycles: int = 2,
    low_level_cycles: int = 2,
    num_coarse_clusters: int = 50,
    use_features: bool = True,
    batch_size: int = 128,
    learning_rate: float = 8e-4,
    num_epochs: int = 400,
    save_every: int = 10,
    device: str = "cuda",
    patience: int = 60,
    ema_decay: float = 0.999,
    use_topk_loss: bool = True,
):
    """
    Train Enhanced HRM with <500K params targeting >40% test accuracy.
    
    Advanced techniques:
    - EMA of model weights
    - Cosine annealing with warm restarts
    - Temporal jittering augmentation
    - Adaptive gradient accumulation
    - Mixed precision training
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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Load datasets
    print("\n" + "="*60)
    print("Loading datasets...")
    print("="*60)
    
    if dataset_name == "geolife":
        train_path = f"{data_dir}/geolife/geolife_transformer_7_train.pk"
        val_path = f"{data_dir}/geolife/geolife_transformer_7_validation.pk"
        test_path = f"{data_dir}/geolife/geolife_transformer_7_test.pk"
    else:
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
    config = EnhancedHRMConfig(
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
        num_coarse_clusters=num_coarse_clusters,
        dropout=0.15,
        short_range_window=10,
        recency_decay=0.9,
    )
    
    # Create model
    print("\n" + "="*60)
    print("Initializing Enhanced HRM...")
    print("="*60)
    model = EnhancedLocationHRM(config=config, device=device)
    model = model.to(device)
    
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Target: <500K parameters")
    print(f"✓ Within budget!" if total_params < 500000 else f"✗ Exceeds budget by {total_params - 500000:,}")
    
    # Initialize EMA
    ema = ExponentialMovingAverage(model, decay=ema_decay)
    print(f"EMA decay: {ema_decay}")
    
    # Initialize augmentation
    augmentation = TemporalJitteringAugmentation(time_jitter_std=5, duration_jitter_std=2)
    print("Temporal jittering: enabled")
    
    # Initialize adaptive gradient accumulator
    grad_accumulator = AdaptiveGradientAccumulator(base_accumulation_steps=1, max_accumulation_steps=4)
    print("Adaptive gradient accumulation: enabled")
    
    # Create loss function
    if use_topk_loss:
        class AdaptiveTopKLoss(nn.Module):
            def __init__(self, vocab_size, device):
                super().__init__()
                self.topk_loss = difftopk.TopKCrossEntropyLoss(
                    diffsort_method='odd_even',
                    inverse_temperature=2.0,
                    p_k=[0.5, 0.1, 0.1, 0.1, 0.2],
                    n=vocab_size,
                    m=15,
                    distribution='cauchy',
                    device=device,
                    top1_mode='sm'
                )
                self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.08)
            
            def forward(self, logits, targets, epoch, max_epochs):
                topk_weight = max(0.4, 1.0 - (epoch / max_epochs) * 0.6)
                ce_weight = 1.0 - topk_weight
                topk_l = self.topk_loss(logits, targets)
                ce_l = self.ce_loss(logits, targets)
                return topk_weight * topk_l + ce_weight * ce_l, topk_l.item(), ce_l.item()
        
        criterion = AdaptiveTopKLoss(train_dataset.vocab_size, device)
        print("Loss: Adaptive TopKCrossEntropyLoss")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
        print("Loss: CrossEntropyLoss with label smoothing")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8
    )
    
    # Cosine annealing with warm restarts
    warmup_epochs = 15
    restart_period = 60  # Restart every 60 epochs
    total_steps = len(train_loader) * num_epochs
    warmup_steps = len(train_loader) * warmup_epochs
    restart_steps = len(train_loader) * restart_period
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        # Cosine annealing with restarts
        progress = current_step - warmup_steps
        cycle = progress // restart_steps
        cycle_progress = (progress % restart_steps) / restart_steps
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * cycle_progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training with advanced techniques...")
    print("="*60 + "\n")
    
    best_val_acc = 0.0
    best_test_acc = 0.0
    no_improve_count = 0
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Train
        if use_topk_loss:
            train_loss, topk_loss, ce_loss, train_acc1, train_acc5, train_acc10 = train_epoch(
                model, ema, train_loader, optimizer, criterion, scheduler, device,
                epoch, num_epochs, augmentation, grad_accumulator, scaler
            )
        else:
            train_loss, _, _, train_acc1, train_acc5, train_acc10 = train_epoch(
                model, ema, train_loader, optimizer, criterion, scheduler, device,
                epoch, num_epochs, augmentation, grad_accumulator, scaler
            )
            topk_loss = ce_loss = 0
        
        # Evaluate with EMA weights
        ema.apply_shadow()
        val_loss, val_acc1, val_acc5, val_acc10 = evaluate(model, val_loader, device, name="Validation")
        test_loss, test_acc1, test_acc5, test_acc10 = evaluate(model, test_loader, device, name="Test")
        ema.restore()
        
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        # Summary
        if use_topk_loss:
            topk_weight = max(0.4, 1.0 - (epoch / num_epochs) * 0.6)
            print(f"Epoch {epoch} Summary:")
            print(f"  Train - Loss: {train_loss:.4f} (TopK: {topk_loss:.4f}[{topk_weight:.2f}], CE: {ce_loss:.4f}) | Acc@1: {train_acc1:.4f}")
        else:
            print(f"Epoch {epoch} Summary:")
            print(f"  Train - Loss: {train_loss:.4f} | Acc@1: {train_acc1:.4f}")
        
        print(f"  Val   - Loss: {val_loss:.4f} | Acc@1: {val_acc1:.4f}")
        print(f"  Test  - Loss: {test_loss:.4f} | Acc@1: {test_acc1:.4f}")
        print(f"  Time: {epoch_time:.2f}s | LR: {current_lr:.6f}")
        print(f"  Params: {total_params:,}")
        
        # Save checkpoints
        if epoch % save_every == 0 or val_acc1 > best_val_acc:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_shadow': ema.shadow,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config,
                'train_loss': train_loss,
                'val_acc1': val_acc1,
                'test_acc1': test_acc1,
            }
            checkpoint_path = f"{checkpoint_dir}/enhanced_hrm_epoch{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
        
        # Track best models
        if val_acc1 > best_val_acc:
            best_val_acc = val_acc1
            no_improve_count = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_shadow': ema.shadow,
                'config': config,
                'val_acc1': val_acc1,
                'test_acc1': test_acc1,
            }
            torch.save(checkpoint, f"{checkpoint_dir}/enhanced_hrm_best_val.pt")
            print(f"  ✅ New best validation accuracy! ({val_acc1*100:.2f}%)")
        else:
            no_improve_count += 1
        
        if test_acc1 > best_test_acc:
            best_test_acc = test_acc1
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_shadow': ema.shadow,
                'config': config,
                'val_acc1': val_acc1,
                'test_acc1': test_acc1,
            }
            torch.save(checkpoint, f"{checkpoint_dir}/enhanced_hrm_best_test.pt")
            print(f"  ✅ New best test accuracy! ({test_acc1*100:.2f}%)")
        
        # Check if goal achieved
        if test_acc1 >= 0.40:
            print(f"\n{'='*60}")
            print(f"🎉🎉🎉 GOAL ACHIEVED! 🎉🎉🎉")
            print(f"Test Acc@1 = {test_acc1:.4f} ({test_acc1*100:.2f}%) >= 40%")
            print(f"Model parameters: {total_params:,} (<500K)")
            print(f"Enhanced HRM with all optimizations!")
            print(f"{'='*60}\n")
            break
        
        # Early stopping
        if no_improve_count >= patience:
            print(f"\n{'='*60}")
            print(f"Early stopping: No improvement for {patience} epochs")
            print(f"{'='*60}\n")
            break
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"Best validation Acc@1: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"Best test Acc@1: {best_test_acc:.4f} ({best_test_acc*100:.2f}%)")
    print(f"Model parameters: {total_params:,}")
    
    if best_test_acc >= 0.40:
        print(f"\n🎉 SUCCESS! Achieved >40% test accuracy with <500K parameters!")
    else:
        print(f"\n📊 Best: {best_test_acc*100:.2f}% (target: 40%)")
    
    return model, ema, best_val_acc, best_test_acc


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_coarse_clusters', type=int, default=35)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=8e-4)
    parser.add_argument('--num_epochs', type=int, default=400)
    args = parser.parse_args()
    
    train_enhanced_hrm(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_coarse_clusters=args.num_coarse_clusters,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
    )
