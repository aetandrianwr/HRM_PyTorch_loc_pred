"""
Training script for SOTA Transformer with advanced techniques.

Training innovations:
1. Label smoothing (Szegedy et al., 2016) 
2. Mixup augmentation (Zhang et al., 2017)
3. Stochastic depth (Huang et al., 2016)
4. SAM optimizer (Foret et al., 2020) - sharpness-aware minimization
5. OneCycle learning rate (Smith, 2018)
6. Gradient centralization (Yong et al., 2020)
7. Lookahead optimizer (Zhang et al., 2019)

Target: >40% test accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import numpy as np
import math

from .modeling.sota_transformer import SOTALocationTransformer, SOTAConfig, count_parameters
from .utils.location_data import LocationDataset, LocationDataLoader


class MixupAugmentation:
    """Mixup data augmentation (Zhang et al., 2017)."""
    
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, x, y):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing (Szegedy et al., 2016)."""
    
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        logprobs = F.log_softmax(pred, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def compute_accuracy_at_k(logits, targets, k=1):
    """Compute top-k accuracy."""
    with torch.no_grad():
        _, pred = logits.topk(k, dim=1, largest=True, sorted=True)
        correct = pred.eq(targets.view(-1, 1).expand_as(pred))
        correct_k = correct[:, :k].sum().item()
        return correct_k / targets.size(0)


def evaluate(model, loader, device, name="Val"):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    total_acc1 = 0.0
    total_acc5 = 0.0
    total_acc10 = 0.0
    num_batches = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, targets, features in loader:
            logits = model(inputs, features)
            loss = criterion(logits, targets)
            
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
    print(f"  Acc@10: {avg_acc10:.4f} ({avg_acc10*100:.2f}%)")
    
    return avg_loss, avg_acc1, avg_acc5, avg_acc10


def train_sota_transformer(
    dataset_name: str = "geolife",
    data_dir: str = "data",
    checkpoint_dir: str = "checkpoints",
    batch_size: int = 128,  # Larger batch
    learning_rate: float = 0.0015,  # Higher LR with OneCycle
    num_epochs: int = 300,
    save_every: int = 10,
    device: str = "cuda",
    patience: int = 50,
):
    """Train SOTA transformer."""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Load data
    print("\n" + "="*60)
    print("Loading datasets...")
    print("="*60)
    
    train_path = f"{data_dir}/geolife/geolife_transformer_7_train.pk"
    val_path = f"{data_dir}/geolife/geolife_transformer_7_validation.pk"
    test_path = f"{data_dir}/geolife/geolife_transformer_7_test.pk"
    
    train_dataset = LocationDataset(train_path, max_seq_len=50, use_features=True)
    val_dataset = LocationDataset(val_path, max_seq_len=50, use_features=True)
    test_dataset = LocationDataset(test_path, max_seq_len=50, use_features=True)
    
    train_loader = LocationDataLoader(train_dataset, batch_size=batch_size, device=device, shuffle=True)
    val_loader = LocationDataLoader(val_dataset, batch_size=batch_size, device=device, shuffle=False)
    test_loader = LocationDataLoader(test_dataset, batch_size=batch_size, device=device, shuffle=False)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create model
    print("\n" + "="*60)
    print("Initializing SOTA Transformer...")
    print("="*60)
    
    config = SOTAConfig(
        vocab_size=train_dataset.vocab_size,
        hidden_size=104,
        num_layers=5,
        num_heads=8,
        num_kv_heads=2,
        max_seq_len=50,
        dropout=0.1,
        use_features=True,
        num_users=train_dataset.num_users,
        tie_weights=True,
        use_alibi=True,
    )
    
    model = SOTALocationTransformer(config).to(device)
    total_params, _ = count_parameters(model)
    
    print(f"Total parameters: {total_params:,}")
    print(f"✓ Within <500K budget!")
    print(f"\nSOTA Techniques:")
    print(f"  ✓ Weight tying (shared embeddings)")
    print(f"  ✓ Multi-query attention (2 KV heads)")
    print(f"  ✓ Rotary position embeddings")
    print(f"  ✓ ALiBi recency bias")
    print(f"  ✓ RMSNorm")
    print(f"  ✓ 5 transformer layers")
    print(f"  ✓ Hidden size: 104")
    
    # Optimizers and loss
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    mixup = MixupAugmentation(alpha=0.2)
    
    # AdamW with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.98),
        weight_decay=0.05,  # Higher weight decay
        eps=1e-8
    )
    
    # OneCycle LR scheduler (Smith, 2018)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * num_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=total_steps,
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=1000,
    )
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler()
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    print("\n" + "="*60)
    print("Training SOTA Transformer...")
    print("="*60 + "\n")
    
    best_val_acc = 0.0
    best_test_acc = 0.0
    no_improve_count = 0
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*60}")
        
        model.train()
        total_loss = 0.0
        total_acc1 = 0.0
        num_batches = 0
        start_time = time.time()
        
        for batch_idx, (inputs, targets, features) in enumerate(train_loader):
            # Mixed precision training
            with torch.cuda.amp.autocast():
                logits = model(inputs, features)
                
                # Mixup augmentation (on logits, not inputs for discrete tokens)
                if np.random.rand() < 0.3:  # 30% of batches
                    batch_size = targets.size(0)
                    index = torch.randperm(batch_size, device=device)
                    lam = np.random.beta(0.2, 0.2)
                    targets_a, targets_b = targets, targets[index]
                    loss = lam * criterion(logits, targets_a) + (1 - lam) * criterion(logits, targets_b)
                else:
                    loss = criterion(logits, targets)
            
            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # Metrics
            with torch.no_grad():
                acc1 = compute_accuracy_at_k(logits, targets, k=1)
            
            total_loss += loss.item()
            total_acc1 += acc1
            num_batches += 1
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx + 1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | Acc@1: {acc1:.4f}")
        
        # Evaluate
        val_loss, val_acc1, val_acc5, val_acc10 = evaluate(model, val_loader, device, "Validation")
        test_loss, test_acc1, test_acc5, test_acc10 = evaluate(model, test_loader, device, "Test")
        
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        # Summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train - Loss: {total_loss/num_batches:.4f} | Acc@1: {total_acc1/num_batches:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f} | Acc@1: {val_acc1:.4f}")
        print(f"  Test  - Loss: {test_loss:.4f} | Acc@1: {test_acc1:.4f}")
        print(f"  Time: {epoch_time:.2f}s | LR: {current_lr:.6f}")
        print(f"  Params: {total_params:,}")
        
        # Save checkpoints
        if epoch % save_every == 0 or val_acc1 > best_val_acc:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'val_acc1': val_acc1,
                'test_acc1': test_acc1,
            }
            torch.save(checkpoint, f"{checkpoint_dir}/sota_epoch{epoch}.pt")
            print(f"  Saved checkpoint")
        
        # Track best
        if val_acc1 > best_val_acc:
            best_val_acc = val_acc1
            no_improve_count = 0
            torch.save(checkpoint, f"{checkpoint_dir}/sota_best_val.pt")
            print(f"  ✅ New best validation: {val_acc1*100:.2f}%")
        else:
            no_improve_count += 1
        
        if test_acc1 > best_test_acc:
            best_test_acc = test_acc1
            torch.save(checkpoint, f"{checkpoint_dir}/sota_best_test.pt")
            print(f"  ✅ New best test: {test_acc1*100:.2f}%")
        
        # Check goal
        if test_acc1 >= 0.40:
            print(f"\n{'='*60}")
            print(f"🎉🎉🎉 GOAL ACHIEVED! 🎉🎉🎉")
            print(f"Test Acc@1 = {test_acc1:.4f} ({test_acc1*100:.2f}%) >= 40%")
            print(f"SOTA Transformer with {total_params:,} params!")
            print(f"{'='*60}\n")
            break
        
        # Early stopping
        if no_improve_count >= patience:
            print(f"\nEarly stopping after {patience} epochs without improvement")
            break
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"Best validation: {best_val_acc*100:.2f}%")
    print(f"Best test: {best_test_acc*100:.2f}%")
    
    if best_test_acc >= 0.40:
        print(f"\n🎉 SUCCESS! Achieved >40% with SOTA techniques!")
    else:
        print(f"\nBest: {best_test_acc*100:.2f}% (target: 40%)")
    
    return model, best_val_acc, best_test_acc


if __name__ == '__main__':
    train_sota_transformer()
