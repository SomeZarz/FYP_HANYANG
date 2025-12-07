import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, Any, Tuple
import numpy as np

from .utils import load_config
from .data_load import create_dataloaders
from .models import build_model

# ==================== Metric Functions ====================
def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error"""
    return torch.mean((pred - target)  ** 2)

def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error"""
    return torch.mean(torch.abs(pred - target))

def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Root Mean Squared Error"""
    return torch.sqrt(torch.mean((pred - target)  ** 2))

def compute_mape(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean Absolute Percentage Error"""
    return torch.mean(torch.abs(pred - target) / (torch.abs(target) + eps)) * 100

# ==================== Training Epoch ====================
def train_epoch(
    model: torch.nn.Module, 
    loader: torch.utils.data.DataLoader, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device,
    grad_clip_norm: float = None
) -> float:
    """
    Train model for one epoch.
    
    Args:
        model: Model to train
        loader: Training data loader
        optimizer: Optimizer instance
        device: Computation device
        grad_clip_norm: Gradient clipping norm value (optional)
    
    Returns:
        Average training loss (MSE)
    """
    model.train()
    total_loss = 0.0
    num_batches = len(loader)
    
    with tqdm(loader, desc="Training", leave=False) as pbar:
        for batch_idx, batch in enumerate(pbar):
            # Prepare inputs
            voltage_map = batch['voltage_map'].to(device, non_blocking=True)
            qhi = batch['qhi_sequence'].to(device, non_blocking=True)
            thi = batch['thi_sequence'].to(device, non_blocking=True)
            scalar_features = batch['scalar_features'].to(device, non_blocking=True)
            soh = batch['soh_label'].to(device, non_blocking=True)
            
            # Forward pass
            pred = model(voltage_map, qhi, thi, scalar_features)
            
            # Compute loss (MSE for training)
            loss = F.mse_loss(pred, soh)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping if specified
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            
            optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches

# ==================== Validation ====================
def validate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model on validation set.
    
    Returns:
        Dictionary with MSE, MAE, RMSE, MAPE metrics
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        with tqdm(loader, desc="Validating", leave=False) as pbar:
            for batch in pbar:
                # Prepare inputs
                voltage_map = batch['voltage_map'].to(device, non_blocking=True)
                qhi = batch['qhi_sequence'].to(device, non_blocking=True)
                thi = batch['thi_sequence'].to(device, non_blocking=True)
                scalar_features = batch['scalar_features'].to(device, non_blocking=True)
                soh = batch['soh_label'].to(device, non_blocking=True)
                
                # Forward pass
                pred = model(voltage_map, qhi, thi, scalar_features)
                
                # Store predictions and targets
                all_preds.append(pred.cpu())
                all_targets.append(soh.cpu())
    
    # Concatenate all predictions and targets
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    
    # Compute metrics
    metrics = {
        'mse': compute_mse(preds, targets).item(),
        'mae': compute_mae(preds, targets).item(),
        'rmse': compute_rmse(preds, targets).item(),
        'mape': compute_mape(preds, targets).item()
    }
    
    return metrics

# ==================== Full Training Loop ====================
# Add this import at the top of train.py
from torch.optim.lr_scheduler import LambdaLR

# In train_model() function, replace the optimizer setup:

def train_model(model, train_loader, val_loader, config, device, checkpoint_dir=None):
    """
    Full training loop with warm-up + exponential decay LR schedule.
    """
    # Extract training configuration
    train_config = config.get('training', {})
    num_epochs = train_config.get('num_epochs', 100)
    learning_rate = train_config.get('learning_rate', 1.0e-3)
    weight_decay = train_config.get('weight_decay', 1.0e-5)
    patience = train_config.get('patience', 10)
    grad_clip_norm = train_config.get('grad_clip_norm', None)
    warmup_epochs = train_config.get('warmup_epochs', 10)  # Paper specification
    
    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    
    # === NEW: Learning rate schedule with warm-up + exponential decay ===
    def lr_lambda(current_epoch):
        """Learning rate schedule: linear warm-up then exponential decay."""
        if current_epoch < warmup_epochs:
            # Linear warm-up: increase from 0 to 1 over warmup_epochs
            return (current_epoch + 1) / warmup_epochs
        else:
            # Exponential decay: decay by factor 0.95 every epoch after warm-up
            decay_factor = 0.95 ** (current_epoch - warmup_epochs)
            return decay_factor
    
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    # Training loop
    best_mape = float('inf')
    patience_counter = 0
    best_model_state = None
    
    if checkpoint_dir is None:
        checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Training for {num_epochs} epochs with patience {patience}")
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"LR Schedule: {warmup_epochs} epoch warm-up + exponential decay (γ=0.95)")
    
    for epoch in range(1, num_epochs + 1):
        # Training phase
        train_loss = train_epoch(model, train_loader, optimizer, device, grad_clip_norm)
        
        # Update learning rate
        scheduler.step()  # Step the scheduler
        
        # Validation phase
        val_metrics = validate(model, val_loader, device)
        
        # Print epoch summary with LR
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{num_epochs} | "
              f"Train MSE: {train_loss:.4f} | "
              f"Val MSE: {val_metrics['mse']:.4f} | "
              f"Val MAPE: {val_metrics['mape']:.2f} | "
              f"Val RMSE: {val_metrics['rmse']:.2f} | "
              f"LR: {current_lr:.6f}")
        
        # Early stopping check
        if val_metrics['mape'] < best_mape:
            best_mape = val_metrics['mape']
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }
            torch.save(checkpoint, checkpoint_dir / 'best_model.pt')
            print(f"  New best model saved (MAPE: {best_mape:.2f})")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"  Early stopping triggered after {patience} epochs without improvement")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Training completed. Best validation MAPE: {best_mape:.2f}")
    else:
        print("No best model found. Returning last model state.")
    
    return model

# ==================== Evaluation Function ====================
def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    return_attention: bool = False
) -> Dict[str, Any]:
    """
    Comprehensive evaluation on test set with optional attention analysis.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Computation device
        return_attention: Whether to return modality attention scores
    
    Returns:
        Dictionary with metrics and optional attention scores
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_attention = [] if return_attention else None
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Prepare inputs
            voltage_map = batch['voltage_map'].to(device, non_blocking=True)
            qhi = batch['qhi_sequence'].to(device, non_blocking=True)
            thi = batch['thi_sequence'].to(device, non_blocking=True)
            scalar_features = batch['scalar_features'].to(device, non_blocking=True)
            soh = batch['soh_label'].to(device, non_blocking=True)
            
            # Forward pass
            if return_attention:
                pred, att = model(voltage_map, qhi, thi, scalar_features, return_attention=True)
                all_attention.append(att.cpu())
            else:
                pred = model(voltage_map, qhi, thi, scalar_features)
            
            all_preds.append(pred.cpu())
            all_targets.append(soh.cpu())
    
    # Concatenate
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    if return_attention:
        attention = torch.cat(all_attention)
    
    # Compute metrics
    results = {
        'mse': compute_mse(preds, targets).item(),
        'mae': compute_mae(preds, targets).item(),
        'rmse': compute_rmse(preds, targets).item(),
        'mape': compute_mape(preds, targets).item(),
        'predictions': preds.numpy(),
        'targets': targets.numpy()
    }
    
    if return_attention:
        results['attention_scores'] = attention.numpy()
    
    # Print summary
    print("\n" + "="*50)
    print("TEST SET EVALUATION")
    print("="*50)
    print(f"MSE:  {results['mse']:.6f}")
    print(f"MAE:  {results['mae']:.6f}")
    print(f"RMSE: {results['rmse']:.6f}")
    print(f"MAPE: {results['mape']:.2f}%")
    
    if return_attention:
        print(f"\nAverage modality attention:")
        print(f"  Voltage Map: {attention[:, 0].mean():.3f}")
        print(f"  Sequence:    {attention[:, 1].mean():.3f}")
        print(f"  Point Features: {attention[:, 2].mean():.3f}")
    
    return results

def model_pipeline(
    config_path: str = "config.yaml",
    return_attention: bool = True
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    End-to-end training + evaluation pipeline.

    Steps:
      1. Load config
      2. Create dataloaders
      3. Select device (XPU / MPS / CPU)
      4. Build model from config
      5. Train model
      6. Evaluate on test set

    Returns:
      trained_model: torch.nn.Module
      test_results: dict with metrics (and optionally attention scores)
    """
    # 1. Load configuration
    config = load_config(config_path)
    print(f"Data profile: {config['orchestration']['model_profile']}")
    print(f"Data profile: {config['orchestration']['data_profile']}")

    # 2. Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(config_path)

    # 3. Setup device (XPU → MPS → CPU)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # 4. Build model (architecture selected from config)
    model = build_model(config)
    model = model.to(device)

    # 5. Train model
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    # 6. Evaluate on test set
    test_results = evaluate_model(
        trained_model,
        test_loader,
        device,
        return_attention=return_attention,
    )

    final_model_path = Path(config['paths']['checkpoint_dir']) / Path(config['orchestration']['model_profile'])
    
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': config,
        'test_metrics': test_results,
        'architecture': config['model'].get('architecture', 'resnet')
    }, final_model_path)
    
    print(f"\n✓ Final model saved to: {final_model_path}")

    return trained_model, test_results
