# src/train.py

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, Any, Tuple

from .utils import load_config
from .data_load import create_dataloaders
from .models import build_model


def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2))


def compute_mape(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target) / (torch.abs(target) + eps)) * 100


def train_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float = None,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = len(loader)

    with tqdm(loader, desc="Training", leave=False) as pbar:
        for batch in pbar:
            voltage_map = batch["voltage_map"].to(device, non_blocking=True)
            qhi = batch["qhi_sequence"].to(device, non_blocking=True)
            thi = batch["thi_sequence"].to(device, non_blocking=True)
            scalar_features = batch["scalar_features"].to(device, non_blocking=True)
            soh = batch["soh_label"].to(device, non_blocking=True)

            # === BRUTE FORCE: Add noise to prevent overfitting on tiny data ===
            if torch.rand(1).item() < 0.3:  # 30% chance
                voltage_map += torch.randn_like(voltage_map) * 0.01
                qhi += torch.randn_like(qhi) * 0.1
                thi += torch.randn_like(thi) * 0.1
                scalar_features += torch.randn_like(scalar_features) * 0.1

            # === BRUTE FORCE: Clamp inputs to prevent explosions ===
            voltage_map = torch.clamp(voltage_map, -5.0, 5.0)
            qhi = torch.clamp(qhi, -5.0, 5.0)
            thi = torch.clamp(thi, -5.0, 5.0)
            scalar_features = torch.clamp(scalar_features, -5.0, 5.0)

            # Debug: print ranges in first batch
            if pbar.n == 0:
                print(
                    f"[DEBUG] voltage_map: [{voltage_map.min():.3f}, {voltage_map.max():.3f}], "
                    f"qhi: [{qhi.min():.3f}, {qhi.max():.3f}], "
                    f"soh: [{soh.min():.3f}, {soh.max():.3f}]"
                )

            pred = model(voltage_map, qhi, thi, scalar_features)

            loss = F.mse_loss(pred, soh)
            optimizer.zero_grad()
            loss.backward()

            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / num_batches


def validate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        with tqdm(loader, desc="Validating", leave=False) as pbar:
            for batch in pbar:
                voltage_map = batch["voltage_map"].to(device, non_blocking=True)
                qhi = batch["qhi_sequence"].to(device, non_blocking=True)
                thi = batch["thi_sequence"].to(device, non_blocking=True)
                scalar_features = batch["scalar_features"].to(device, non_blocking=True)
                soh = batch["soh_label"].to(device, non_blocking=True)

                # Clamp inputs
                voltage_map = torch.clamp(voltage_map, -5.0, 5.0)
                qhi = torch.clamp(qhi, -5.0, 5.0)
                thi = torch.clamp(thi, -5.0, 5.0)
                scalar_features = torch.clamp(scalar_features, -5.0, 5.0)

                pred = model(voltage_map, qhi, thi, scalar_features)
                all_preds.append(pred.cpu())
                all_targets.append(soh.cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    metrics = {
        "mse": compute_mse(preds, targets).item(),
        "mae": compute_mae(preds, targets).item(),
        "rmse": compute_rmse(preds, targets).item(),
        "mape": compute_mape(preds, targets).item(),
    }

    if metrics["mape"] > 50.0:
        print(f"\n⚠️  WARNING: MAPE is {metrics['mape']:.2f}% - check data normalization!")

    return metrics


def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    checkpoint_dir: Path = None,
) -> torch.nn.Module:
    train_cfg = config.get("training", {})
    num_epochs = train_cfg.get("num_epochs", 100)
    learning_rate = train_cfg.get("learning_rate", 5.0e-5)
    weight_decay = train_cfg.get("weight_decay", 1.0e-3)
    patience = train_cfg.get("patience", 20)
    grad_clip_norm = train_cfg.get("grad_clip_norm", 1.0)
    warmup_epochs = train_cfg.get("warmup_epochs", 5)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    def lr_lambda(current_epoch: int) -> float:
        if current_epoch < warmup_epochs:
            return (current_epoch + 1) / warmup_epochs
        else:
            decay_factor = 0.95 ** (current_epoch - warmup_epochs)
            return decay_factor

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    if checkpoint_dir is None:
        checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_mape = float("inf")
    patience_counter = 0
    best_model_state = None

    print(f"Training for {num_epochs} epochs with patience={patience}")
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"LR schedule: {warmup_epochs} epoch warm-up + exponential decay (γ=0.95)")

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, device, grad_clip_norm
        )
        scheduler.step()
        val_metrics = validate(model, val_loader, device)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"Train MSE: {train_loss:.4f} | "
            f"Val MSE: {val_metrics['mse']:.4f} | "
            f"Val MAPE: {val_metrics['mape']:6.2f}% | "
            f"Val RMSE: {val_metrics['rmse']:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        if val_metrics["mape"] < best_mape:
            best_mape = val_metrics["mape"]
            patience_counter = 0
            best_model_state = model.state_dict().copy()

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": best_model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_metrics": val_metrics,
                "config": config,
            }
            torch.save(checkpoint, checkpoint_dir / "best_model.pt")
            print(f"   ✓ Best model saved (MAPE: {best_mape:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"\n   Early stopping triggered: {patience} epochs without improvement"
                )
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nTraining complete. Best validation MAPE: {best_mape:.2f}%")
    else:
        print("\nWarning: No best model found. Using last state.")

    return model


def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    return_attention: bool = False,
) -> Dict[str, Any]:
    model.eval()
    all_preds = []
    all_targets = []
    all_attention = [] if return_attention else None

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            voltage_map = batch["voltage_map"].to(device, non_blocking=True)
            qhi = batch["qhi_sequence"].to(device, non_blocking=True)
            thi = batch["thi_sequence"].to(device, non_blocking=True)
            scalar_features = batch["scalar_features"].to(device, non_blocking=True)
            soh = batch["soh_label"].to(device, non_blocking=True)

            # Clamp inputs
            voltage_map = torch.clamp(voltage_map, -5.0, 5.0)
            qhi = torch.clamp(qhi, -5.0, 5.0)
            thi = torch.clamp(thi, -5.0, 5.0)
            scalar_features = torch.clamp(scalar_features, -5.0, 5.0)

            if return_attention:
                pred, att = model(
                    voltage_map, qhi, thi, scalar_features, return_attention=True
                )
                all_attention.append(att.cpu())
            else:
                pred = model(voltage_map, qhi, thi, scalar_features)

            all_preds.append(pred.cpu())
            all_targets.append(soh.cpu())

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    # === BRUTE FORCE: Clamp predictions to [0,1] ===
    preds = np.clip(preds, 0.0, 1.0)

    results = {
        "mse": compute_mse(torch.from_numpy(preds), torch.from_numpy(targets)).item(),
        "mae": compute_mae(torch.from_numpy(preds), torch.from_numpy(targets)).item(),
        "rmse": compute_rmse(torch.from_numpy(preds), torch.from_numpy(targets)).item(),
        "mape": compute_mape(torch.from_numpy(preds), torch.from_numpy(targets)).item(),
        "predictions": preds,
        "targets": targets,
    }

    if return_attention:
        results["attention_scores"] = torch.cat(all_attention).numpy()

    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    print(f"MSE:  {results['mse']:.6f}")
    print(f"MAE:  {results['mae']:.6f}")
    print(f"RMSE: {results['rmse']:.6f}")
    print(f"MAPE: {results['mape']:.2f}%")
    if results["mape"] > 50.0:
        print("⚠️  MAPE is very high - check data normalization!")
    if return_attention:
        att = results["attention_scores"]
        print(f"\nAverage Modality Attention Weights:")
        print(f"  Voltage Map:    {att[:, 0].mean():.3f}")
        print(f"  Sequence (Q/T): {att[:, 1].mean():.3f}")
        print(f"  Point Features: {att[:, 2].mean():.3f}")
    print("=" * 60)

    return results


def model_pipeline(
    config_path: str = "config.yaml",
    return_attention: bool = True,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    config = load_config(config_path)
    print(f"Data profile: {config['data_profile']}")
    print(f"Dataset: {config['dataset_name']}")
    print(f"Model architecture: {config.get('model', {}).get('architecture', 'resnet')}")

    train_loader, val_loader, test_loader = create_dataloaders(config_path)
    print(
        f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}"
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    model = build_model(config)
    model = model.to(device)

    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    test_results = evaluate_model(
        trained_model,
        test_loader,
        device,
        return_attention=return_attention,
    )

    final_model_path = (
        Path(config["paths"]["checkpoint_dir"])
        / f"{config['data_profile']}_{config['dataset_name']}_final.pt"
    )
    torch.save(
        {
            "model_state_dict": trained_model.state_dict(),
            "config": config,
            "test_metrics": {
                k: v
                for k, v in test_results.items()
                if k not in ["predictions", "targets", "attention_scores"]
            },
            "architecture": config.get("model", {}).get("architecture", "resnet"),
        },
        final_model_path,
    )
    print(f"\n✓ Final model saved to: {final_model_path}")

    return trained_model, test_results
