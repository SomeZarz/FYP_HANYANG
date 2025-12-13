# src/data_load.py

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


class MemoryMappedSOHDataset(Dataset):
    """
    Lazy-loading dataset with BRUTE-FORCE normalization.
    Computes global stats from entire HDF5 file once at initialization.
    """
    def __init__(self, hdf5_path: str, indices: Optional[List[int]] = None):
        self.hdf5_path = str(hdf5_path)
        self.file = None

        # Compute global statistics ONCE for normalization
        with h5py.File(self.hdf5_path, "r") as f:
            self.total_samples = len(f["soh_labels"])
            
            print(f"Computing global stats from {self.total_samples} samples...")
            
            # Load entire datasets to compute stats (this is brute-force but works)
            vm = f["voltage_maps"][:]
            qhi = f["qhi_sequences"][:]
            thi = f["thi_sequences"][:]
            sf = f["scalar_features"][:]
            soh = f["soh_labels"][:]
            
            # Voltage maps: min-max range
            self.vm_min = float(vm.min())
            self.vm_max = float(vm.max())
            
            # QHI/THI: global mean/std for standardization
            self.qhi_mean = float(qhi.mean())
            self.qhi_std = float(qhi.std())
            self.thi_mean = float(thi.mean())
            self.thi_std = float(thi.std())
            
            # Scalar features: per-feature mean/std
            self.sf_mean = sf.mean(axis=0).astype(np.float32)
            self.sf_std = sf.std(axis=0).astype(np.float32)
            
            # SOH: max value to force scaling to [0,1]
            self.soh_max = float(soh.max())
            
            print(f"Voltage map range: [{self.vm_min:.3f}, {self.vm_max:.3f}]")
            print(f"QHI stats: mean={self.qhi_mean:.3f}, std={self.qhi_std:.3f}")
            print(f"THI stats: mean={self.thi_mean:.3f}, std={self.thi_std:.3f}")
            print(f"SOH max: {self.soh_max:.3f}")
            
            # Clamp std to avoid division by zero
            self.qhi_std = max(self.qhi_std, 1e-4)
            self.thi_std = max(self.thi_std, 1e-4)
            self.sf_std = np.maximum(self.sf_std, 1e-4)

            required_keys = [
                "voltage_maps",
                "qhi_sequences",
                "thi_sequences",
                "scalar_features",
                "soh_labels",
            ]
            for key in required_keys:
                if key not in f:
                    raise KeyError(f"Missing required dataset: {key}")

        self.indices = indices if indices is not None else list(range(self.total_samples))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.file is None:
            self.file = h5py.File(self.hdf5_path, "r")

        actual_idx = self.indices[idx]

        # Load raw data
        voltage_map = torch.from_numpy(
            self.file["voltage_maps"][actual_idx].astype(np.float32)
        )
        qhi = torch.from_numpy(
            self.file["qhi_sequences"][actual_idx].astype(np.float32)
        )
        thi = torch.from_numpy(
            self.file["thi_sequences"][actual_idx].astype(np.float32)
        )
        scalar_features = torch.from_numpy(
            self.file["scalar_features"][actual_idx].astype(np.float32)
        )
        soh_label = torch.tensor(
            self.file["soh_labels"][actual_idx], dtype=torch.float32
        )

        # === BRUTE-FORCE NORMALIZATION ===
        
        # Voltage map: min-max normalize using GLOBAL stats
        voltage_map = (voltage_map - self.vm_min) / (self.vm_max - self.vm_min)
        voltage_map = torch.clamp(voltage_map, 0.0, 1.0)  # Force [0,1]
        
        # QHI/THI: standardize using GLOBAL stats
        qhi = (qhi - self.qhi_mean) / self.qhi_std
        thi = (thi - self.thi_mean) / self.thi_std
        
        # Scalar features: standardize using GLOBAL stats
        scalar_features = (scalar_features - self.sf_mean) / self.sf_std
        
        # SOH: force to [0,1] range
        if self.soh_max > 1.0:
            soh_label = soh_label / self.soh_max
        soh_label = torch.clamp(soh_label, 0.0, 1.0)

        sample = {
            "voltage_map": voltage_map,
            "qhi_sequence": qhi,
            "thi_sequence": thi,
            "scalar_features": scalar_features,
            "soh_label": soh_label,
        }

        # Optional metadata
        try:
            vehicle_id = self.file["vehicle_ids"][actual_idx]
            if isinstance(vehicle_id, bytes):
                vehicle_id = vehicle_id.decode("utf-8")
            sample["vehicle_id"] = vehicle_id
        except Exception:
            sample["vehicle_id"] = "unknown"

        try:
            ts_start = self.file["timestamps_start"][actual_idx]
            if isinstance(ts_start, bytes):
                ts_start = ts_start.decode("utf-8")
            sample["timestamp_start"] = ts_start
        except Exception:
            sample["timestamp_start"] = ""

        try:
            ts_end = self.file["timestamps_end"][actual_idx]
            if isinstance(ts_end, bytes):
                ts_end = ts_end.decode("utf-8")
            sample["timestamp_end"] = ts_end
        except Exception:
            sample["timestamp_end"] = ""

        return sample

    def __del__(self):
        if getattr(self, "file", None) is not None:
            try:
                self.file.close()
            except Exception:
                pass


def _resolve_hdf5_path(config: Dict[str, Any]) -> Path:
    return Path(config["paths"]["extracted_datadir"]) / config["paths"]["hdf5_filename"]


def get_dataloaders(
    config: Dict[str, Any],
    indices_train: List[int],
    indices_val: List[int],
    indices_test: List[int],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    hdf5_path = _resolve_hdf5_path(config)
    batch_size = config["training"]["batch_size"]

    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    # For tiny datasets, set batch_size to min of requested and train size
    train_size = len(indices_train)
    actual_batch_size = min(batch_size, train_size)
    if actual_batch_size != batch_size:
        print(f"WARNING: Reducing batch_size from {batch_size} to {actual_batch_size} (train size: {train_size})")

    train_dataset = MemoryMappedSOHDataset(str(hdf5_path), indices_train)
    val_dataset = MemoryMappedSOHDataset(str(hdf5_path), indices_val)
    test_dataset = MemoryMappedSOHDataset(str(hdf5_path), indices_test)

    # Disable multiprocessing and pin_memory for tiny datasets on MPS
    train_loader = DataLoader(
        train_dataset,
        batch_size=actual_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=actual_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=actual_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


def create_dataloaders(
    config_path: str = "config.yaml",
    indices_train: Optional[List[int]] = None,
    indices_val: Optional[List[int]] = None,
    indices_test: Optional[List[int]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    from .utils import load_config

    config = load_config(config_path)
    hdf5_path = _resolve_hdf5_path(config)

    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as f:
        total_samples = len(f["soh_labels"])

    if indices_train is None or indices_val is None or indices_test is None:
        indices_train, indices_val, indices_test = auto_split_indices(
            total_samples,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,
        )

    return get_dataloaders(config, indices_train, indices_val, indices_test)


def auto_split_indices(
    total_samples: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    rng = np.random.RandomState(random_seed)
    all_indices = rng.permutation(total_samples)

    n_train = int(np.round(total_samples * train_ratio))
    n_val = int(np.round(total_samples * val_ratio))

    indices_train = sorted(all_indices[:n_train].tolist())
    indices_val = sorted(all_indices[n_train : n_train + n_val].tolist())
    indices_test = sorted(all_indices[n_train + n_val :].tolist())

    return indices_train, indices_val, indices_test
