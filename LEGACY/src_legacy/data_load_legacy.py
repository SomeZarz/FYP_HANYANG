# src/data_load.py

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader

class MemoryMappedSOHDataset(Dataset):
    """
    Lazy-loading dataset from HDF5 for memory-efficient training.
    
    Opens HDF5 file only on first access (thread-safe per worker).
    Returns all modalities as PyTorch tensors.
    """
    
    def __init__(self, hdf5_path: str, indices: Optional[List[int]] = None):
        """
        Args:
            hdf5_path: Path to HDF5 file (from Stage 1 data_process.py)
            indices: Optional list of indices to subset data.
                     If None, uses all samples.
        """
        self.hdf5_path = str(hdf5_path)
        self.file = None  # Will open on first access
        
        # Open temporarily to get length and validate
        with h5py.File(self.hdf5_path, 'r') as f:
            self.total_samples = len(f['soh_labels'])
            
            # Validate dataset structure
            required_keys = ['voltagemaps', 'qhisequences', 'thisequences', 
                           'scalarfeatures', 'soh_labels']
            for key in required_keys:
                if key not in f:
                    raise KeyError(f"Missing required dataset: {key}")
        
        # Set indices
        self.indices = indices if indices is not None else list(range(self.total_samples))
    
    def __len__(self) -> int:
        """Return number of samples (after subsetting if applicable)."""
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and return one sample as a dictionary of tensors.
        
        Args:
            idx: Index in the (potentially subsetted) dataset
            
        Returns:
            Dictionary with keys:
            - voltage_map: (96, 96) float32 tensor
            - qhi_sequence: (151,) float32 tensor
            - thi_sequence: (151,) float32 tensor
            - scalar_features: (15,) float32 tensor
            - soh_label: scalar float32 tensor
            - vehicle_id: string identifier
            - timestamp_start: string (seconds since epoch)
            - timestamp_end: string (seconds since epoch)
        """
        # Lazy open HDF5 (thread-safe for each worker)
        if self.file is None:
            self.file = h5py.File(self.hdf5_path, 'r')
        
        # Get actual index in the full dataset
        actual_idx = self.indices[idx]
        
        # Load all modalities
        sample = {
            'voltage_map': torch.from_numpy(
                self.file['voltagemaps'][actual_idx].astype(np.float32)
            ),
            'qhi_sequence': torch.from_numpy(
                self.file['qhisequences'][actual_idx].astype(np.float32)
            ),
            'thi_sequence': torch.from_numpy(
                self.file['thisequences'][actual_idx].astype(np.float32)
            ),
            'scalar_features': torch.from_numpy(
                self.file['scalarfeatures'][actual_idx].astype(np.float32)
            ),
            'soh_label': torch.tensor(
                self.file['soh_labels'][actual_idx],
                dtype=torch.float32
            ),
        }
        
        # Load metadata (optional, for debugging/analysis)
        try:
            vehicle_id = self.file['vehicle_ids'][actual_idx]
            if isinstance(vehicle_id, bytes):
                vehicle_id = vehicle_id.decode('utf-8')
            sample['vehicle_id'] = vehicle_id
        except:
            sample['vehicle_id'] = 'unknown'
        
        try:
            sample['timestamp_start'] = self.file['timestamps_start'][actual_idx]
            if isinstance(sample['timestamp_start'], bytes):
                sample['timestamp_start'] = sample['timestamp_start'].decode('utf-8')
        except:
            sample['timestamp_start'] = ''
        
        try:
            sample['timestamp_end'] = self.file['timestamps_end'][actual_idx]
            if isinstance(sample['timestamp_end'], bytes):
                sample['timestamp_end'] = sample['timestamp_end'].decode('utf-8')
        except:
            sample['timestamp_end'] = ''
        
        return sample
    
    def __del__(self):
        """Cleanup: close HDF5 file when dataset is garbage collected."""
        if self.file is not None:
            self.file.close()

def get_dataloaders(
    config: Dict[str, Any],
    indices_train: List[int],
    indices_val: List[int],
    indices_test: List[int]
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders with memory-mapped datasets.
    
    Args:
        config: Merged configuration dictionary from utils.load_config()
                Must contain:
                - paths.hdf5path: Path to HDF5 file
                - training.batch_size: Batch size for loaders
        indices_train: List of sample indices for training
        indices_val: List of sample indices for validation
        indices_test: List of sample indices for testing
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    hdf5_path = config['paths']['hdf5path']
    batch_size = config['training']['batch_size']
    
    # Validate HDF5 file exists
    if not Path(hdf5_path).exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
    
    # Create datasets
    train_dataset = MemoryMappedSOHDataset(hdf5_path, indices_train)
    val_dataset = MemoryMappedSOHDataset(hdf5_path, indices_val)
    test_dataset = MemoryMappedSOHDataset(hdf5_path, indices_test)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,  # Use >0 for async loading (adjust based on hardware)
        pin_memory=True,  # Pinned memory for fast GPU transfer
        drop_last=False  # Keep incomplete batches
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader

def create_dataloaders(
    config_path: str = "config.yaml",
    indices_train: Optional[List[int]] = None,
    indices_val: Optional[List[int]] = None,
    indices_test: Optional[List[int]] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders from config file with automatic train/val/test split.
    
    If indices not provided, performs a 70/15/15 split automatically.
    
    Args:
        config_path: Path to config.yaml
        indices_train: Optional pre-defined training indices
        indices_val: Optional pre-defined validation indices
        indices_test: Optional pre-defined test indices
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from .utils import load_config
    
    config = load_config(config_path)
    
    hdf5_path = config['paths']['hdf5path']
    
    # Get total number of samples
    with h5py.File(hdf5_path, 'r') as f:
        total_samples = len(f['soh_labels'])
    
    # Auto-split if indices not provided
    if indices_train is None or indices_val is None or indices_test is None:
        indices_train, indices_val, indices_test = auto_split_indices(
            total_samples,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42
        )
    
    return get_dataloaders(config, indices_train, indices_val, indices_test)

def auto_split_indices(
    total_samples: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Automatically split sample indices into train/val/test sets.
    
    Args:
        total_samples: Total number of samples in dataset
        train_ratio: Fraction for training (default: 0.70)
        val_ratio: Fraction for validation (default: 0.15)
        test_ratio: Fraction for testing (default: 0.15)
        random_seed: Seed for reproducibility (default: 42)
        
    Returns:
        Tuple of (indices_train, indices_val, indices_test)
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Generate shuffled indices
    rng = np.random.RandomState(random_seed)
    all_indices = rng.permutation(total_samples)
    
    # Split
    n_train = int(np.round(total_samples * train_ratio))
    n_val = int(np.round(total_samples * val_ratio))
    # n_test gets the remainder to ensure all samples are used
    
    indices_train = sorted(all_indices[:n_train].tolist())
    indices_val = sorted(all_indices[n_train:n_train + n_val].tolist())
    indices_test = sorted(all_indices[n_train + n_val:].tolist())
    
    return indices_train, indices_val, indices_test

def get_dataset_info(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Get basic information about the HDF5 dataset.
    
    Args:
        config_path: Path to config.yaml
        
    Returns:
        Dictionary with dataset statistics
    """
    from .utils import load_config
    
    config = load_config(config_path)
    hdf5_path = config['paths']['hdf5path']
    
    if not Path(hdf5_path).exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
    
    info = {}
    
    with h5py.File(hdf5_path, 'r') as f:
        info['total_samples'] = len(f['soh_labels'])
        info['voltage_map_shape'] = f['voltagemaps'].shape
        info['qhi_sequence_shape'] = f['qhisequences'].shape
        info['thi_sequence_shape'] = f['thisequences'].shape
        info['scalar_features_shape'] = f['scalarfeatures'].shape
        info['soh_labels_shape'] = f['soh_labels'].shape
        
        # Compute SOH statistics
        soh_values = f['soh_labels'][:]
        info['soh_min'] = float(soh_values.min())
        info['soh_max'] = float(soh_values.max())
        info['soh_mean'] = float(soh_values.mean())
        info['soh_std'] = float(soh_values.std())
        
        # Unique vehicles
        try:
            vehicle_ids = f['vehicle_ids'][:]
            vehicle_ids = [vid.decode('utf-8') if isinstance(vid, bytes) else vid 
                          for vid in vehicle_ids]
            info['unique_vehicles'] = len(set(vehicle_ids))
            info['total_vehicles_sampled'] = len(vehicle_ids)
        except:
            info['unique_vehicles'] = 0
            info['total_vehicles_sampled'] = 0
    
    return info

# ============================================================================
# Test Harness
# ============================================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # Load config
    from src.utils import load_config_for_testing
    
    config = load_config_for_testing()
    
    # Get dataset info
    print("Dataset Information:")
    try:
        info = get_dataset_info("config.yaml")
        for key, value in info.items():
            print(f"  {key}: {value}")
    except FileNotFoundError as e:
        print(f"  HDF5 file not found. Run data_process.py first.")
        sys.exit(1)
    
    # Try to create dataloaders
    print("\nCreating dataloaders...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders("config.yaml")
        
        print(f"  Train loader: {len(train_loader)} batches")
        print(f"  Val loader: {len(val_loader)} batches")
        print(f"  Test loader: {len(test_loader)} batches")
        
        # Load one sample
        print("\nLoading one sample from train set...")
        sample = next(iter(train_loader))
        
        print(f"  voltage_map shape: {sample['voltage_map'].shape}")
        print(f"  qhi_sequence shape: {sample['qhi_sequence'].shape}")
        print(f"  thi_sequence shape: {sample['thi_sequence'].shape}")
        print(f"  scalar_features shape: {sample['scalar_features'].shape}")
        print(f"  soh_label shape: {sample['soh_label'].shape}")
        print(f"  vehicle_id: {sample['vehicle_id']}")
        
        # Verify tensor device and dtype
        print(f"\n  voltage_map dtype: {sample['voltage_map'].dtype}, device: {sample['voltage_map'].device}")
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
