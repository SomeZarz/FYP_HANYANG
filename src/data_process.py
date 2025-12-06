# src/data_process.py

import os
import h5py
import numpy as np
from pathlib import Path
import json
import pandas as pd
from tqdm.auto import tqdm
from typing import Dict, Any, List

def get_vehicle_paths(csv_dir: Path, pattern: str = "*.csv") -> List[Path]:
    """Enumerate and sort all vehicle CSV files."""
    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")
    
    return sorted(csv_dir.glob(pattern))

def is_hdf5_corrupted(hdf5_path: Path) -> bool:
    """Test if HDF5 file is corrupted by attempting to open it."""
    if not hdf5_path.exists():
        return False
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            f.keys()  # Try to access root group
        return False
    except OSError:
        return True

def setup_hdf5(hdf5_path: Path, config: Dict[str, Any]) -> h5py.File:
    """
    Create HDF5 file with corruption detection and auto-repair.
    Returns opened h5py.File object.
    """
    # Check for corruption
    if is_hdf5_corrupted(hdf5_path):
        print(f"⚠ Corrupted HDF5 detected: {hdf5_path}")
        print(f" Deleting and recreating...")
        hdf5_path.unlink(missing_ok=True)
        mode = 'w'
    else:
        if hdf5_path.exists():
            # Verify all required datasets exist and are readable
            try:
                with h5py.File(hdf5_path, 'r') as test_f:
                    required_datasets = [
                        'voltagemaps', 'qhisequences', 'thisequences',
                        'scalarfeatures', 'soh_labels', 'vehicle_ids',
                        'timestamps_start', 'timestamps_end'
                    ]
                    missing = [ds for ds in required_datasets if ds not in test_f]
                    if missing:
                        print(f"⚠ Existing HDF5 missing datasets {missing}: {hdf5_path}")
                        print(f" Recreating file...")
                        mode = 'w'
                    else:
                        # Try reading a small slice to verify integrity
                        for ds_name in required_datasets:
                            ds = test_f[ds_name]
                            _ = ds[0:1]  # Test read
                        mode = 'a'
            except Exception as e:
                print(f"⚠ HDF5 verification failed: {e}")
                print(f" Recreating file...")
                mode = 'w'
        else:
            mode = 'w'
    
    # Ensure parent directory exists
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Open file
    try:
        hdf5_file = h5py.File(hdf5_path, mode)
    except OSError as e:
        raise RuntimeError(f"Failed to open HDF5 file: {hdf5_path}") from e
    
    # Initialize datasets if new file
    if mode == 'w':
        print(f"Creating new HDF5 file: {hdf5_path}")
        
        # Get shapes from config
        num_cells = config['dataset']['num_cells']
        seq_length = 151  # As per paper specification
        scalar_dim = 15   # As per paper specification
        
        hdf5_file.create_dataset(
            'voltagemaps', shape=(0, num_cells, num_cells),
            maxshape=(None, num_cells, num_cells),
            compression='gzip', dtype=np.float32, chunks=(32, num_cells, num_cells)
        )
        
        hdf5_file.create_dataset(
            'qhisequences', shape=(0, seq_length),
            maxshape=(None, seq_length),
            compression='gzip', dtype=np.float32, chunks=(32, seq_length)
        )
        
        hdf5_file.create_dataset(
            'thisequences', shape=(0, seq_length),
            maxshape=(None, seq_length),
            compression='gzip', dtype=np.float32, chunks=(32, seq_length)
        )
        
        hdf5_file.create_dataset(
            'scalarfeatures', shape=(0, scalar_dim),
            maxshape=(None, scalar_dim),
            compression='gzip', dtype=np.float32, chunks=(32, scalar_dim)
        )
        
        hdf5_file.create_dataset(
            'soh_labels', shape=(0,), maxshape=(None,),
            compression='gzip', dtype=np.float32, chunks=(32,)
        )
        
        hdf5_file.create_dataset(
            'vehicle_ids', shape=(0,), maxshape=(None,),
            compression='gzip', dtype=h5py.string_dtype()
        )
        
        hdf5_file.create_dataset(
            'timestamps_start', shape=(0,), maxshape=(None,),
            compression='gzip', dtype=h5py.string_dtype()
        )
        
        hdf5_file.create_dataset(
            'timestamps_end', shape=(0,), maxshape=(None,),
            compression='gzip', dtype=h5py.string_dtype()
        )
    
    return hdf5_file

def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Load or initialize checkpoint."""
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"⚠ Corrupted checkpoint: {checkpoint_path}")
            checkpoint_path.unlink()
    
    return {
        'last_vehicle': None,
        'total_samples': 0,
        'processed_vehicles': 0,
        'failed_vehicles': []
    }

def save_checkpoint(checkpoint_path: Path, checkpoint: Dict[str, Any]):
    """Atomically save checkpoint to disk."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to temp file then atomic rename
    temp_path = checkpoint_path.with_suffix('.tmp')
    with open(temp_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    temp_path.replace(checkpoint_path)

def process_vehicle(csv_path: Path, config: Dict[str, Any], hdf5_file: h5py.File) -> Dict[str, Any]:
    """Process one vehicle CSV and append features to HDF5."""
    vehicle_id = csv_path.stem
    
    try:
        # Load and prepare data
        df = pd.read_csv(csv_path)
        df['vehicle_id'] = vehicle_id
        
        # Import here to avoid circular dependency
        from . import data_extract
        
        # Extract all features
        features = data_extract.extract_all_features(df, config)
        
        # Handle no valid segments
        if features is None:
            return {
                'vehicle_id': vehicle_id,
                'samples_extracted': 0,
                'status': 'no_valid_segments'
            }
        
        # Get number of new samples
        n_new = len(features['soh_labels'])
        
        # Append to HDF5 datasets
        for key in ['voltagemaps', 'qhisequences', 'thisequences', 'scalarfeatures', 'soh_labels']:
            hdf5_file[key].resize(hdf5_file[key].shape[0] + n_new, axis=0)
            hdf5_file[key][-n_new:] = features[key]
        
        # Append metadata
        for key in ['vehicle_ids', 'timestamps_start', 'timestamps_end']:
            ds = hdf5_file[key]
            ds.resize(ds.shape[0] + n_new, axis=0)
            ds[-n_new:] = features[key]
        
        hdf5_file.flush()
        
        return {
            'vehicle_id': vehicle_id,
            'samples_extracted': n_new,
            'status': 'success'
        }
        
    except Exception as e:
        print(f"❌ Error in {vehicle_id}: {str(e)}")
        return {
            'vehicle_id': vehicle_id,
            'samples_extracted': 0,
            'status': f'error: {str(e)}'
        }

def extraction_pipeline(config_path: str = "config.yaml"):
    """... existing docstring ..."""
    from .utils import load_config
    
    config = load_config(config_path)
    print(f"Data profile: {config['orchestration']['data_profile']}\n")

    csv_paths = get_vehicle_paths(config['paths']['csv_dir'])
    print(f"Found {len(csv_paths)} vehicle files")
    
    # Setup checkpoint first
    checkpoint = load_checkpoint(config['paths']['checkpoint_path'])
    
    # Setup HDF5 (with corruption detection)
    hdf5_file = setup_hdf5(config['paths']['hdf5path'], config)
    
    vehicle_batch_size = config['processing'].get('vehicle_batch_size', 1)
    
    # Initialize counters for summary
    stats = {
        'processed': 0,
        'success': 0,
        'no_valid_segments': 0,
        'errors': 0,
        'failed_vehicles': []
    }
    
    try:
        # Determine starting point
        if checkpoint['last_vehicle']:
            start_idx = next((i for i, p in enumerate(csv_paths) 
                            if p.name == checkpoint['last_vehicle']), -1) + 1
        else:
            start_idx = 0
        
        # Process vehicles
        for csv_path in tqdm(csv_paths[start_idx:], desc="Processing"):
            try:
                stats['processed'] += 1
                result = process_vehicle(csv_path, config, hdf5_file)
                
                # Update checkpoint
                checkpoint['last_vehicle'] = csv_path.name
                checkpoint['total_samples'] += result['samples_extracted']
                checkpoint['processed_vehicles'] += 1
                
                # Update stats
                if result['status'] == 'success':
                    stats['success'] += 1
                elif result['status'] == 'no_valid_segments':
                    stats['no_valid_segments'] += 1
                else:
                    stats['errors'] += 1
                    stats['failed_vehicles'].append(csv_path.name)
                    checkpoint['failed_vehicles'].append(csv_path.name)
                
                # Periodic checkpoint save
                if checkpoint['processed_vehicles'] % vehicle_batch_size == 0:
                    save_checkpoint(config['paths']['checkpoint_path'], checkpoint)
                
            except Exception as e:
                print(f"❌ Error in {csv_path.name}: {e}")
                stats['errors'] += 1
                stats['failed_vehicles'].append(csv_path.name)
                checkpoint['failed_vehicles'].append(csv_path.name)
                
                if config['processing'].get('fail_on_error', False):
                    raise
        
        # Final save
        save_checkpoint(config['paths']['checkpoint_path'], checkpoint)
        
        # Print summary instead of per-vehicle messages
        print(f"\n{'='*50}")
        print(f"✅ Processing Complete")
        print(f"{'='*50}")
        print(f"Total vehicles processed: {stats['processed']}")
        print(f"Successful extractions: {stats['success']}")
        print(f"No valid segments: {stats['no_valid_segments']}")
        print(f"Errors: {stats['errors']}")
        print(f"Total samples extracted: {checkpoint['total_samples']}")
        if stats['failed_vehicles']:
            print(f"Failed vehicles: {stats['failed_vehicles']}")
        print(f"{'='*50}")
        
    finally:
        # CRITICAL: Always close HDF5 file, even on crash
        hdf5_file.close()
        print("HDF5 file closed safely")


def cleanup_corrupted_files(config_path: str = "config.yaml"):
    """Delete corrupted HDF5 and checkpoint files."""
    from .utils import load_config
    
    config = load_config(config_path)
    
    files_to_remove = [
        config['paths']['hdf5path'],
        config['paths']['checkpoint_path']
    ]
    
    for path in files_to_remove:
        if path.exists():
            try:
                # Test if HDF5 is corrupted before deleting
                if path.suffix == '.h5' and not is_hdf5_corrupted(path):
                    print(f"✓ File OK, skipping: {path}")
                    continue
                
                path.unlink()
                print(f"🗑 Deleted: {path}")
            except Exception as e:
                print(f"⚠ Could not delete {path}: {e}")
    
    print("Cleanup complete. You can now re-run the pipeline.")
