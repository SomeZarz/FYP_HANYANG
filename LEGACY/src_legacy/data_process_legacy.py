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

def process_vehicle(csv_path: Path, config: Dict[str, Any], hdf5_file=None) -> Dict[str, Any]:
    """Process one vehicle CSV and return features dict."""
    vehicle_id = csv_path.stem
    
    try:
        # Load and prepare data
        df = pd.read_csv(csv_path)
        df['vehicle_id'] = vehicle_id
        
        from . import data_extract
        features = data_extract.extract_all_features(df, config)
        
        # Handle no valid segments
        if features is None:
            return {
                'vehicle_id': vehicle_id,
                'samples_extracted': 0,
                'status': 'no_valid_segments',
                # Add empty arrays for unpacking in main thread
                'voltagemaps': [],
                'qhisequences': [],
                'thisequences': [],
                'scalarfeatures': [],
                'soh_labels': [],
                'vehicle_ids': [],
                'timestamps_start': [],
                'timestamps_end': []
            }
        
        # Return full result with all required keys
        return {
            'vehicle_id': vehicle_id,
            'samples_extracted': len(features['soh_labels']),
            'status': 'success',
            'voltagemaps': features['voltagemaps'],
            'qhisequences': features['qhisequences'],
            'thisequences': features['thisequences'],
            'scalarfeatures': features['scalarfeatures'],
            'soh_labels': features['soh_labels'],
            'vehicle_ids': features['vehicle_ids'],
            'timestamps_start': features['timestamps_start'],
            'timestamps_end': features['timestamps_end']
        }
        
    except Exception as e:
        print(f"❌ Error in {vehicle_id}: {str(e)}")
        return {
            'vehicle_id': vehicle_id,
            'samples_extracted': 0,
            'status': f'error: {str(e)}',
            'voltagemaps': [],
            'qhisequences': [],
            'thisequences': [],
            'scalarfeatures': [],
            'soh_labels': [],
            'vehicle_ids': [],
            'timestamps_start': [],
            'timestamps_end': []
        }

def extraction_pipeline(config_path: str = "config.yaml"):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from .utils import load_config
    
    config = load_config(config_path)
    print(f"Data profile: {config['orchestration']['data_profile']}\n")
    csv_paths = get_vehicle_paths(config['paths']['csv_dir'])
    print(f"Found {len(csv_paths)} vehicle files")
    
    # Setup checkpoint (main thread)
    checkpoint = load_checkpoint(config['paths']['checkpoint_path'])
    
    # Setup HDF5 (main thread only)
    hdf5_file = setup_hdf5(config['paths']['hdf5path'], config)
    
    batch_size = config['processing'].get('vehicle_batch_size', 4)
    print(f"Batch Size: {batch_size}")

    # Initialize stats
    stats = {
        'processed': 0, 'success': 0, 'no_valid_segments': 0,
        'errors': 0, 'failed_vehicles': []
    }
    
    try:
        # Resume from checkpoint
        if checkpoint['last_vehicle']:
            start_idx = next((i for i, p in enumerate(csv_paths) 
                            if p.name == checkpoint['last_vehicle']), -1) + 1
        else:
            start_idx = 0
        
        # Process in batches
        for i in tqdm(range(start_idx, len(csv_paths), batch_size), 
                     desc="Processing batches"):
            batch_paths = csv_paths[i:i + batch_size]
            
            # Process batch in parallel (workers don't touch HDF5)
            with ProcessPoolExecutor(max_workers=batch_size) as executor:
                futures = {
                    executor.submit(process_vehicle, path, config, None): path 
                    for path in batch_paths
                }
                
                for future in as_completed(futures):
                    result = future.result()
                    stats['processed'] += 1
                    
                    # Write to HDF5 (main thread only)
                    if result['samples_extracted'] > 0:
                        # Write to HDF5 (main thread only)
                        n_new = result['samples_extracted']
                        # Append all modalities
                        for key in ['voltagemaps', 'qhisequences', 'thisequences', 
                                'scalarfeatures', 'soh_labels']:
                            hdf5_file[key].resize(hdf5_file[key].shape[0] + n_new, axis=0)
                            hdf5_file[key][-n_new:] = result[key]
                        
                        # Append metadata
                        for key in ['vehicle_ids', 'timestamps_start', 'timestamps_end']:
                            ds = hdf5_file[key]
                            ds.resize(ds.shape[0] + n_new, axis=0)
                            ds[-n_new:] = result[key]
                        
                        hdf5_file.flush()
                        stats['success'] += 1
                        
                    elif result['status'] == 'no_valid_segments':
                        stats['no_valid_segments'] += 1
                    else:
                        stats['errors'] += 1
                        stats['failed_vehicles'].append(result['vehicle_id'])
                    
                    # Update checkpoint
                    checkpoint['last_vehicle'] = result['vehicle_id']
                    checkpoint['total_samples'] += result['samples_extracted']
                    checkpoint['processed_vehicles'] += 1
            
            # Save checkpoint after each batch
            save_checkpoint(config['paths']['checkpoint_path'], checkpoint)
        
        # Final summary
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
