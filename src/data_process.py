# src/data_process.py
import os
import h5py
import numpy as np
from pathlib import Path
import json
import pandas as pd
from tqdm.auto import tqdm
from typing import Dict, Any, List
from concurrent.futures import ProcessPoolExecutor, as_completed
from . import data_extract

def get_vehicle_paths(csv_dir: Path, pattern: str) -> List[Path]:
    """Enumerate and sort all vehicle CSV files."""
    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")
    
    return sorted(csv_dir.glob(pattern))

def setup_hdf5(hdf5_path: Path, config: dict) -> h5py.File:
    """Create HDF5 file with resizable datasets."""
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    
    # If file exists, try to open in append mode
    if hdf5_path.exists():
        try:
            f = h5py.File(hdf5_path, 'r+')
            # Verify required datasets exist
            required = ['voltage_maps', 'qhi_sequences', 'thi_sequences', 
                       'scalar_features', 'soh_labels', 'vehicle_ids', 
                       'timestamps_start', 'timestamps_end']
            missing = [ds for ds in required if ds not in f]
            if missing:
                print(f"Missing datasets {missing}, recreating file...")
                f.close()
                hdf5_path.unlink()
            else:
                return f
        except OSError:
            print(f"Corrupted HDF5 detected, recreating...")
            hdf5_path.unlink()
    
    # Create new file
    f = h5py.File(hdf5_path, 'w')
    
    # Get dimensions from config
    seq_len = 151
    num_cells = config['dataset']['num_cells']
    num_scalars = 15
    
    # Create datasets with initial size 0, expandable along axis 0
    f.create_dataset('voltage_maps', 
                     shape=(0, num_cells, num_cells),
                     maxshape=(None, num_cells, num_cells),
                     dtype=np.float32, compression='gzip', chunks=(32, num_cells, num_cells))
    
    f.create_dataset('qhi_sequences',
                     shape=(0, seq_len),
                     maxshape=(None, seq_len),
                     dtype=np.float32, compression='gzip', chunks=(32, seq_len))
    
    f.create_dataset('thi_sequences',
                     shape=(0, seq_len),
                     maxshape=(None, seq_len),
                     dtype=np.float32, compression='gzip', chunks=(32, seq_len))
    
    f.create_dataset('scalar_features',
                     shape=(0, num_scalars),
                     maxshape=(None, num_scalars),
                     dtype=np.float32, compression='gzip', chunks=(32, num_scalars))
    
    f.create_dataset('soh_labels',
                     shape=(0,),
                     maxshape=(None,),
                     dtype=np.float32, compression='gzip', chunks=(32,))
    
    f.create_dataset('vehicle_ids',
                     shape=(0,),
                     maxshape=(None,),
                     dtype=h5py.string_dtype(),
                     compression='gzip', chunks=(32,))
    
    f.create_dataset('timestamps_start',
                     shape=(0,),
                     maxshape=(None,),
                     dtype=h5py.string_dtype(),
                     compression='gzip', chunks=(32,))
    
    f.create_dataset('timestamps_end',
                     shape=(0,),
                     maxshape=(None,),
                     dtype=h5py.string_dtype(),
                     compression='gzip', chunks=(32,))
    
    return f

def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Load or initialize checkpoint."""
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Corrupted checkpoint, starting fresh...")
            checkpoint_path.unlink()
    
    return {
        'last_vehicle': None,
        'total_samples': 0,
        'processed_vehicles': 0,
        'failed_vehicles': []
    }

def save_checkpoint(checkpoint_path: Path, checkpoint: Dict[str, Any]) -> None:
    """Atomically save checkpoint."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = checkpoint_path.with_suffix('.tmp')
    
    with open(temp_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    
    temp_path.replace(checkpoint_path)

def append_to_hdf5(hdf5_file: h5py.File, features: Dict[str, Any]) -> None:
    """Append features to HDF5 datasets."""
    n_new = len(features['soh_labels'])
    if n_new == 0:
        return
    
    # Resize datasets
    for key in ['voltage_maps', 'qhi_sequences', 'thi_sequences', 
                'scalar_features', 'soh_labels']:
        ds = hdf5_file[key]
        old_size = ds.shape[0]
        ds.resize(old_size + n_new, axis=0)
        ds[old_size:] = features[key]
    
    # Handle string datasets
    for key in ['vehicle_ids', 'timestamps_start', 'timestamps_end']:
        ds = hdf5_file[key]
        old_size = ds.shape[0]
        ds.resize(old_size + n_new, axis=0)
        ds[old_size:] = features[key]

def extraction_pipeline(config_path: str) -> None:
    """Main extraction pipeline with parallel processing."""
    from .utils import load_config
    
    # Load and resolve config
    config = load_config(config_path)
    
    # Setup paths
    csv_dir = Path(config['paths']['datasetdir']) / config['dataset_name']
    hdf5_path = Path(config['paths']['extracted_datadir']) / config['paths']['hdf5_filename']
    checkpoint_path = Path(config['paths']['checkpoint_dir']) / f"{config['data_profile']}_{config['dataset_name']}_checkpoint.json"
    print(checkpoint_path)
    print(f"Data Profile: {config['data_profile']}")
    print(f"Dataset: {config['dataset_name']}")

    # Create directories
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get vehicle files
    csv_pattern = config['paths']['csv_pattern']
    vehicle_paths = get_vehicle_paths(csv_dir, csv_pattern)
    print(f"Found {len(vehicle_paths)} vehicle files")
    
    if not vehicle_paths:
        print(f"No CSV files found in {csv_dir}")
        return
    
    # Setup HDF5 and checkpoint
    hdf5_file = setup_hdf5(hdf5_path, config)
    checkpoint = load_checkpoint(checkpoint_path)
    
    # Resume logic
    start_idx = 0
    if checkpoint['last_vehicle']:
        try:
            start_idx = [p.stem for p in vehicle_paths].index(checkpoint['last_vehicle']) + 1
        except ValueError:
            pass
    
    # Processing stats
    stats = {
        'processed': 0,
        'success': 0,
        'no_valid_segments': 0,
        'errors': 0,
        'failed_vehicles': checkpoint['failed_vehicles'].copy()
    }
    
    batch_size = config['processing']['parallel_batch_size']
    
    try:
        # Process in batches
        for i in tqdm(range(start_idx, len(vehicle_paths), batch_size), 
                     desc="Processing batches"):
            batch_paths = vehicle_paths[i:i + batch_size]
            
            with ProcessPoolExecutor(max_workers=batch_size) as executor:
                futures = {
                    executor.submit(data_extract.process_vehicle, path, config): path
                    for path in batch_paths
                }
                
                for future in as_completed(futures):
                    result = future.result()
                    stats['processed'] += 1
                    
                    if result['status'] == 'success' and result['samples_extracted'] > 0:
                        append_to_hdf5(hdf5_file, result['features'])
                        hdf5_file.flush()
                        stats['success'] += 1
                        
                        # Update checkpoint
                        checkpoint['last_vehicle'] = result['vehicle_id']
                        checkpoint['total_samples'] += result['samples_extracted']
                        checkpoint['processed_vehicles'] += 1
                        save_checkpoint(checkpoint_path, checkpoint)
                        
                    elif result['status'] == 'no_valid_segments':
                        stats['no_valid_segments'] += 1
                    else:
                        stats['errors'] += 1
                        stats['failed_vehicles'].append(result['vehicle_id'])
        
        # Final summary
        print("\n" + "="*50)
        print("EXTRACTION COMPLETE")
        print("="*50)
        print(f"Total vehicles processed: {stats['processed']}")
        print(f"Successful extractions: {stats['success']}")
        print(f"No valid segments: {stats['no_valid_segments']}")
        print(f"Errors: {stats['errors']}")
        print(f"Total samples extracted: {checkpoint['total_samples']}")
        print(f"SOH range: {hdf5_file['soh_labels'][:].min():.3f} to {hdf5_file['soh_labels'][:].max():.3f}")
        print("="*50)
        
        if stats['failed_vehicles']:
            print(f"Failed vehicles: {stats['failed_vehicles']}")
        
    finally:
        hdf5_file.close()
        print("HDF5 file closed safely")

# Backward compatibility
process_vehicle = data_extract.process_vehicle
