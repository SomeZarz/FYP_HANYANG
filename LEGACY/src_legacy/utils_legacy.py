
# src/utils.py

import os
from pathlib import Path
from typing import Dict, Any
import yaml

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and merge main config with profile overrides.
    
    Args:
        config_path: Path to main config.yaml file
        
    Returns:
        Fully resolved configuration dictionary with merged profiles
        and resolved absolute paths.
    """
    config_path = Path(config_path).resolve()
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load main config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract profile information
    profiles_dir = Path(config['orchestration']['profiles_dir'])
    # Make profiles_dir absolute relative to config file location
    if not profiles_dir.is_absolute():
        profiles_dir = config_path.parent / profiles_dir
    
    if not profiles_dir.exists():
        raise FileNotFoundError(f"Profiles directory not found: {profiles_dir}")
    
    data_profile_name = config['orchestration']['data_profile']
    model_profile_name = config['orchestration']['model_profile']
    
    # Load data profile
    data_profile_path = profiles_dir / f"{data_profile_name}.yaml"
    if not data_profile_path.exists():
        raise FileNotFoundError(f"Data profile not found: {data_profile_path}")
    
    with open(data_profile_path, 'r') as f:
        data_profile = yaml.safe_load(f)
    
    # Load model profile
    model_profile_path = profiles_dir / f"{model_profile_name}.yaml"
    if not model_profile_path.exists():
        raise FileNotFoundError(f"Model profile not found: {model_profile_path}")
    
    with open(model_profile_path, 'r') as f:
        model_profile = yaml.safe_load(f)
    
    # Merge profiles into main config
    config['processing'] = data_profile.get('processing', {})
    config['model'] = model_profile.get('model', {})
    config['training'] = model_profile.get('training', {})
    
    # Ensure dataset metadata section exists
    if 'dataset' not in config:
        config['dataset'] = {}
    
    # Resolve paths
    paths = config['paths']
    
    # Make dataset_dir absolute
    dataset_dir = Path(paths['dataset_dir'])
    if not dataset_dir.is_absolute():
        dataset_dir = config_path.parent / dataset_dir
    paths['dataset_dir'] = dataset_dir.resolve()
    
    # Construct csv_dir
    csv_dir = dataset_dir / paths['dataset_name']
    paths['csv_dir'] = csv_dir.resolve()
    
    # Resolve extracted_data_dir
    extracted_data_dir = Path(paths['extracted_data_dir'])
    if not extracted_data_dir.is_absolute():
        extracted_data_dir = config_path.parent / extracted_data_dir
    paths['extracted_data_dir'] = extracted_data_dir.resolve()
    
    # Resolve checkpoint_dir
    checkpoint_dir = Path(paths['checkpoint_dir'])
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = config_path.parent / checkpoint_dir
    paths['checkpoint_dir'] = checkpoint_dir.resolve()
    
    # Ensure output directories exist
    paths['extracted_data_dir'].mkdir(parents=True, exist_ok=True)
    paths['checkpoint_dir'].mkdir(parents=True, exist_ok=True)
    
    # Resolve template paths
    paths['hdf5path'] = Path(paths['hdf5path'].format(
        data_profile=data_profile_name,
        dataset_name=paths['dataset_name'],
        extracted_data_dir=paths['extracted_data_dir']
    )).resolve()

    paths['checkpoint_path'] = Path(paths['data_checkpoint_filename'].format(
    data_profile=data_profile_name,
    dataset_name=paths['dataset_name'],
    extracted_data_dir=paths['extracted_data_dir']
)).resolve()
    
    return config
