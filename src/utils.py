# src/utils.py

from pathlib import Path
from typing import Dict, Any
import yaml

def open_config(path_to_open: str) -> Dict[str, Any]:
    with open(path_to_open, 'r') as f:
        config_name = yaml.safe_load(f)

    return config_name

def load_config(config_path: str) -> Dict[str, Any]:
    # Opening the main config
    config_path = Path(config_path).resolve()
    config = open_config(config_path)

    # Resolving profile paths
    data_profile_path = f"{config['paths']['data_profile_dir']}/{config['orchestration']['data_profile']}.yaml"
    model_profile_path = f"{config['paths']['model_profile_dir']}/{config['orchestration']['model_profile']}.yaml"
    
    # Opening the profile configs
    data_profile = open_config(data_profile_path)
    model_profile = open_config(model_profile_path)

    # Merge profiles into config
    config['dataset'] = data_profile.get('dataset', {})
    config['preprocessing'] = data_profile.get('preprocessing', {})
    config['extraction'] = data_profile.get('extraction', {})

    return config
