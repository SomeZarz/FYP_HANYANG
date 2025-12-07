import yaml
from pathlib import Path

def load_config(config_path: str) -> dict:
    """Load and resolve config with profile merging from config_profiles/."""
    config_path = Path(config_path)
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Extract values for template substitution
    data_profile = config['data_profile']
    dataset_name = config['dataset_name']
    
    # Format HDF5 filename with actual values
    hdf5_template = config['paths']['hdf5_filename']
    config['paths']['hdf5_filename'] = hdf5_template.format(
        data_profile=data_profile, 
        dataset_name=dataset_name
    )
    
    # Construct profile file path and merge
    profile_path = config_path.parent / 'config_profiles' / f"{data_profile}.yaml"
    
    if profile_path.exists():
        with open(profile_path) as f:
            profile = yaml.safe_load(f)
        config.update(profile)
    else:
        print(f"Warning: Profile not found at {profile_path}")
    
    return config

