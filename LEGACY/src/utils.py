import yaml
from pathlib import Path

def load_config(config_path: str) -> dict:
    """
    Load and resolve config with profile merging from config_profiles/.
    Loads data_profile and model_profile if specified.
    """
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

    # Load data profile
    data_profile_path = config_path.parent / 'config_profiles' / f"{data_profile}.yaml"
    if data_profile_path.exists():
        with open(data_profile_path) as f:
            data_profile_cfg = yaml.safe_load(f)
        config.update(data_profile_cfg)
    else:
        print(f"Warning: Data profile not found at {data_profile_path}")

    # Load model profile if specified
    model_profile = config.get('model_profile', 'model_final')
    if model_profile:
        model_profile_path = config_path.parent / 'config_profiles' / f"{model_profile}.yaml"
        if model_profile_path.exists():
            with open(model_profile_path) as f:
                model_profile_cfg = yaml.safe_load(f)
            config.update(model_profile_cfg)
        else:
            print(f"Warning: Model profile not found at {model_profile_path}")

    return config
