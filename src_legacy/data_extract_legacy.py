
# src/data_extract.py

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import sys
from pathlib import Path

def parse_battery_voltage(cell_string: Any) -> np.ndarray:
    """
    Robustly parse batteryvoltage column which can be:
    - String: "4.101~4.125~4.119~..."
    - Float/NaN: corrupted data
    - Other types: fallback to zeros
    
    Returns array of 96 cell voltages in mV
    """
    try:
        if isinstance(cell_string, str):
            # Normal case: tilde-separated string
            voltages = np.fromstring(cell_string, sep='~', dtype=float)
        elif isinstance(cell_string, (int, float)) and not pd.isna(cell_string):
            # Single value: replicate across 96 cells
            voltages = np.full(96, float(cell_string))
        else:
            # NaN or other corrupted data
            voltages = np.zeros(96)
        
        # Ensure we have exactly 96 values
        if len(voltages) != 96:
            if len(voltages) > 96:
                voltages = voltages[:96]
            else:
                # Pad with zeros
                voltages = np.pad(voltages, (0, 96 - len(voltages)), 'constant')
        
        return voltages * 1000  # Convert V → mV
    
    except Exception as e:
        print(f"Warning: Failed to parse batteryvoltage '{cell_string}': {e}")
        return np.zeros(96)  # Return zeros as fallback

def voltage_to_soc(voltage_mv: float, ocv_soc_table: Optional[np.ndarray] = None) -> float:
    """Convert OCV (mV) to SOC (0-1) using lookup table or linear approximation."""
    if ocv_soc_table is not None:
        voltages = ocv_soc_table[:, 0]
        socs = ocv_soc_table[:, 1]
        return np.interp(voltage_mv, voltages, socs)
    else:
        # Linear approximation (mV range)
        soc = (voltage_mv - 3000) / (4200 - 3000)
        return np.clip(soc, 0.0, 1.0)

def detect_rest_periods(
    df: pd.DataFrame,
    min_rest_hours: float = 1.0,
    ocv_soc_table: Optional[np.ndarray] = None
) -> List[Dict[str, Any]]:
    """
    Detect rest periods > min_rest_hours and calibrate SOC via OCV.
    
    Parameters
    ----------
    df : pd.DataFrame
        Vehicle DataFrame with columns:
        - terminaltime (cumulative seconds)
        - maxvoltagebattery (max cell voltage in V)
        - soc (BMS SOC %, 0-100)
    min_rest_hours : float
        Minimum rest duration in hours (default: 1.0)
    ocv_soc_table : np.ndarray, optional
        2D array mapping OCV (mV) → SOC (0-1). If None, uses linear approximation.
    """
    # Copy to avoid modifying original
    df = df.copy()
    
    # Ensure terminaltime is numeric
    df['terminaltime'] = pd.to_numeric(df['terminaltime'], errors='coerce')
    
    # Compute time gaps in seconds
    df = df.sort_values('terminaltime').reset_index(drop=True)
    time_gaps = df['terminaltime'].diff()
    
    # Find rest periods
    min_gap_seconds = min_rest_hours * 3600
    rest_mask = time_gaps > min_gap_seconds
    
    rest_events = []
    for idx in df[rest_mask].index:
        t_s = df.loc[idx, 'terminaltime']
        max_voltage_v = df.loc[idx, 'maxvoltagebattery']
        max_voltage_mv = max_voltage_v * 1000  # Convert V → mV
        
        # OCV → SOC calibration
        soc_ocv = voltage_to_soc(max_voltage_mv, ocv_soc_table)
        
        # Filter: keep only if SOC < 0.6
        if soc_ocv < 0.6:
            rest_events.append({
                't_s': t_s,
                'soc_ocv': soc_ocv,
                'max_voltage': max_voltage_mv,
                'start_idx': idx
            })
    
    return rest_events

def identify_charging_segments(
    df: pd.DataFrame,
    rest_events: List[Dict[str, Any]],
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Identify full-charge segments following each rest event.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full vehicle DataFrame
    rest_events : list
        Output from detect_rest_periods()
    config : dict
        Must contain processing parameters: min_charging_records, voltage_window_mv
        
    Returns
    -------
    list of dicts
        Each dict contains segment data and quality flags
    """
    segments = []
    df = df.sort_values('terminaltime').reset_index(drop=True)
    
    for rest_event in rest_events:
        t_s = rest_event['t_s']
        start_idx = rest_event['start_idx']
        
        # Scan forward from rest point
        post_rest = df.iloc[start_idx:]
        
        # Find charging start: chargestatus == 3
        charging_mask = post_rest['chargestatus'] == 3
        if not charging_mask.any():
            continue  # No charging after this rest
        
        charging_start_idx = post_rest[charging_mask].index[0]
        
        # Find full charge: soc == 100 AND maxvoltagebattery >= 4.25
        full_charge_mask = (
            (post_rest['soc'] >= 99.5) &  # Allow 99.5+ for float precision
            (post_rest['maxvoltagebattery'] >= 4.24)  # 4.25V threshold
        )
        
        if not full_charge_mask.any():
            continue  # Never reached full charge
        
        full_charge_idx = post_rest[full_charge_mask].index[0]
        
        # Extract segment
        segment_df = df.loc[charging_start_idx:full_charge_idx].copy()
        segment_df = segment_df.sort_values('terminaltime').reset_index(drop=True)
        
        # Compute time gaps for validation
        time_gaps = segment_df['terminaltime'].diff().dropna()
        
        # Validate segment quality
        quality_flags = validate_charging_segment(segment_df, time_gaps, config)
        
        if quality_flags['is_valid']:
            segments.append({
                'segment_df': segment_df,
                't_s': t_s,
                't_e': segment_df['terminaltime'].iloc[-1],
                'soc_ocv': rest_event['soc_ocv'],
                'quality_flags': quality_flags
            })
    
    return segments

def validate_charging_segment(
    segment_df: pd.DataFrame,
    time_gaps: pd.Series,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate charging segment quality."""
    flags = {}
    
    # Check 1: Minimum length
    min_records = config['processing']['min_charging_records']
    flags['has_min_records'] = len(segment_df) >= min_records
    
    # Check 2: No large gaps
    max_gap_seconds = config['processing']['max_gap_seconds']
    flags['no_large_gaps'] = (time_gaps <= max_gap_seconds).all()
    
    # Check 3: Voltage monotonic increase (allow small drops)
    voltage_diff = segment_df['maxvoltagebattery'].diff().dropna()
    threshold = config['processing']['voltage_increasing_threshold']
    flags['voltage_increasing'] = (voltage_diff > -0.001).mean() >= threshold
    
    # Check 4: Current within realistic bounds
    current_bounds = config['processing']['current_bounds']
    current_abs = segment_df['totalcurrent'].abs()
    flags['current_in_bounds'] = (
        (current_abs >= current_bounds[0]) &
        (current_abs <= current_bounds[1])
    ).all()
    
    # Check 5: Voltage window coverage (3900-4050 mV)
    voltage_window = config['dataset']['voltage_window_mv']
    voltage_mv = segment_df['maxvoltagebattery'] * 1000
    flags['covers_voltage_window'] = (
        (voltage_mv >= voltage_window[0]).any() and
        (voltage_mv <= voltage_window[1]).any()
    )
    
    # Check 6: SOH plausibility
    try:
        dt = segment_df['terminaltime'].diff().fillna(0).values
        current_a = segment_df['totalcurrent'].abs().values
        capacity_ah = (current_a * dt).sum() / 3600.0
        rated_capacity = config['dataset']['rated_capacity_ah']
        soh_estimate = capacity_ah / rated_capacity
        flags['soh_plausible'] = 0.5 <= soh_estimate <= 1.2
    except:
        flags['soh_plausible'] = False
    
    # Overall validity
    flags['is_valid'] = all([
        flags['has_min_records'],
        flags['no_large_gaps'],
        flags['voltage_increasing'],
        flags['current_in_bounds'],
        flags['covers_voltage_window'],
        flags['soh_plausible']
    ])
    
    return flags

def compute_soh(
    segment_df: pd.DataFrame,
    soc_ocv: float,
    rated_capacity_ah: float = 155.0
) -> float:
    """
    Compute SOH via ampere-hour integration.
    
    Parameters
    ----------
    segment_df : pd.DataFrame
        Charging segment with columns: terminaltime, totalcurrent
    soc_ocv : float
        Calibrated SOC at segment start (0-1)
    rated_capacity_ah : float
        Nominal pack capacity (default: 155.0 Ah)
    """
    # Time step in seconds
    dt = segment_df['terminaltime'].diff().fillna(0)
    
    # Use charging current magnitude
    if 'totalcurrent' not in segment_df.columns:
        raise KeyError("totalcurrent column missing in segment_df")
    
    current_a = segment_df['totalcurrent'].abs().values
    
    # Ampere-hour integration
    capacity_ah = (current_a * dt.values).sum() / 3600.0
    soh = capacity_ah / rated_capacity_ah
    
    if not (0.5 <= soh <= 1.2):
        raise ValueError(f"Unrealistic SOH: {soh:.3f} (capacity={capacity_ah:.2f}Ah)")
    
    return soh

def extract_qhi_thi(segment_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract QHI (charge capacity sequence) and THI (temperature sequence).
    
    QHI is the cumulative capacity delivered during charging, normalized to [0, 1].
    THI is the temperature profile, normalized to [0, 1].
    
    Returns
    -------
    tuple (qhi, thi)
        Both are np.ndarray of shape (151,) normalized to [0, 1]
    """
    # Ensure required columns exist
    required_cols = ['terminaltime', 'totalcurrent', 'maxtemperaturevalue']
    for col in required_cols:
        if col not in segment_df.columns:
            raise KeyError(f"Missing required column: {col}")
    
    # Extract raw values
    terminaltime = segment_df['terminaltime'].values
    current_a = segment_df['totalcurrent'].abs().values
    temperature_c = segment_df['maxtemperaturevalue'].values
    
    # Normalize time to [0, 1] for interpolation
    t_norm = (terminaltime - terminaltime[0]) / (terminaltime[-1] - terminaltime[0] + 1e-8)
    
    # Target interpolation points (151 points as per paper)
    t_target = np.linspace(0, 1, 151)
    
    # === CORRECT QHI: Cumulative capacity integration ===
    # Compute time differences in hours
    dt_hours = np.diff(terminaltime, prepend=terminaltime[0]) / 3600.0
    
    # Compute cumulative capacity (Ah)
    capacity_ah = np.cumsum(current_a * dt_hours)
    
    # Interpolate QHI (capacity sequence)
    qhi_raw = np.interp(t_target, t_norm, capacity_ah)
    
    # === THI extraction (unchanged) ===
    thi_raw = np.interp(t_target, t_norm, temperature_c)
    
    # Normalize QHI to [0, 1] (capacity range normalization)
    qhi_min, qhi_max = qhi_raw.min(), qhi_raw.max()
    if qhi_max > qhi_min:
        qhi = (qhi_raw - qhi_min) / (qhi_max - qhi_min)
    else:
        qhi = np.zeros_like(qhi_raw)
    
    # Normalize THI to [0, 1]
    thi_min, thi_max = thi_raw.min(), thi_raw.max()
    if thi_max > thi_min:
        thi = (thi_raw - thi_min) / (thi_max - thi_min)
    else:
        thi = np.zeros_like(thi_raw)
    
    return qhi, thi


def extract_voltage_map(segment_df: pd.DataFrame, config: Dict[str, Any]) -> np.ndarray:
    """
    Extract 96×96 voltage difference matrix from segment.
    
    Parameters
    ----------
    segment_df : pd.DataFrame
        Charging segment with batteryvoltage column
    config : dict
        Must contain dataset.voltage_window_mv
        
    Returns
    -------
    np.ndarray
        96×96 voltage difference map normalized to [0, 1]
    """
    # Step 1: Parse batteryvoltage strings
    if 'batteryvoltage' not in segment_df.columns:
        raise KeyError("batteryvoltage column missing")
    
    cell_voltages = np.array([
        parse_battery_voltage(s) for s in segment_df['batteryvoltage'].values
    ])
    
    # Validate shape
    if cell_voltages.shape[1] != 96:
        raise ValueError(f"Expected 96 cells, got {cell_voltages.shape[1]}")
    
    # Step 2: Filter to voltage window
    max_voltages = cell_voltages.max(axis=1)
    voltage_window = config['dataset']['voltage_window_mv']
    window_mask = (max_voltages >= voltage_window[0]) & (max_voltages <= voltage_window[1])
    
    if not window_mask.any():
        return np.zeros((96, 96))
    
    window_voltages = cell_voltages[window_mask]
    
    # Step 3: Interpolate to exactly 96 timepoints
    original_len = len(window_voltages)
    if original_len < 2:
        return np.zeros((96, 96))
    
    x = np.linspace(0, 1, original_len)
    x_target = np.linspace(0, 1, 96)
    
    interpolated = np.array([
        np.interp(x_target, x, window_voltages[:, i])
        for i in range(96)
    ]).T  # Shape: (96, 96)
    
    # Step 4: Create 96×96 difference matrices for each timepoint
    maps_3d = np.zeros((96, 96, 96))
    for t in range(96):
        for i in range(96):
            for j in range(96):
                maps_3d[i, j, t] = interpolated[t, i] - interpolated[t, j]
    
    # Step 5: Aggregate across time (mean)
    final_map = maps_3d.mean(axis=2)
    
    # Step 6: Normalize to [0, 1]
    map_min, map_max = final_map.min(), final_map.max()
    if map_max > map_min:
        final_map = (final_map - map_min) / (map_max - map_min)
    else:
        final_map = np.zeros_like(final_map)
    
    return final_map

def extract_scalar_features(
    segment_df: pd.DataFrame,
    qhi: np.ndarray,
    thi: np.ndarray,
    soc_ocv: float,
    config: Dict[str, Any]
) -> np.ndarray:
    """
    Extract exactly 15-point degradation indicator vector matching paper's Table 1.
    
    Paper specifications:
    1.  Mileage (km)
    2.  Tave: Average battery temperature during charging (°C)
    3.  Qave: Average QHI value (Ah)
    4.  Qmed: Median QHI value (Ah)
    5.  Qstd: Standard deviation of QHI (Ah)
    6.  Qran: Range of QHI (Ah)
    7.  dVave: Average cell voltage range (mV)
    8.  dVmed: Median cell voltage range (mV)
    9.  dVstd: Standard deviation of cell voltage range (mV)
    10. dVran: Range of cell voltage range (mV)
    11. Cmax: Maximum charging current (A)
    12. Cave: Average charging current (A)
    13. Vend: Maximum charge-ending voltage (V)
    14. Tmin: Minimum temperature during charging (°C)
    15. Tmin_ave: Average of minimum temperature sequence (°C)
    """
    features = []
    
    # 1. Mileage (km)
    try:
        mileage = segment_df['totalodometer'].iloc[-1] if 'totalodometer' in segment_df.columns else 0.0
    except:
        mileage = 0.0
    features.append(mileage)
    
    # 2. Tave: Average temperature during charging
    try:
        tave = segment_df['maxtemperaturevalue'].mean()
    except:
        tave = 25.0
    features.append(tave)
    
    # 3-6: QHI statistics
    # QHI is already normalized to [0,1], convert back to Ah using SOH
    try:
        rated_capacity = config['dataset']['rated_capacity_ah']
        # Denormalize QHI to approximate Ah values
        qhi_ah = qhi * (soc_ocv * rated_capacity)  # Approximate capacity range
        qave = float(np.mean(qhi_ah))
        qmed = float(np.median(qhi_ah))
        qstd = float(np.std(qhi_ah))
        qran = float(np.max(qhi_ah) - np.min(qhi_ah))
    except:
        qave = qmed = qstd = qran = 0.0
    features.extend([qave, qmed, qstd, qran])
    
    # 7-10: Cell voltage range (dV) statistics
    # dV sequence: max(cell_v) - min(cell_v) at each timestep
    try:
        cell_voltages = np.array([
            parse_battery_voltage(s) for s in segment_df['batteryvoltage'].values
        ])  # Shape: (timesteps, 96)
        dV_sequence = np.max(cell_voltages, axis=1) - np.min(cell_voltages, axis=1)  # mV
        dVave = float(np.mean(dV_sequence))
        dVmed = float(np.median(dV_sequence))
        dVstd = float(np.std(dV_sequence))
        dVran = float(np.max(dV_sequence) - np.min(dV_sequence))
    except Exception as e:
        print(f"Warning: dV extraction failed: {e}")
        dVave = dVmed = dVstd = dVran = 0.0
    features.extend([dVave, dVmed, dVstd, dVran])
    
    # 11. Cmax: Maximum charging current
    try:
        cmax = segment_df['totalcurrent'].abs().max()
    except:
        cmax = 0.0
    features.append(cmax)
    
    # 12. Cave: Average charging current
    try:
        cave = segment_df['totalcurrent'].abs().mean()
    except:
        cave = 0.0
    features.append(cave)
    
    # 13. Vend: Maximum charge-ending voltage
    try:
        vend = segment_df['maxvoltagebattery'].max()
    except:
        vend = 4.2
    features.append(vend)
    
    # 14. Tmin: Minimum temperature during charging
    try:
        tmin = segment_df['mintemperaturevalue'].min() if 'mintemperaturevalue' in segment_df.columns else thi.min()
    except:
        tmin = 25.0
    features.append(tmin)
    
    # 15. Tmin_ave: Average of minimum temperature sequence
    # This is the average of THI (temperature sequence)
    try:
        tmin_ave = float(np.mean(thi))
    except:
        tmin_ave = 25.0
    features.append(tmin_ave)
    
    # Validate exactly 15 features
    if len(features) != 15:
        print(f"Warning: Expected 15 features, got {len(features)}. Padding to 15...")
        padded = np.zeros(15, dtype=np.float32)
        padded[:len(features)] = features[:15]
        features = padded.tolist()
    
    return np.array(features, dtype=np.float32)


def extract_all_features(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Full feature extraction pipeline for one vehicle.
    
    Returns
    -------
    dict or None
        If valid segments found, returns dict with keys:
        - voltagemaps: np.array (N, 96, 96)
        - qhisequences: np.array (N, 151)
        - thisequences: np.array (N, 151)
        - scalarfeatures: np.array (N, 15)
        - soh_labels: np.array (N,)
        - vehicle_ids: list[str]
        - timestamps_start: list[str]
        - timestamps_end: list[str]
    """
    # Step 1: Detect rest periods
    rest_events = detect_rest_periods(
        df,
        min_rest_hours=config['processing']['min_rest_hours']
    )
    
    if not rest_events:
        return None
    
    # Step 2: Identify charging segments
    segments = identify_charging_segments(df, rest_events, config)
    
    if not segments:
        return None
    
    # Step 3: Extract features from each segment
    all_features = {
        'voltagemaps': [],
        'qhisequences': [],
        'thisequences': [],
        'scalarfeatures': [],
        'soh_labels': [],
        'vehicle_ids': [],
        'timestamps_start': [],
        'timestamps_end': []
    }
    
    for seg in segments:
        try:
            # Compute SOH
            soh = compute_soh(seg['segment_df'], seg['soc_ocv'], config['dataset']['rated_capacity_ah'])
            
            # Extract QHI/THI sequences
            qhi, thi = extract_qhi_thi(seg['segment_df'])
            
            # Extract voltage map
            vmap = extract_voltage_map(seg['segment_df'], config)
            
            # Extract scalar features
            scalars = extract_scalar_features(seg['segment_df'], qhi, thi, seg['soc_ocv'], config)
            
            # Append to collection
            all_features['voltagemaps'].append(vmap)
            all_features['qhisequences'].append(qhi)
            all_features['thisequences'].append(thi)
            all_features['scalarfeatures'].append(scalars)
            all_features['soh_labels'].append(soh)
            all_features['vehicle_ids'].append(seg['segment_df'].iloc[0]['vehicle_id'])
            all_features['timestamps_start'].append(str(seg['t_s']))
            all_features['timestamps_end'].append(str(seg['t_e']))
            
        except Exception as e:
            print(f"Warning: Failed to process segment: {e}")
            continue
    
    if not all_features['soh_labels']:
        return None
    
    # Convert lists to arrays
    return {k: np.array(v) if isinstance(v[0], np.ndarray) else v
            for k, v in all_features.items()}

def run_extraction_pipeline(config_path: str = "config.yaml"):
    """Orchestrate full Stage 1-2 pipeline with new config system."""
    from .utils import load_config
    
    config = load_config(config_path)
    print(f"Running extraction pipeline for dataset: {config['paths']['dataset_name']}")
    print(f"Extracted data will be saved to: {config['paths']['hdf5path']}")
    
    # Import data_process to run the pipeline
    from .data_process import run_pipeline
    run_pipeline(config)

# ============================================================================
# Test Harness
# ============================================================================
if __name__ == "__main__":
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # Load config
    from src.utils import load_config_for_testing
    config = load_config_for_testing()
    
    csv_dir = config['paths']['csv_dir']
    
    # Find first CSV
    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {csv_dir}")
        sys.exit(1)
    
    first_csv = csv_files[0]
    print(f"Testing full pipeline on: {first_csv.name}")
    
    # Load and test
    df = pd.read_csv(first_csv)
    df['vehicle_id'] = first_csv.stem  # Add vehicle_id column
    
    features = extract_all_features(df, config)
    
    if features is None:
        print("No valid features extracted")
    else:
        print(f"Successfully extracted {len(features['soh_labels'])} samples")
        print(f"SOH range: {features['soh_labels'].min():.3f} to {features['soh_labels'].max():.3f}")
        print(f"Feature shapes:")
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: list of length {len(value)}")
