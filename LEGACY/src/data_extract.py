# src/data_extraction.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import warnings

# Paper-specified constants (hardcoded as discussed)
SEQUENCE_LENGTH = 151
NUM_CELLS = 96
NUM_SCALAR_FEATURES = 15

def parse_battery_voltage(cell_string: Any) -> np.ndarray:
    """Robustly parse battery voltage string into 96 cell voltages in mV."""
    try:
        if isinstance(cell_string, str):
            voltages = np.fromstring(cell_string, sep='~', dtype=float)
        elif isinstance(cell_string, (int, float)) and not pd.isna(cell_string):
            voltages = np.full(96, float(cell_string))
        else:
            voltages = np.zeros(96)
        
        # Ensure exactly 96 values
        if len(voltages) != 96:
            if len(voltages) < 96:
                voltages = np.pad(voltages, (0, 96 - len(voltages)), 'constant')
            else:
                voltages = voltages[:96]
        
        return voltages * 1000  # Convert V to mV
    except Exception as e:
        warnings.warn(f"Failed to parse battery voltage: {e}")
        return np.zeros(96)

def voltage_to_soc(voltage_mv: float, ocv_table_path: Optional[Path] = None) -> float:
    """Convert OCV (mV) to SOC using lookup table or NCM typical curve."""
    try:
        if ocv_table_path and ocv_table_path.exists():
            table = np.loadtxt(ocv_table_path)
            return np.interp(voltage_mv, table[:, 0], table[:, 1])
    except:
        pass
    
    # NCM typical OCV-SOC curve (hardcoded fallback)
    ncm_lut = np.array([
        [2800, 0.00], [3300, 0.05], [3500, 0.10], [3600, 0.20],
        [3650, 0.30], [3700, 0.40], [3750, 0.50], [3800, 0.60],
        [3900, 0.70], [4000, 0.80], [4100, 0.90], [4150, 0.95], [4200, 1.00]
    ])
    soc = np.interp(voltage_mv, ncm_lut[:, 0], ncm_lut[:, 1])
    return np.clip(soc, 0.0, 1.0)

def detect_rest_periods(df: pd.DataFrame, config: dict) -> List[Dict[str, Any]]:
    """Detect rest periods and calibrate SOC via OCV."""
    df = df.sort_values('terminaltime').reset_index(drop=True)
    
    min_gap = config['ocv_calibration']['min_rest_hours'] * 3600
    max_soc = config['ocv_calibration']['max_soc_start']
    ocv_path = config['ocv_calibration'].get('ocv_table_path')
    if ocv_path:
        ocv_path = Path(ocv_path)
    
    time_gaps = df['terminaltime'].diff()
    rest_mask = time_gaps > min_gap
    
    rest_events = []
    for idx in df[rest_mask].index:
        voltage_v = df.loc[idx, 'maxvoltagebattery']
        voltage_mv = voltage_v * 1000
        
        soc_ocv = voltage_to_soc(voltage_mv, ocv_path)
        
        if soc_ocv < max_soc:
            rest_events.append({
                'ts': df.loc[idx, 'terminaltime'],
                'soc_ocv': soc_ocv,
                'maxvoltage_mv': voltage_mv,
                'startidx': idx
            })
    
    return rest_events

'''
    PARKING PERIOD FIX
'''
def identify_charging_segments(df: pd.DataFrame, rest_events: List[dict], config: dict) -> List[dict]:
    """Identify charging segments that contain the voltage window."""
    df = df.sort_values('terminaltime').reset_index(drop=True)
    
    status_value = config['charging']['status_value']
    voltage_window = config['dataset']['voltage_window_mv']
    min_samples = config['quality_checks']['min_samples_in_window']
    
    segments = []
    
    for rest in rest_events:
        start_idx = rest['startidx']
        post_rest = df.iloc[start_idx:].reset_index(drop=True)
        
        # Find charging start
        charging_mask = post_rest['chargestatus'] == status_value
        if not charging_mask.any():
            continue
        
        # Get the full charging session
        charging_start_idx = post_rest[charging_mask].index[0]
        segment_df = post_rest.iloc[charging_start_idx:]
        
        # Find where voltage window occurs within this charging session
        voltages = segment_df['maxvoltagebattery'].values * 1000
        
        # Find indices where voltage is within the window
        in_window = (voltages >= voltage_window[0]) & (voltages <= voltage_window[1])
        
        if not in_window.any():
            continue
        
        # Get contiguous block within window
        window_indices = np.where(in_window)[0]
        if len(window_indices) < min_samples:
            continue
        
        # Extract only the window portion
        window_start = window_indices[0]
        window_end = window_indices[-1]
        window_df = segment_df.iloc[window_start:window_end + 1]
        
        # Check for gaps WITHIN the window (not the parking period before)
        time_gaps = window_df['terminaltime'].diff().dropna()
        if (time_gaps > config['quality_checks']['max_gap_seconds']).any():
            continue
        
        segments.append({
            'segment_df': window_df,
            'ts': window_df['terminaltime'].iloc[0],
            'te': window_df['terminaltime'].iloc[-1],
            'soc_ocv': rest['soc_ocv']
        })
    
    return segments

def validate_and_clean_segment(segment_df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, dict]:
    """Apply data quality corrections and return cleaned segment."""
    flags = {}

    '''
        CURRENT UNIT DETECTION FIX
    '''
    current_raw = segment_df['totalcurrent'].values
    if np.abs(current_raw).max() > 500:  # If max current > 500, it's likely mA
        # warnings.warn(f"Current appears to be in mA (max={current_raw.max():.1f}). Converting to A.")
        segment_df['totalcurrent'] = current_raw / 1000.0

    # Check minimum length
    min_samples = config['quality_checks']['min_samples_in_window']
    flags['has_min_records'] = len(segment_df) >= min_samples
    
    # Check temporal gaps
    max_gap = config['quality_checks']['max_gap_seconds']
    time_gaps = segment_df['terminaltime'].diff().dropna()
    flags['no_large_gaps'] = (time_gaps <= max_gap).all()
    
    # Voltage smoothing (default: interpolation)
    smoothing_method = config['quality_checks'].get('voltage_smoothing', 'interpolation')
    if smoothing_method == 'interpolation':
        voltage = segment_df['maxvoltagebattery'].values
        for i in range(1, len(voltage)):
            if voltage[i] < voltage[i-1]:
                voltage[i] = voltage[i-1]
        segment_df['maxvoltagebattery'] = voltage
    
    # Current sign correction
    sign_convention = config['quality_checks']['current_sign_convention']
    if sign_convention == 'forcepositive':
        segment_df['totalcurrent'] = segment_df['totalcurrent'].abs()
    elif sign_convention == 'inverted':
        segment_df['totalcurrent'] = -segment_df['totalcurrent']
    
    # Check current bounds
    current_bounds = config['quality_checks']['current_bounds_a']
    current_abs = segment_df['totalcurrent'].abs()
    flags['current_in_bounds'] = (
        (current_abs >= current_bounds[0]) & 
        (current_abs <= current_bounds[1])
    ).all()
    
    # Check voltage window coverage
    voltage_window = config['dataset']['voltage_window_mv']
    voltages = segment_df['maxvoltagebattery'].values * 1000
    flags['covers_voltage_window'] = (
        (voltages >= voltage_window[0]).any() and 
        (voltages <= voltage_window[1]).any()
    )
    
    '''
        DEBUG LOGGING FIX
    '''
    flags['is_valid'] = True
    failure_reasons = []
    
    if not flags['has_min_records']:
        flags['is_valid'] = False
        failure_reasons.append(f"too_few_samples_{len(segment_df)}")
    
    if not flags['no_large_gaps']:
        flags['is_valid'] = False
        max_gap_found = time_gaps.max()
        failure_reasons.append(f"large_gap_{max_gap_found:.1f}s")
    
    if not flags['current_in_bounds']:
        flags['is_valid'] = False
        current_min = current_abs.min()
        current_max = current_abs.max()
        failure_reasons.append(f"current_out_of_bounds_{current_min:.1f}-{current_max:.1f}A")
    
    if not flags['covers_voltage_window']:
        flags['is_valid'] = False
        v_min = voltages.min()
        v_max = voltages.max()
        failure_reasons.append(f"voltage_window_{v_min:.0f}-{v_max:.0f}mV")
    
    if failure_reasons:
        vehicle_id = segment_df.get('vehicleid', ['unknown'])[0]
        # print(f"DEBUG: Rejected segment from {vehicle_id}: {'; '.join(failure_reasons)}")
    
    return segment_df, flags

def compute_soh(segment_df: pd.DataFrame, soc_ocv: float, config: dict) -> float:
    """Compute SOH using paper-correct formula with OCV-SOC correction."""
    rated_capacity = config['dataset']['rated_capacity_ah']
    soh_bounds = config['soh_calculation']['soh_bounds']
    min_delta_soc = config['soh_calculation']['min_abs_delta_soc']
    
    # Ampere-hour integration
    dt = segment_df['terminaltime'].diff().fillna(0).values
    current = segment_df['totalcurrent'].abs().values
    charged_capacity = np.sum(current * dt) / 3600.0
    
    # SOC at end (prefer voltage-based if full charge reached)
    end_voltage_mv = segment_df['maxvoltagebattery'].iloc[-1] * 1000
    ocv_path = config['ocv_calibration'].get('ocv_table_path')
    if ocv_path:
        ocv_path = Path(ocv_path)
    soc_end = voltage_to_soc(end_voltage_mv, ocv_path)
    
    # Fallback to BMS SOC if voltage method fails
    if soc_end < 0 or soc_end > 1:
        soc_end = segment_df['soc'].iloc[-1] / 100.0
    
    delta_soc = max(soc_end - soc_ocv, min_delta_soc)

    '''
        DEBUG CHECK
    '''
    # print(f"DEBUG: Current stats - min={current.min():.2f}, max={current.max():.2f}, mean={current.mean():.2f}")
    # print(f"DEBUG: dt stats - min={dt.min():.2f}, max={dt.max():.2f}, mean={dt.mean():.2f}")
    # print(f"DEBUG: Charged capacity = {charged_capacity:.6f} Ah")
    # print(f"DEBUG: SOC start={soc_ocv:.3f}, end={soc_end:.3f}, delta={delta_soc:.3f}")
    
    # Paper-correct formula: SOH = (charged_capacity / ΔSOC) / rated_capacity
    soh = (charged_capacity / delta_soc) / rated_capacity
    
    '''
        VALIDATION BOUNDS FIX
    '''
    if not (soh_bounds[0] <= soh <= soh_bounds[1]):
        return np.nan
    
    return soh

def extract_qhi_thi(segment_df: pd.DataFrame, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Extract QHI and THI sequences using voltage-based interpolation."""
    voltage_window = config['dataset']['voltage_window_mv']
    
    # Filter to voltage window
    voltages = segment_df['maxvoltagebattery'].values * 1000
    in_window = (voltages >= voltage_window[0]) & (voltages <= voltage_window[1])
    
    if not in_window.any():
        raise ValueError("No data in voltage window")
    
    window_df = segment_df[in_window]
    
    # Extract values
    voltage_points = window_df['maxvoltagebattery'].values * 1000
    capacity = window_df['totalcurrent'].abs().cumsum() * (window_df['terminaltime'].diff().fillna(0) / 3600.0)
    capacity = capacity.values
    
    # Temperature (use maxtemp if available, else estimate)
    if 'maxtemperaturevalue' in window_df.columns:
        temperature = window_df['maxtemperaturevalue'].values
    else:
        temperature = np.full(len(window_df), 25.0)  # Default room temp
    
    # Interpolate to fixed voltage points
    v_target = np.linspace(voltage_window[0], voltage_window[1], SEQUENCE_LENGTH)
    
    qhi = np.interp(v_target, voltage_points, capacity)
    thi = np.interp(v_target, voltage_points, temperature)
    
    # Normalize to [0,1] for model input
    qhi_norm = (qhi - qhi.min()) / (qhi.max() - qhi.min() + 1e-8)
    thi_norm = (thi - thi.min()) / (thi.max() - thi.min() + 1e-8)
    
    return qhi_norm, thi_norm

def extract_voltage_map(segment_df: pd.DataFrame, config: dict) -> np.ndarray:
    """Extract 96x96 voltage map (time steps × cells)."""
    voltage_window = config['dataset']['voltage_window_mv']
    
    # Parse cell voltages for each timestep
    cell_voltages = []
    for voltage_str in segment_df['batteryvoltage'].values:
        cells = parse_battery_voltage(voltage_str)
        cell_voltages.append(cells)
    cell_voltages = np.array(cell_voltages)  # Shape: (timesteps, 96)
    
    # Filter to voltage window
    max_voltages = cell_voltages.max(axis=1)
    in_window = (max_voltages >= voltage_window[0]) & (max_voltages <= voltage_window[1])
    
    if not in_window.any():
        raise ValueError("No data in voltage window")
    
    window_voltages = cell_voltages[in_window]
    
    # Interpolate to exactly 96 timesteps
    n_timesteps = len(window_voltages)
    if n_timesteps < 2:
        raise ValueError("Insufficient timesteps")
    
    x_original = np.linspace(0, 1, n_timesteps)
    x_target = np.linspace(0, 1, 96)
    
    voltage_map = np.zeros((96, 96))
    for cell_idx in range(96):
        voltage_map[:, cell_idx] = np.interp(x_target, x_original, window_voltages[:, cell_idx])
    
    # Normalize to [0,1]
    voltage_map = (voltage_map - voltage_map.min()) / (voltage_map.max() - voltage_map.min() + 1e-8)
    
    return voltage_map

def extract_scalar_features(segment_df: pd.DataFrame, qhi_raw: np.ndarray, thi_raw: np.ndarray, soc_ocv: float, config: dict) -> np.ndarray:
    """Extract 15-point scalar feature vector matching Paper Table 1."""
    features = []
    
    # 1. Mileage (km)
    if 'totalodometer' in segment_df.columns:
        mileage = segment_df['totalodometer'].iloc[-1]
    else:
        mileage = 0.0
    features.append(mileage)
    
    # 2. Average temperature
    if 'maxtemperaturevalue' in segment_df.columns:
        t_avg = segment_df['maxtemperaturevalue'].mean()
    else:
        t_avg = 25.0
    features.append(t_avg)
    
    # 3-6. QHI statistics (from raw values)
    features.extend([
        float(np.mean(qhi_raw)),
        float(np.median(qhi_raw)),
        float(np.std(qhi_raw)),
        float(np.ptp(qhi_raw))
    ])
    
    # 7-10. Cell voltage range statistics
    try:
        cell_voltages = [parse_battery_voltage(v) for v in segment_df['batteryvoltage'].values]
        cell_voltages = np.array(cell_voltages)
        dV = np.max(cell_voltages, axis=1) - np.min(cell_voltages, axis=1)
        features.extend([
            float(np.mean(dV)),
            float(np.median(dV)),
            float(np.std(dV)),
            float(np.ptp(dV))
        ])
    except:
        features.extend([0.0, 0.0, 0.0, 0.0])
    
    # 11-12. Current statistics
    features.extend([
        float(segment_df['totalcurrent'].abs().max()),
        float(segment_df['totalcurrent'].abs().mean())
    ])
    
    # 13. End voltage
    features.append(float(segment_df['maxvoltagebattery'].iloc[-1]))
    
    # 14-15. Temperature features
    if 'mintemperaturevalue' in segment_df.columns:
        t_min = segment_df['mintemperaturevalue'].min()
        t_min_avg = segment_df['mintemperaturevalue'].mean()
    else:
        t_min = 25.0
        t_min_avg = 25.0
    features.extend([t_min, t_min_avg])
    
    # Ensure exactly 15 features
    if len(features) != NUM_SCALAR_FEATURES:
        warnings.warn(f"Expected 15 features, got {len(features)}. Padding...")
        features = (features + [0.0] * NUM_SCALAR_FEATURES)[:NUM_SCALAR_FEATURES]
    
    return np.array(features, dtype=np.float32)

def extract_all_features(df: pd.DataFrame, config: dict) -> Optional[Dict[str, Any]]:
    """Full feature extraction pipeline for one vehicle."""
    ''' VEHICLE ID LOGGING '''
    vehicle_id = df.get('vehicleid', ['unknown'])[0]
    
    # Ensure required columns exist
    required_cols = ['terminaltime', 'totalcurrent', 'maxvoltagebattery', 
                     'batteryvoltage', 'chargestatus', 'soc']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        ''' VEHICLE ID LOGGING '''
        warnings.warn(f"Vehicle {vehicle_id}: Missing columns {missing_cols}")
        return None
    
    # Stage 1: Detect rest periods
    rest_events = detect_rest_periods(df, config)
    if not rest_events:
        ''' VEHICLE ID LOGGING '''
        warnings.warn(f"Vehicle {vehicle_id}: No rest events")
        return None
    
    # Stage 2: Identify charging segments
    segment_candidates = identify_charging_segments(df, rest_events, config)
    if not segment_candidates:
        ''' VEHICLE ID LOGGING '''
        # warnings.warn(f"Vehicle {vehicle_id}: No charging segments identified")
        return None
    
    # Collections for all valid segments
    all_features = {
        'voltage_maps': [],
        'qhi_sequences': [],
        'thi_sequences': [],
        'scalar_features': [],
        'soh_labels': [],
        'vehicle_ids': [],
        'timestamps_start': [],
        'timestamps_end': []
    }
    
    # Stage 3: Process each segment
    for seg in segment_candidates:
        try:
            segment_df, quality_flags = validate_and_clean_segment(seg['segment_df'], config)
            
            '''
                FAILURE LOGGING
            '''
            if not quality_flags['is_valid']:
                reasons = []
                if not quality_flags['has_min_records']:
                    reasons.append(f"too_few_samples_{len(segment_df)}")
                if not quality_flags['no_large_gaps']:
                    reasons.append("large_gaps")
                if not quality_flags['current_in_bounds']:
                    reasons.append("current_out_of_bounds")
                if not quality_flags['covers_voltage_window']:
                    reasons.append("voltage_window_not_covered")
                
                warnings.warn(f"Vehicle {vehicle_id}, segment {i}: {'; '.join(reasons)}")
                continue
            
            '''
                COMPUTE SOH FIX
            '''
            soh = compute_soh(segment_df, seg['soc_ocv'], config)
            if np.isnan(soh):
                continue
            
            # Extract features
            qhi, thi = extract_qhi_thi(segment_df, config)
            voltage_map = extract_voltage_map(segment_df, config)
            # Get raw QHI/THI for scalar features (before normalization)
            qhi_raw, thi_raw = qhi, thi  # These are already extracted
            
            scalars = extract_scalar_features(segment_df, qhi_raw, thi_raw, seg['soc_ocv'], config)
            
            # Append to collections
            all_features['voltage_maps'].append(voltage_map)
            all_features['qhi_sequences'].append(qhi)
            all_features['thi_sequences'].append(thi)
            all_features['scalar_features'].append(scalars)
            all_features['soh_labels'].append(soh)
            all_features['vehicle_ids'].append(seg['segment_df']['vehicleid'].iloc[0])
            all_features['timestamps_start'].append(str(int(seg['ts'])))
            all_features['timestamps_end'].append(str(int(seg['te'])))
            
        except Exception as e:
            warnings.warn(f"Failed to process segment: {e}")
            continue
    
    # Check if any valid segments
    if not all_features['soh_labels']:
        return None
    
    # Convert to numpy arrays
    return {
        'voltage_maps': np.array(all_features['voltage_maps'], dtype=np.float32),
        'qhi_sequences': np.array(all_features['qhi_sequences'], dtype=np.float32),
        'thi_sequences': np.array(all_features['thi_sequences'], dtype=np.float32),
        'scalar_features': np.array(all_features['scalar_features'], dtype=np.float32),
        'soh_labels': np.array(all_features['soh_labels'], dtype=np.float32),
        'vehicle_ids': all_features['vehicle_ids'],
        'timestamps_start': all_features['timestamps_start'],
        'timestamps_end': all_features['timestamps_end']
    }

def process_vehicle(csv_path: Path, config: dict) -> Dict[str, Any]:
    """Process one vehicle CSV and return features dict."""
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        df['vehicleid'] = csv_path.stem  # Add vehicle ID column
        
        features = extract_all_features(df, config)
        
        if features is None:
            return {
                'vehicle_id': csv_path.stem,
                'samples_extracted': 0,
                'status': 'no_valid_segments'
            }
        
        return {
            'vehicle_id': csv_path.stem,
            'samples_extracted': len(features['soh_labels']),
            'status': 'success',
            'features': features
        }
    except Exception as e:
        warnings.warn(f"Error processing {csv_path.stem}: {e}")
        return {
            'vehicle_id': csv_path.stem,
            'samples_extracted': 0,
            'status': f'error: {str(e)}'
        }
