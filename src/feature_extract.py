# src/feature_extract.py

import numpy as np
import polars as pl
from typing import Optional

def extract_voltage_map(segment_data: pl.DataFrame, config: dict) -> Optional[np.ndarray]:
    """
    Minimal voltage-map extractor using voltage-based interpolation.
    Everything in VOLTS (consistent with logged data).
    """
    min_voltage = config['extraction']['min_voltage_threshold']  # 3.90 V
    max_voltage = config['extraction']['feature_extraction_max_voltage']  # 4.05 V
    NUM_CELLS = 96
    MAP_SIZE = 96

    # 1. Filter rows by max cell voltage in the fixed window (in volts)
    voltage_window = segment_data.filter(
        (pl.col("maxvoltagebattery") >= min_voltage)
        & (pl.col("maxvoltagebattery") <= max_voltage)
    )

    if voltage_window.height < 10:
        return None

    # 2. Parse cell voltages
    voltage_strings = voltage_window.select("batteryvoltage").to_series().to_list()
    max_voltages = voltage_window.select("maxvoltagebattery").to_series().to_list()

    cell_rows = []
    valid_max_v = []

    for volt_str, max_v in zip(voltage_strings, max_voltages):
        if not volt_str or not isinstance(volt_str, str):
            continue
        try:
            cells = [float(v) for v in volt_str.split("~") if v]
            if len(cells) != NUM_CELLS:
                continue
            cell_rows.append(cells)
            valid_max_v.append(float(max_v))  # already in volts
        except ValueError:
            continue

    if len(cell_rows) < 5:
        return None

    voltage_matrix = np.vstack(cell_rows)          # shape (T, 96)
    max_voltage_array = np.asarray(valid_max_v)    # shape (T,)

    # 3. Interpolate each cell against max-voltage sequence onto fixed voltage grid
    voltage_grid = np.linspace(min_voltage, max_voltage, MAP_SIZE)
    interpolated = np.zeros((MAP_SIZE, NUM_CELLS))

    for cell_idx in range(NUM_CELLS):
        interpolated[:, cell_idx] = np.interp(
            voltage_grid,
            max_voltage_array,
            voltage_matrix[:, cell_idx],
        )

    # 4. Normalize using the fixed [3.90, 4.05] V window, then clip
    v_range = max_voltage - min_voltage
    normalized = (interpolated - min_voltage) / v_range
    normalized = np.clip(normalized, 0.0, 1.0)

    # Quality check
    if normalized.std() < 0.001:
        return None

    return normalized

def extract_qhi_sequence(segment_data: pl.DataFrame, config: dict) -> Optional[np.ndarray]:
    """
    Extract charge capacity sequence QHI (151 points) via voltage-based interpolation.
    QHI represents cumulative charge capacity integrated over time and mapped to voltage.
    """
    min_voltage = config['extraction']['min_voltage_threshold']  # 3.90 V
    max_voltage = config['extraction']['feature_extraction_max_voltage']  # 4.05 V
    NUM_POINTS = 151
    MIN_CHARGING_CURRENT = 0.1  # Minimum current to consider as actual charging (A)
    
    # Filter to voltage window (same as voltage map)
    voltage_window = segment_data.filter(
        (pl.col("maxvoltagebattery") >= min_voltage) &
        (pl.col("maxvoltagebattery") <= max_voltage)
    )
    
    if voltage_window.height < 10:
        return None
    
    # Extract time, current, and max voltage
    time_series = voltage_window.select("terminaltime").to_series().to_numpy()
    current_series = voltage_window.select("totalcurrent").to_series().to_numpy()
    max_voltage_series = voltage_window.select("maxvoltagebattery").to_series().to_numpy()
    
    # Remove leading zeros/no-current periods
    valid_charge_mask = current_series > MIN_CHARGING_CURRENT
    if not np.any(valid_charge_mask):
        return None
    
    first_valid_idx = np.where(valid_charge_mask)[0][0]
    time_series = time_series[first_valid_idx:]
    current_series = current_series[first_valid_idx:]
    max_voltage_series = max_voltage_series[first_valid_idx:]
    
    # Calculate cumulative charge capacity: Q(t) = ∫ I dt
    # Convert time difference to hours for Ah calculation
    time_diff = np.diff(time_series, prepend=time_series[0]) / 3600.0  # seconds to hours
    capacity_increment = current_series * time_diff  # Ah
    cumulative_capacity = np.cumsum(capacity_increment)
    
    # Ensure strict monotonic voltage for interpolation
    voltage_diff = np.diff(max_voltage_series)
    if np.any(voltage_diff <= 0):
        # Remove duplicate/backward voltage points to maintain monotonicity
        valid_voltage_mask = np.concatenate([[True], voltage_diff > 0])
        max_voltage_series = max_voltage_series[valid_voltage_mask]
        cumulative_capacity = cumulative_capacity[valid_voltage_mask]
    
    if len(max_voltage_series) < 5:  # Ensure sufficient points after cleaning
        return None
    
    # Interpolate to fixed voltage grid (voltage-based, NOT time-based)
    voltage_grid = np.linspace(min_voltage, max_voltage, NUM_POINTS)
    qhi_sequence = np.interp(voltage_grid, max_voltage_series, cumulative_capacity)
    
    # Normalize by rated capacity for consistent scaling
    rated_capacity = config['dataset']['rated_capacity_ah']
    qhi_sequence = qhi_sequence / rated_capacity
    
    return qhi_sequence

def extract_thi_sequence(segment_data: pl.DataFrame, config: dict) -> Optional[np.ndarray]:
    """
    Extract temperature sequence THI (151 points) via uniform sampling across voltage window.
    Handles 32 temperature probes stored as string column with ~ delimiters.
    """
    min_voltage = config['extraction']['min_voltage_threshold']  # 3.90 V
    max_voltage = config['extraction']['feature_extraction_max_voltage']  # 4.05 V
    NUM_POINTS = 151
    
    # Filter to voltage window
    voltage_window = segment_data.filter(
        (pl.col("maxvoltagebattery") >= min_voltage) &
        (pl.col("maxvoltagebattery") <= max_voltage)
    )
    
    if voltage_window.height < 10:
        return None
    
    # Extract max voltage series for sampling alignment
    max_voltage_series = voltage_window.select("maxvoltagebattery").to_series().to_numpy()
    
    # Find the temperature column (single column with string data)
    temp_col = [col for col in voltage_window.columns if 'probetemperature' in col.lower()]
    
    if not temp_col:
        # No temperature data available
        return None
    
    # Extract and parse temperature strings
    temp_strings = voltage_window.select(temp_col[0]).to_series().to_list()
    
    probe_temps = []
    for temp_str in temp_strings:
        if not temp_str or not isinstance(temp_str, str):
            continue
        try:
            temps = [float(t) for t in temp_str.split("~") if t]
            probe_temps.append(temps)
        except ValueError:
            continue
    
    if not probe_temps:
        return None
    
    # Convert to numpy array and average across probes for each time step
    temp_matrix = np.vstack(probe_temps)  # shape (T, num_probes)
    avg_temperature = np.mean(temp_matrix, axis=1)  # average across probes
    
    if len(avg_temperature) != len(max_voltage_series):
        return None
    
    # Interpolate to same voltage grid as QHI for alignment
    voltage_grid = np.linspace(min_voltage, max_voltage, NUM_POINTS)
    thi_sequence = np.interp(voltage_grid, max_voltage_series, avg_temperature)
    
    # Normalize temperature (optional, helps with model convergence)
    temp_range = config['extraction'].get('temperature_normalization_range', 50.0)  # typical operating range
    thi_sequence = thi_sequence / temp_range
    
    return thi_sequence

def extract_scalar_features(segment_data: pl.DataFrame, qhi_sequence: np.ndarray, config: dict) -> Optional[np.ndarray]:
    """
    Extract 15 scalar point features for SOH estimation.
    Features combine vehicle operation data, QHI statistics, cell inconsistency metrics,
    and charging condition indicators.
    """
    if len(segment_data) < 10 or qhi_sequence is None or len(qhi_sequence) == 0:
        return None
    
    features = []
    
    # 1. Total odometer (accumulated mileage in km)
    # Use tail value as representative of segment
    odometer_vals = segment_data.select("totalodometer").to_series().to_numpy()
    features.append(odometer_vals[-1] if len(odometer_vals) > 0 else 0.0)
    
    # 2. T_ave - Average temperature during charging
    # Use mintemperaturevalue and maxtemperaturevalue if available
    if "mintemperaturevalue" in segment_data.columns and "maxtemperaturevalue" in segment_data.columns:
        min_temp = segment_data.select("mintemperaturevalue").to_series().to_numpy()
        max_temp = segment_data.select("maxtemperaturevalue").to_series().to_numpy()
        avg_temp = (np.mean(min_temp) + np.mean(max_temp)) / 2.0
        features.append(avg_temp)
    else:
        # Fallback to parsing probetemperatures string column
        temp_col = [col for col in segment_data.columns if 'probetemperature' in col.lower()]
        if temp_col:
            temp_strings = segment_data.select(temp_col[0]).to_series().to_list()
            all_temps = []
            for temp_str in temp_strings:
                if isinstance(temp_str, str):
                    try:
                        temps = [float(t) for t in temp_str.split("~") if t]
                        all_temps.extend(temps)
                    except:
                        continue
            features.append(np.mean(all_temps) if all_temps else 0.0)
        else:
            features.append(0.0)
    
    # 3-6. QHI statistics (using the extracted QHI sequence)
    features.append(np.mean(qhi_sequence))      # Q_ave
    features.append(np.median(qhi_sequence))    # Q_med
    features.append(np.std(qhi_sequence))       # Q_std
    features.append(np.max(qhi_sequence) - np.min(qhi_sequence))  # Q_ran
    
    # 7-10. Cell voltage range statistics (dV = max_cell_voltage - min_cell_voltage)
    voltage_strings = segment_data.select("batteryvoltage").to_series().to_list()
    voltage_ranges = []
    for volt_str in voltage_strings:
        if isinstance(volt_str, str):
            try:
                cells = [float(v) for v in volt_str.split("~") if v]
                voltage_ranges.append(np.max(cells) - np.min(cells))
            except:
                continue
    
    if voltage_ranges:
        features.extend([
            np.mean(voltage_ranges),    # dV_ave
            np.median(voltage_ranges),  # dV_med
            np.std(voltage_ranges),     # dV_std
            np.max(voltage_ranges) - np.min(voltage_ranges)  # dV_ran
        ])
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])
    
    # 11-12. Charging current statistics
    current_series = segment_data.select("totalcurrent").to_series().to_numpy()
    features.append(np.max(current_series))  # C_max
    features.append(np.mean(current_series)) # C_ave
    
    # 13. V_end - Maximum charge-ending voltage (max voltage in segment)
    max_voltage_series = segment_data.select("maxvoltagebattery").to_series().to_numpy()
    features.append(np.max(max_voltage_series))
    
    # 14-15. Minimum temperature statistics
    if "mintemperaturevalue" in segment_data.columns:
        min_temps = segment_data.select("mintemperaturevalue").to_series().to_numpy()
        features.append(np.min(min_temps))      # T_min
        features.append(np.mean(min_temps))     # T_min_ave
    elif temp_col and all_temps:
        # Fallback if only string column available
        min_temps = [np.min([float(t) for t in temp_str.split("~") if t]) 
                    for temp_str in temp_strings if isinstance(temp_str, str)]
        features.append(np.min(min_temps))      # T_min
        features.append(np.mean(min_temps))     # T_min_ave
    else:
        features.extend([0.0, 0.0])
    
    # Validate we have exactly 15 features
    if len(features) != 15:
        return None
    
    return np.array(features, dtype=np.float32)