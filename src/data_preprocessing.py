# src/data_preprocessing.py

import polars as pl
from typing import Optional


def data_clean(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """
    Basic EV battery data cleaning with essential filters only:
    - Enforce current sign convention
    - Optional simple voltage smoothing
    - Remove clearly impossible voltage/current values
    """
    # Extract thresholds from config
    MAX_VOLTAGE = config["dataset"].get("max_voltage_threshold", 5.0)
    MAX_CURRENT = config["preprocessing"].get("max_current_threshold", 500.0)

    # 1) Remove rows with missing critical data
    df = df.drop_nulls(["maxvoltagebattery", "totalcurrent"])

    # 2) Current sign convention
    if config["preprocessing"]["current_sign_convention"] == "positive":
        df = df.with_columns(pl.col("totalcurrent").abs())
    else:
        df = df.with_columns(-pl.col("totalcurrent").abs())

    # 3) Simple voltage smoothing with fixed window (optional)
    if config["preprocessing"].get("enable_voltage_smoothing", False):
        window = config["preprocessing"].get("smoothing_window", 5)
        df = df.with_columns(
            pl.col("maxvoltagebattery")
            .rolling_mean(window, min_samples=1)
            .alias("maxvoltagebattery")
        )

    # 4) Basic physical bounds filtering
    df = df.filter(
        (pl.col("maxvoltagebattery") > 0)
        & (pl.col("maxvoltagebattery") < MAX_VOLTAGE)
        & (pl.col("totalcurrent").abs() < MAX_CURRENT)
    )

    return df


def extract_rest_segments(df: pl.DataFrame, gap_period: int) -> pl.Series:
    """
    Extract rest start times based on gaps in terminaltime.
    """
    time_diff = df["terminaltime"].diff()
    rest_segments = df.filter(
        (time_diff > gap_period) & (pl.col("terminaltime") > 0)
    )["terminaltime"]
    return rest_segments



def default_ocv_lookup():
    """Default approximated OCV-SOC lookup table for NCM batteries at rest"""
    return {
        4.200: 1.000, 4.180: 0.970, 4.160: 0.940, 4.140: 0.915,
        4.120: 0.890, 4.100: 0.860, 4.080: 0.825, 4.060: 0.785,
        4.040: 0.740, 4.020: 0.695, 4.000: 0.650, 3.980: 0.605,
        3.960: 0.560, 3.940: 0.515, 3.920: 0.470, 3.900: 0.425,
        3.880: 0.380, 3.860: 0.340, 3.840: 0.300, 3.820: 0.265,
        3.800: 0.230, 3.780: 0.200, 3.760: 0.175, 3.740: 0.150,
        3.720: 0.130, 3.700: 0.115, 3.680: 0.100, 3.660: 0.085,
        3.640: 0.070, 3.620: 0.055, 3.600: 0.045, 3.580: 0.035,
        3.560: 0.025, 3.540: 0.015, 3.520: 0.008, 3.500: 0.000
    }

import polars as pl
import numpy as np

def calibrate_soc(df: pl.DataFrame, rest_start: float, ocv_lookup: dict, 
                          min_sample: int = 100, current_at_rest: float = 1.0,
                          depolarization_period: int = 3600, max_soc_threshold: float = 0.6) -> Optional[float]:
    # Vectorized filtering
    restdata = df.filter(
        (pl.col("terminaltime") >= rest_start) &
        (pl.col("terminaltime") <= rest_start + depolarization_period)
    )
    
    if restdata.height < min_sample:
        return None
        
    if "totalcurrent" in restdata.columns:
        if restdata.select(pl.col("totalcurrent").abs().max()).item() > current_at_rest:
            return None
    
    # Vectorized OCV lookup using numpy interpolation
    tail_voltage = restdata.select(pl.col("maxvoltagebattery").tail(5).mean()).item()
    
    ocv_items = sorted(ocv_lookup.items())  # Keep as Python floats
    ocv_voltages = np.array([v for v, _ in ocv_items], dtype=np.float32)
    ocv_socs = np.array([soc for _, soc in ocv_items], dtype=np.float32)
    
    # Single interpolation call
    calibrated_soc = float(np.interp(tail_voltage, ocv_voltages, ocv_socs))
    
    if calibrated_soc is None or calibrated_soc > max_soc_threshold:
        return None
        
    return calibrated_soc


'''def calibrate_soc(
    df: pl.DataFrame,
    rest_start: float,
    ocv_lookup: dict,
    min_sample: int = 100,
    current_at_rest: float = 1.0,
    depolarization_period: int = 3600,
    max_soc_threshold: float = 0.6,
) -> Optional[float]:
    """
    Calibrate SOC using OCV after a depolarization period, with minimal but
    important guards on sample count and rest current.
    """
    # Extract rest window data
    rest_data = df.filter(
        (pl.col("terminaltime") >= rest_start)
        & (pl.col("terminaltime") <= rest_start + depolarization_period)
    )

    # Minimum sample count
    if rest_data.height < min_sample:
        return None

    # Enforce near-zero current during rest if current is available
    if "totalcurrent" in rest_data.columns:
        if rest_data["totalcurrent"].abs().max() > current_at_rest:
            return None

    # Use last few samples for voltage (post-depolarization OCV)
    tail_voltage = float(rest_data["maxvoltagebattery"].tail(5).mean())

    # Linear interpolation between nearest OCV points
    voltages = sorted(ocv_lookup.keys())
    lower_v = max([v for v in voltages if v <= tail_voltage], default=None)
    upper_v = min([v for v in voltages if v >= tail_voltage], default=None)
    # print(lower_v)

    # DEBUG: Print what we found
    # print(f"[DEBUG] tail_voltage: {tail_voltage}, lower_v: {lower_v}, upper_v: {upper_v}")

    # Fix: Ensure we ALWAYS get a valid SOC, never a voltage
    if lower_v is None or upper_v is None:
        # Voltage out of range - this is a data quality issue
        # print(f"[DEBUG] Voltage {tail_voltage} out of OCV range {min(voltages)}-{max(voltages)}")
        calibrated_soc = None
    elif lower_v == upper_v:
        # Exact match in lookup table
        calibrated_soc = ocv_lookup[lower_v]
        # print(f"[DEBUG] Exact OCV match: {tail_voltage} -> SOC {calibrated_soc}")
    else:
        # Interpolation
        lower_soc = ocv_lookup[lower_v]
        upper_soc = ocv_lookup[upper_v]
        weight = (tail_voltage - lower_v) / (upper_v - lower_v)
        calibrated_soc = lower_soc + weight * (upper_soc - lower_soc)
        # print(f"[DEBUG] Interpolated: {tail_voltage}V -> SOC {calibrated_soc}")

    # Print the final result
    # print(f"[DEBUG] Final calibrated_soc: {calibrated_soc}")

    # Apply SOC threshold
    if calibrated_soc is None or calibrated_soc > max_soc_threshold:
        return None

    return calibrated_soc'''

def identify_capacity_segment(
    df: pl.DataFrame,
    rest_start: float,
    calibrated_soc: float,
    full_charge_threshold: float,
    min_soc_interval: float = 0.4,
) -> Optional[pl.DataFrame]:
    """
    Identify capacity calculation segment from rest moment until full charge.
    Requires that the segment spans a sufficient SOC range.
    """
    # Get all data after rest start
    segment_data = df.filter(pl.col("terminaltime") >= rest_start)

    # Find when pack/cell voltage hits full charge threshold
    full_charge_points = segment_data.filter(
        pl.col("maxvoltagebattery") >= full_charge_threshold
    )

    if full_charge_points.height == 0:
        return None

    # Get the first time when full charge condition is met
    full_charge_time = full_charge_points.select("terminaltime").row(0)[0]

    # Extract the complete segment
    capacity_segment = segment_data.filter(pl.col("terminaltime") <= full_charge_time)

    # Validate sufficient data points
    if capacity_segment.height < 10:
        return None

    # Validate SOC span based on calibrated SOC
    soc_span = 1.0 - calibrated_soc
    if soc_span < min_soc_interval:
        return None

    # Ensure segment is actually charging (positive current in chosen convention)
    if "totalcurrent" in capacity_segment.columns:
        if capacity_segment["totalcurrent"].mean() <= 0:
            return None

    return capacity_segment


def calculate_soh_label(
    capacity_segment: pl.DataFrame,
    calibrated_soc: float,
    rated_capacity_ah: float = 155.0,
) -> Optional[float]:
    """
    Calculate SOH using ampere-hour integration:
    SOH = (Q_charged / ΔSOC) / Q_rated
    """
    # Filter out rows with nulls in required columns
    valid_segment = capacity_segment.filter(
        pl.col("totalcurrent").is_not_null()
        & pl.col("terminaltime").is_not_null()
        & pl.col("soc").is_not_null()
    )
    if valid_segment.height < 2:
        return None

    # Time intervals in hours (terminaltime in seconds)
    time_diff = (valid_segment["terminaltime"].diff() / 3600.0).fill_null(0)

    # Ampere-hour integration
    q_charged = (valid_segment["totalcurrent"] * time_diff).sum()

    # SOC at end (clamped)
    soc_end = min(valid_segment.select("soc").tail(1).row(0)[0], 1.0)
    soc_start = calibrated_soc

    # Validate SOC interval
    delta_soc = soc_end - soc_start
    if delta_soc <= 0:
        return None

    # SOH calculation
    soh = (q_charged / delta_soc) / rated_capacity_ah

    # Basic sanity bounds
    if soh < 0.5 or soh > 1.2:
        return None

    return soh