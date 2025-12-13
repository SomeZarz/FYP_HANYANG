# src/data_processing.py

import polars as pl
from pathlib import Path
from typing import Iterator, Tuple, List, Dict, Any

from .utils import load_config

def data_scanning(dataset_path: Path) -> Iterator[Tuple[str, pl.LazyFrame]]:
    """
    Convert CSV to Parquet if needed, then scan Parquet files.
    Keeps storage-efficient columnar format while keeping the logic simple.
    """
    # If no Parquet exists yet, create them from CSV
    if not dataset_path.exists() or not list(dataset_path.glob("*.parquet")):
        for csv_file in dataset_path.glob("*.csv"):
            df = pl.read_csv(csv_file)
            df.write_parquet(dataset_path / f"{csv_file.stem}.parquet")

    # Scan all Parquet files lazily
    for pq_file in dataset_path.glob("*.parquet"):
        lf = pl.scan_parquet(pq_file)
        yield str(pq_file), lf


def preprocess_pipeline(config_path: str = "config.yaml"):
    """
    Main pipeline for SOH label extraction.
    """
    from .data_preprocessing import (
    data_clean, extract_rest_segments,
    default_ocv_lookup, calibrate_soc, 
    identify_capacity_segment,
    calculate_soh_label,
    )

    config = load_config(config_path)

    dataset_path = Path(
        f"{config['paths']['dataset_dir']}/{config['orchestration']['dataset_name']}"
    )

    ocv_lookup = default_ocv_lookup()
    soh_labels = []

    for file_path, lf in data_scanning(dataset_path):
        # Materialize the lazy frame
        df = lf.collect(engine='streaming')

        # Basic cleaning (sign convention, hard bounds, optional smoothing)
        df = data_clean(df, config)

        # Find rest segments based on time gaps
        rest_segments = extract_rest_segments(
            df,
            gap_period=config["preprocessing"]["gap_period"],
        )
        # For each rest segment, try to calibrate SOC and compute SOH
        for rest_start in rest_segments:
            calibrated_soc = calibrate_soc(
                df,
                rest_start=rest_start,
                ocv_lookup=ocv_lookup,
                min_sample=config["preprocessing"]["min_sample"],
                current_at_rest=config["preprocessing"]["current_at_rest"],
                depolarization_period=config["preprocessing"]["depolarization_period"],
                max_soc_threshold=config["preprocessing"]["max_soc_threshold"],
            )
            if calibrated_soc is not None:
                segment = identify_capacity_segment(
                    df,
                    rest_start=rest_start,
                    calibrated_soc=calibrated_soc,
                    full_charge_threshold=config["preprocessing"]["full_charge_threshold"],
                )

                if segment is not None:
                    soh = calculate_soh_label(
                        segment,
                        calibrated_soc=calibrated_soc,
                        rated_capacity_ah=config["dataset"]["rated_capacity_ah"],
                    )

                    if soh is not None:
                        soh_labels.append(
                            {
                                "file": file_path,
                                "rest_start": rest_start,
                                "calibrated_soc": calibrated_soc,
                                "segment": segment,
                                "segment_end": segment.select("terminaltime")
                                .tail(1)
                                .row(0)[0],
                                "segment_length": segment.height,
                                "soh": soh,
                            }
                        )

    print(f"Found {len(soh_labels)} usable SOH labels across all files")
    return soh_labels

def extraction_pipeline(sohlabels: List[Dict[str, Any]], configpath: str = "config.yaml") -> List[Dict[str, Any]]:
    """Batch processing version for efficiency"""
    from .feature_extract import (
    extract_voltage_map,
    extract_qhi_sequence, extract_thi_sequence,
    extract_scalar_features
    )

    config = load_config(configpath)
    extractedsamples = []
    skipped = 0
    
    # Process in batches of 10
    batch_size = config['performance']['batch_size']
    for i in range(0, len(sohlabels), batch_size):
        batch = sohlabels[i:i + batch_size]
        
        for sample in batch:
            try:
                segment = sample['segment']
                
                # Basic validation
                if not (0.5 <= sample['soh'] <= 1.2):
                    skipped += 1
                    continue
                
                # Extract all features in optimized functions
                voltagemap = extract_voltage_map(segment, config)
                qhiseq = extract_qhi_sequence(segment, config)
                thiseq = extract_thi_sequence(segment, config)
                scalarfeatures = extract_scalar_features(segment, qhiseq, config)
                
                # Check all modalities present
                if all(v is not None for v in [voltagemap, qhiseq, thiseq, scalarfeatures]):
                    extractedsamples.append({
                        'file': sample['file'],
                        'soh': sample['soh'],
                        'voltagemap': voltagemap,
                        'qhisequence': qhiseq,
                        'thisequence': thiseq,
                        'scalarfeatures': scalarfeatures
                    })
                else:
                    skipped += 1
                    
            except Exception:
                skipped += 1
                continue
    
    print(f"Extracted: {len(extractedsamples)}, Skipped: {skipped}")
    return extractedsamples

def write_features_to_parquet(extracted_samples: List[Dict[str, Any]], configpath: str = "config.yaml"):
    """Write extracted features to Parquet file using config-based path."""
    config = load_config(configpath)
    if not extracted_samples:
        print("No samples to write")
        return
    
    # Convert numpy arrays to lists for Parquet compatibility
    rows = []
    for sample in extracted_samples:
        rows.append({
            'file': sample['file'],
            'soh': sample['soh'],
            'voltagemap': sample['voltagemap'].flatten().tolist() if sample['voltagemap'] is not None else None,
            'qhisequence': sample['qhisequence'].tolist() if sample['qhisequence'] is not None else None,
            'thisequence': sample['thisequence'].tolist() if sample['thisequence'] is not None else None,
            'scalarfeatures': sample['scalarfeatures'].tolist() if sample['scalarfeatures'] is not None else None
        })
    
    # Create DataFrame
    df = pl.DataFrame(rows)
    
    # Use config-based path format
    features_path = Path(
        f"{config['paths']['extracted_data_dir']}/{config['orchestration']['data_profile']}_{config['orchestration']['dataset_name']}.parquet"
    )
    
    # Ensure directory exists
    features_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to Parquet
    df.write_parquet(features_path)
    print(f"Wrote {len(rows)} samples to {features_path}")
