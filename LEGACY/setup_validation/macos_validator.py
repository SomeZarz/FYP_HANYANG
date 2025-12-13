#!/usr/bin/env python3
"""
macos_validator.py
Stage 0 environment validator for macOS + PyTorch (MPS) pipeline.
Verifies dependencies, GPU availability, and HDF5 I/O integrity.
"""

import os
import subprocess
import warnings
import time
import tempfile
from importlib import import_module
from typing import Dict, Any, Optional, Callable

warnings.filterwarnings("ignore")

# ============================================================================
# DEPENDENCY VERIFICATION
# ============================================================================
DEPENDENCIES = {
    "torch": (True, "PyTorch deep learning framework"),
    "torchvision": (True, "PyTorch vision utilities"),
    "torchaudio": (True, "PyTorch audio utilities"),
    "pandas": (True, "Data manipulation library"),
    "yaml": (True, "Config manipulation library"),
    "numpy": (True, "Numerical computing library"),
    "scipy": (True, "Scientific computing library"),
    "sklearn": (True, "Scikit-learn ML library"),
    "h5py": (True, "HDF5 file format support"),
    "tqdm": (True, "Progress bar library"),
    "matplotlib": (False, "Plotting library"),
    "seaborn": (False, "Statistical visualization library"),
    "memory_profiler": (False, "Memory profiling tool"),
}

def dependency_check() -> Dict[str, Dict[str, Any]]:
    """Verify all dependencies and return their status."""
    results = {}
    for import_name, (required, description) in DEPENDENCIES.items():
        try:
            module = import_module(import_name)
            version = getattr(module, "__version__", "unknown")
            status = "✓"
            installed = True
        except ImportError:
            version = None
            status = "✗"
            installed = False
        
        results[import_name] = {
            "installed": installed,
            "version": version,
            "required": required,
            "status": status,
            "description": description,
        }
    return results

# ============================================================================
# GPU AVAILABILITY VERIFICATION
# ============================================================================
def check_torch_mps() -> Dict[str, Any]:
    """Check PyTorch MPS availability on Apple Silicon."""
    try:
        import torch
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        return {
            "installed": True,
            "version": torch.__version__,
            "mps_available": has_mps,
            "device": "mps" if has_mps else "cpu",
            "errors": [],
        }
    except ImportError:
        return {
            "installed": False,
            "version": None,
            "mps_available": False,
            "device": "cpu",
            "errors": ["PyTorch not installed"],
        }

def platform_info() -> Dict[str, Any]:
    """Detect platform architecture."""
    try:
        arch = subprocess.check_output(["uname", "-m"], text=True).strip()
        return {
            "platform": arch,
            "is_apple_silicon": "arm64" in arch,
            "is_unix": True,
        }
    except Exception:
        return {
            "platform": "unknown",
            "is_apple_silicon": False,
            "is_unix": False,
        }

# ============================================================================
# HDF5 FUNCTIONAL TEST
# ============================================================================
def h5py_test() -> Dict[str, Any]:
    """Test HDF5 write/read integrity with sample data structure."""
    result = {"working": False, "errors": []}
    try:
        import h5py
        import numpy as np
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            fname = f.name
        
        try:
            # Write sample mimicking Stage 2 output
            with h5py.File(fname, "w") as hf:
                # 96x96 voltage map
                hf.create_dataset("voltagemaps", data=np.random.randn(1, 96, 96).astype(np.float32), 
                                compression="gzip")
                # 2x151 sequences
                hf.create_dataset("qhisequences", data=np.random.randn(1, 151).astype(np.float32))
                hf.create_dataset("thisequences", data=np.random.randn(1, 151).astype(np.float32))
                # 15 scalars
                hf.create_dataset("scalarfeatures", data=np.random.randn(1, 15).astype(np.float32))
                # Attributes
                hf.attrs["vehicle_id"] = "test_vin_123"
                hf.attrs["timestamp"] = "2025-12-05T19:49:00"
            
            # Read back and verify
            with h5py.File(fname, "r") as hf:
                assert hf["voltagemaps"].shape == (1, 96, 96)
                assert hf["qhisequences"].shape == (1, 151)
                assert hf.attrs["vehicle_id"] == "test_vin_123"
            
            result["working"] = True
        finally:
            os.unlink(fname)
            
    except Exception as e:
        result["errors"].append(f"HDF5 test failed: {e}")
    
    return result

# ============================================================================
# GPU FUNCTIONAL TEST
# ============================================================================
def benchmark_op(op: Callable, sync: Optional[Callable] = None, warmup: int = 3, iters: int = 10) -> Dict[str, float]:
    """Standardized operation benchmark with warmup and synchronization."""
    for _ in range(warmup):
        op()
        if sync:
            sync()
    
    start = time.perf_counter()
    for _ in range(iters):
        op()
        if sync:
            sync()
    elapsed = time.perf_counter() - start
    
    return {
        "iters": iters,
        "time_ms": elapsed * 1000 / iters,
        "throughput": iters / elapsed,
    }

def gpu_functional_test(run_benchmark: bool = True) -> Dict[str, Any]:
    """Functional test for MPS GPU operations with optional benchmark."""
    result = {"working": False, "tensor_ops": False, "perf": None, "errors": []}
    
    torch_info = check_torch_mps()
    if not torch_info["installed"]:
        result["errors"].extend(torch_info["errors"])
        return result
    
    device = torch_info["device"]
    try:
        import torch
        # Test tensor ops
        x = torch.randn(100, 100, device=device)
        torch.matmul(x, x)
        model = torch.nn.Linear(100, 50).to(device)
        model(torch.randn(32, 100, device=device))
        result["tensor_ops"] = True
        
        # Benchmark
        if run_benchmark and device == "mps":
            def op():
                torch.matmul(
                    torch.randn(1000, 1000, device=device),
                    torch.randn(1000, 1000, device=device)
                )
            result["perf"] = benchmark_op(op, torch.mps.synchronize)
        
        result["working"] = True
    except Exception as e:
        result["errors"].append(f"Tensor ops failed: {e}")
    
    return result

# ============================================================================
# REPORTING
# ============================================================================
def format_deps_report(deps: Dict[str, Dict[str, Any]]) -> str:
    """Format dependency check results as human-readable report."""
    lines = ["\n=== DEPENDENCY STATUS ==="]
    required_fail = [pkg for pkg, info in deps.items() if info["required"] and not info["installed"]]
    
    for pkg, info in sorted(deps.items()):
        status = "✓" if info["installed"] else "✗"
        ver = f"v{info['version']}" if info["installed"] else "missing"
        lines.append(f" {status} {pkg:20} {ver:15} {info['description']}")
    
    if required_fail:
        lines.append(f"\n⚠ CRITICAL: Missing required packages: {', '.join(required_fail)}")
    
    return "\n".join(lines)

def format_gpu_report(gpu_info: Dict[str, Any]) -> str:
    """Format GPU status check results."""
    lines = ["\n=== GPU/ACCELERATOR STATUS ==="]
    lines.append(f"Platform: {gpu_info['platform']}")
    lines.append(f"Apple Silicon: {'Yes' if gpu_info['is_apple_silicon'] else 'No'}")
    
    torch_info = gpu_info["torch"]
    lines.append(f"\nPyTorch MPS:")
    lines.append(f" Installed: {'Yes' if torch_info['installed'] else 'No'}")
    if torch_info["installed"]:
        lines.append(f" Version: {torch_info['version']}")
        lines.append(f" MPS Available: {'Yes' if torch_info['mps_available'] else 'No'}")
        lines.append(f" Device: {torch_info['device']}")
        if torch_info["errors"]:
            lines.append(f" Errors: {'; '.join(torch_info['errors'])}")
    
    lines.append(f"\nRecommended Device: {gpu_info['recommended_device'].upper()}")
    return "\n".join(lines)

def format_h5py_report(h5_result: Dict[str, Any]) -> str:
    """Format HDF5 test results."""
    lines = ["\n=== HDF5 I/O TEST ==="]
    lines.append(f"Working: {'✓ Yes' if h5_result['working'] else '✗ No'}")
    if h5_result["errors"]:
        lines.append(" Errors:")
        for error in h5_result["errors"]:
            lines.append(f" ✗ {error}")
    return "\n".join(lines)

def format_benchmark_report(gpu_test: Dict[str, Any]) -> str:
    """Format benchmark results."""
    if not gpu_test["perf"]:
        return ""
    
    lines = ["\n=== GPU BENCHMARK ==="]
    perf = gpu_test["perf"]
    lines.append(f"Time per iter: {perf['time_ms']:.2f} ms")
    lines.append(f"Throughput: {perf['throughput']:.2f} ops/sec")
    return "\n".join(lines)

# ============================================================================
# MAIN VALIDATION FUNCTION
# ============================================================================
def main(run_benchmark: bool = None, verbose: bool = True) -> Dict[str, Any]:
    """Comprehensive environment validation."""
    if run_benchmark is None:
        run_benchmark = not os.getenv("SKIP_BENCHMARK", "").lower() in ("1", "true")
    
    # Run checks
    deps = dependency_check()
    gpu_info = {
        **platform_info(),
        "torch": check_torch_mps(),
        "recommended_device": "cpu",
        "errors": [],
    }
    if gpu_info["torch"]["mps_available"]:
        gpu_info["recommended_device"] = "mps"
    
    h5_result = h5py_test()
    gpu_test = gpu_functional_test(run_benchmark)
    
    # Aggregate results
    results = {
        "dependencies": deps,
        "gpu_status": gpu_info,
        "h5py_test": h5_result,
        "gpu_functional_test": gpu_test,
        "timestamp": __import__("datetime").datetime.now().isoformat(),
    }
    
    # Check for critical failures
    required_missing = [pkg for pkg, info in deps.items() if info["required"] and not info["installed"]]
    results["critical_failures"] = required_missing
    
    # Print reports
    if verbose:
        print(format_deps_report(deps))
        print(format_gpu_report(gpu_info))
        print(format_h5py_report(h5_result))
        if run_benchmark:
            print(format_benchmark_report(gpu_test))
        if required_missing:
            print(f"\n⚠ CRITICAL: Missing required packages: {', '.join(required_missing)}")
    
    return results

if __name__ == "__main__":
    main()
