#!/usr/bin/env python3

"""
b580_validator.py

Stage 0 environment validator for Arch Linux + Intel Arc (XPU) pipeline.

Checks Python dependencies, Intel XPU availability in PyTorch, and HDF5 I/O.
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
    """Try importing each dependency and record status."""
    results: Dict[str, Dict[str, Any]] = {}
    for name, (required, description) in DEPENDENCIES.items():
        try:
            module = import_module(name)
            version = getattr(module, "__version__", "unknown")
            installed = True
        except ImportError:
            module = None  # noqa: F841
            version = None
            installed = False
        results[name] = {
            "installed": installed,
            "version": version,
            "required": required,
            "status": "✓" if installed else "✗",
            "description": description,
        }
    return results


# ============================================================================
# PLATFORM + GPU AVAILABILITY
# ============================================================================

def platform_info() -> Dict[str, Any]:
    """Return basic OS/arch information; expect Arch + x86_64."""
    try:
        arch = subprocess.check_output(["uname", "-m"], text=True).strip()
    except Exception:
        arch = "unknown"

    try:
        kernel = subprocess.check_output(["uname", "-r"], text=True).strip()
    except Exception:
        kernel = "unknown"

    os_id = "unknown"
    try:
        with open("/etc/os-release", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("ID="):
                    os_id = line.strip().split("=", 1)[1].strip('"')
                    break
    except Exception:
        pass

    return {
        "arch": arch,
        "kernel": kernel,
        "os_id": os_id,
        "is_arch": os_id == "arch",
        "is_x86_64": arch == "x86_64",
        "is_unix": True,
    }


def low_level_gpu_info() -> Dict[str, Any]:
    """
    Quick low-level checks: /dev/dri and clinfo output.

    This is complementary to torch.xpu, and helps debug driver/runtime issues.
    """
    info: Dict[str, Any] = {
        "dri_entries": [],
        "has_render_node": False,
        "clinfo_ok": False,
        "clinfo_error": None,
    }

    dri_path = "/dev/dri"
    if os.path.isdir(dri_path):
        try:
            entries = sorted(os.listdir(dri_path))
            info["dri_entries"] = entries
            info["has_render_node"] = any(e.startswith("renderD") for e in entries)
        except Exception as e:
            info["dri_entries"] = [f"error: {e}"]

    if shutil_which("clinfo") is not None:
        try:
            out = subprocess.check_output(
                ["clinfo"],
                stderr=subprocess.STDOUT,
                text=True,
                timeout=10,
            )
            # Very loose heuristic: just check that something mentioning Intel shows up.
            info["clinfo_ok"] = "Intel" in out
        except Exception as e:
            info["clinfo_error"] = str(e)

    return info


def shutil_which(cmd: str) -> Optional[str]:
    """Local tiny replacement for shutil.which to avoid importing it globally."""
    for path in os.getenv("PATH", "").split(os.pathsep):
        full = os.path.join(path, cmd)
        if os.path.isfile(full) and os.access(full, os.X_OK):
            return full
    return None


def check_torch_xpu() -> Dict[str, Any]:
    """
    Check whether PyTorch is installed and whether Intel XPU is usable.

    Uses the official torch.xpu API as documented by PyTorch's Intel GPU guide. [web:14][web:87]
    """
    try:
        import torch  # type: ignore

        has_xpu_module = hasattr(torch, "xpu")
        xpu_available = bool(has_xpu_module and torch.xpu.is_available())
        device_count = torch.xpu.device_count() if xpu_available else 0
        device_name = (
            torch.xpu.get_device_name(0) if xpu_available and device_count > 0 else None
        )

        return {
            "installed": True,
            "version": torch.__version__,
            "xpu_module": has_xpu_module,
            "xpu_available": xpu_available,
            "device_count": device_count,
            "device_name": device_name,
            "device": "xpu" if xpu_available else "cpu",
            "errors": [],
        }
    except ImportError:
        return {
            "installed": False,
            "version": None,
            "xpu_module": False,
            "xpu_available": False,
            "device_count": 0,
            "device_name": None,
            "device": "cpu",
            "errors": ["PyTorch not installed"],
        }
    except Exception as e:
        return {
            "installed": True,
            "version": None,
            "xpu_module": True,
            "xpu_available": False,
            "device_count": 0,
            "device_name": None,
            "device": "cpu",
            "errors": [f"torch.xpu check failed: {e}"],
        }


# ============================================================================
# HDF5 FUNCTIONAL TEST
# ============================================================================

def h5py_test() -> Dict[str, Any]:
    """
    Write and read back a small HDF5 file approximating your Stage 2 structure.
    """
    result: Dict[str, Any] = {"working": False, "errors": []}
    try:
        import h5py  # type: ignore
        import numpy as np  # type: ignore

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            fname = f.name

        try:
            with h5py.File(fname, "w") as hf:
                hf.create_dataset(
                    "voltagemaps",
                    data=np.random.randn(1, 96, 96).astype(np.float32),
                    compression="gzip",
                )
                hf.create_dataset(
                    "qhisequences",
                    data=np.random.randn(1, 151).astype(np.float32),
                )
                hf.create_dataset(
                    "thisequences",
                    data=np.random.randn(1, 151).astype(np.float32),
                )
                hf.create_dataset(
                    "scalarfeatures",
                    data=np.random.randn(1, 15).astype(np.float32),
                )
                hf.attrs["vehicle_id"] = "test_vin_123"
                hf.attrs["timestamp"] = "2025-12-05T19:49:00"

            with h5py.File(fname, "r") as hf:
                assert hf["voltagemaps"].shape == (1, 96, 96)
                assert hf["qhisequences"].shape == (1, 151)
                assert hf.attrs["vehicle_id"] == "test_vin_123"

            result["working"] = True
        finally:
            try:
                os.unlink(fname)
            except OSError:
                pass
    except Exception as e:
        result["errors"].append(f"HDF5 test failed: {e}")
    return result


# ============================================================================
# GPU FUNCTIONAL TEST
# ============================================================================

def benchmark_op(
    op: Callable[[], None],
    sync: Optional[Callable[[], None]] = None,
    warmup: int = 3,
    iters: int = 10,
) -> Dict[str, float]:
    """Run op repeatedly with warmup and optional device synchronization."""
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
        "time_ms": elapsed * 1000.0 / iters,
        "throughput": iters / elapsed,
    }


def gpu_functional_test(run_benchmark: bool = True) -> Dict[str, Any]:
    """
    Run a few tensor operations on the chosen device (XPU if available, else CPU)
    and optionally benchmark a matrix multiply. [web:14][web:9]
    """
    result: Dict[str, Any] = {
        "working": False,
        "tensor_ops": False,
        "perf": None,
        "errors": [],
    }

    torch_info = check_torch_xpu()
    if not torch_info["installed"]:
        result["errors"].extend(torch_info["errors"])
        return result

    try:
        import torch  # type: ignore

        device = torch_info["device"]
        # Basic tensor + linear model check
        x = torch.randn(100, 100, device=device)
        torch.matmul(x, x)
        model = torch.nn.Linear(100, 50).to(device)
        model(torch.randn(32, 100, device=device))
        result["tensor_ops"] = True

        # Optional benchmark if running on XPU
        if run_benchmark and device == "xpu":
            def op() -> None:
                a = torch.randn(1000, 1000, device=device)
                b = torch.randn(1000, 1000, device=device)
                torch.matmul(a, b)

            sync = getattr(torch.xpu, "synchronize", None)
            result["perf"] = benchmark_op(op, sync)
        result["working"] = True
    except Exception as e:
        result["errors"].append(f"Tensor ops failed: {e}")

    return result


# ============================================================================
# REPORTING
# ============================================================================

def format_deps_report(deps: Dict[str, Dict[str, Any]]) -> str:
    lines = ["\n=== DEPENDENCY STATUS ==="]
    required_missing = [
        name for name, info in deps.items()
        if info["required"] and not info["installed"]
    ]

    for name, info in sorted(deps.items()):
        status = "✓" if info["installed"] else "✗"
        version = f"v{info['version']}" if info["installed"] else "missing"
        lines.append(f" {status} {name:20} {version:15} {info['description']}")

    if required_missing:
        lines.append(
            f"\n⚠ CRITICAL: Missing required packages: {', '.join(required_missing)}"
        )
    return "\n".join(lines)


def format_gpu_report(
    plat: Dict[str, Any],
    torch_info: Dict[str, Any],
    low_level: Dict[str, Any],
    recommended_device: str,
) -> str:
    lines = ["\n=== GPU / ACCELERATOR STATUS ==="]
    lines.append(f"OS ID: {plat['os_id']}")
    lines.append(f"Arch: {plat['arch']}")
    lines.append(f"Kernel: {plat['kernel']}")
    lines.append(f"Arch Linux: {'Yes' if plat['is_arch'] else 'No'}")
    lines.append(f"x86_64: {'Yes' if plat['is_x86_64'] else 'No'}")

    lines.append("\nPyTorch XPU:")
    lines.append(f" Installed: {'Yes' if torch_info['installed'] else 'No'}")
    if torch_info["installed"]:
        lines.append(f" Version: {torch_info['version']}")
        lines.append(f" XPU module present: {'Yes' if torch_info['xpu_module'] else 'No'}")
        lines.append(f" XPU available: {'Yes' if torch_info['xpu_available'] else 'No'}")
        lines.append(f" Device count: {torch_info['device_count']}")
        if torch_info["device_name"]:
            lines.append(f" Device[0] name: {torch_info['device_name']}")
        if torch_info["errors"]:
            lines.append(f" Errors: {'; '.join(torch_info['errors'])}")

    lines.append("\nLow-level GPU (DRM / OpenCL):")
    lines.append(f" /dev/dri entries: {', '.join(low_level['dri_entries']) or 'None'}")
    lines.append(f" Render node present: {'Yes' if low_level['has_render_node'] else 'No'}")
    lines.append(f" clinfo status: {'OK' if low_level['clinfo_ok'] else 'Not OK or missing'}")
    if low_level["clinfo_error"]:
        lines.append(f" clinfo error: {low_level['clinfo_error']}")

    lines.append(f"\nRecommended device: {recommended_device.upper()}")
    return "\n".join(lines)


def format_h5py_report(h5_result: Dict[str, Any]) -> str:
    lines = ["\n=== HDF5 I/O TEST ==="]
    lines.append(f"Working: {'✓ Yes' if h5_result['working'] else '✗ No'}")
    if h5_result["errors"]:
        lines.append(" Errors:")
        for err in h5_result["errors"]:
            lines.append(f"  ✗ {err}")
    return "\n".join(lines)


def format_benchmark_report(gpu_test: Dict[str, Any]) -> str:
    if not gpu_test["perf"]:
        return ""
    perf = gpu_test["perf"]
    lines = ["\n=== GPU BENCHMARK (XPU) ==="]
    lines.append(f"Time per iter: {perf['time_ms']:.2f} ms")
    lines.append(f"Throughput: {perf['throughput']:.2f} ops/sec")
    return "\n".join(lines)


# ============================================================================
# MAIN ENTRYPOINT
# ============================================================================

def main(run_benchmark: Optional[bool] = None, verbose: bool = True) -> Dict[str, Any]:
    """
    End-to-end validation entrypoint, similar to macos_validator.main. [file:1]
    """
    if run_benchmark is None:
        run_benchmark = os.getenv("SKIP_BENCHMARK", "").lower() not in ("1", "true")

    deps = dependency_check()
    plat = platform_info()
    torch_info = check_torch_xpu()
    low_level = low_level_gpu_info()
    recommended_device = "cpu"
    if torch_info["xpu_available"]:
        recommended_device = "xpu"

    h5_result = h5py_test()
    gpu_test = gpu_functional_test(run_benchmark=run_benchmark)

    results: Dict[str, Any] = {
        "dependencies": deps,
        "platform": plat,
        "torch_xpu": torch_info,
        "low_level_gpu": low_level,
        "h5py_test": h5_result,
        "gpu_functional_test": gpu_test,
        "recommended_device": recommended_device,
        "timestamp": __import__("datetime").datetime.now().isoformat(),
    }

    required_missing = [
        name for name, info in deps.items()
        if info["required"] and not info["installed"]
    ]
    results["critical_failures"] = required_missing

    if verbose:
        print(format_deps_report(deps))
        print(format_gpu_report(plat, torch_info, low_level, recommended_device))
        print(format_h5py_report(h5_result))
        if run_benchmark:
            bench_text = format_benchmark_report(gpu_test)
            if bench_text:
                print(bench_text)
        if required_missing:
            print(
                f"\n⚠ CRITICAL: Missing required packages: {', '.join(required_missing)}"
            )

    return results


if __name__ == "__main__":
    main()
