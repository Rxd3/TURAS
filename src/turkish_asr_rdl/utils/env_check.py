"""Print an environment fingerprint for the ASR project and say whether it is ready.

Run inside the project venv:  python scripts/check_env.py
Exit code 0 means ready, 1 means not ready (the reasons are printed).
"""
import importlib.metadata
import platform
import subprocess
import sys
import warnings

REQUIRED_PYTHON = (3, 13)
CORE_PACKAGES = ("torch", "torchcodec", "transformers", "datasets", "accelerate",
                 "peft", "jiwer", "soundfile", "librosa")
MIN_DRIVER = "580"  # NVIDIA driver needed by the CUDA 13.0 build of torch
NVIDIA_SMI_TIMEOUT_S = 10
GPU_TEST_SIZE = 64
BYTES_PER_GIB = 1024 ** 3
EXIT_READY = 0
EXIT_NOT_READY = 1
INSTALL_HINT = "pip install -r requirements/windows-cu130.txt"


def one_line(exc: BaseException) -> str:
    """First non-empty line of the message, or the exception type name."""
    for line in str(exc).splitlines():
        if line.strip():
            return line.strip()
    return type(exc).__name__


def query_driver() -> tuple[str | None, str | None]:
    """Return (driver_version, None) or (None, friendly error)."""
    cmd = ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=NVIDIA_SMI_TIMEOUT_S, check=False)
    except FileNotFoundError:
        return None, "nvidia-smi not found (is the NVIDIA driver installed?)"
    except subprocess.TimeoutExpired:
        return None, f"nvidia-smi did not answer within {NVIDIA_SMI_TIMEOUT_S} s"
    except OSError as exc:
        return None, f"nvidia-smi could not run: {one_line(exc)}"
    if result.returncode != 0:
        return None, f"nvidia-smi failed with exit code {result.returncode}"
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return (lines[0], None) if lines else (None, "nvidia-smi reported no driver version")


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in CORE_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def run_gpu_op(torch) -> str | None:
    """Run a small matmul on the GPU; return an error line or None if it worked."""
    try:
        a = torch.ones((GPU_TEST_SIZE, GPU_TEST_SIZE), device="cuda")
        (a @ a).sum().item()  # .item() waits for the kernel; CUDA errors are async
        torch.cuda.synchronize()
    except RuntimeError as exc:
        return one_line(exc)
    return None


def torch_facts() -> dict:
    facts = {"torch_version": None, "torch_load_error": None, "cuda_build": None,
             "cuda_available": False, "cuda_warning": None, "gpu_name": None,
             "compute_capability": None, "vram_bytes": None, "bf16_supported": None,
             "arch_list": [], "gpu_op_error": None}
    try:
        import torch  # lazy: torch may be missing or broken
    except ImportError:
        return facts
    except OSError as exc:  # Windows DLL load failure
        return {**facts, "torch_load_error": one_line(exc)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        available = bool(torch.cuda.is_available())
    facts.update(torch_version=torch.__version__, cuda_build=torch.version.cuda,
                 cuda_available=available,
                 cuda_warning=one_line(caught[0].message) if caught else None)
    if not available:
        return facts
    props = torch.cuda.get_device_properties(0)
    facts.update(gpu_name=props.name, vram_bytes=props.total_memory,
                 compute_capability=tuple(torch.cuda.get_device_capability(0)),
                 bf16_supported=torch.cuda.is_bf16_supported(including_emulation=False),
                 arch_list=list(torch.cuda.get_arch_list()), gpu_op_error=run_gpu_op(torch))
    return facts


def gather() -> dict:
    driver_version, driver_error = query_driver()
    return {"os": platform.platform(), "python": tuple(sys.version_info[:3]),
            "packages": package_versions(), "driver_version": driver_version,
            "driver_error": driver_error, **torch_facts()}


def evaluate(info: dict) -> list[str]:
    """Return the problems that block training; empty means ready."""
    problems = []
    if tuple(info["python"][:2]) != REQUIRED_PYTHON:
        found = ".".join(str(p) for p in info["python"])
        need = ".".join(str(p) for p in REQUIRED_PYTHON)
        problems.append(f"Python {found} found; this project needs Python {need}.x "
                        "(see docs/reproducibility.md).")
    problems.extend(torch_problems(info))
    return problems


def torch_problems(info: dict) -> list[str]:
    # arch_list is deliberately ignored: sm_89 runs via binary compatibility; the GPU op decides.
    if info["torch_version"] is None:
        if info["torch_load_error"]:
            return [f"torch is installed but failed to load ({info['torch_load_error']}). "
                    f"Reinstall it: {INSTALL_HINT}"]
        return [f"torch is not installed. Activate .venv, then run: {INSTALL_HINT}"]
    if info["cuda_build"] is None:
        return [f"torch {info['torch_version']} is a CPU-only build. Reinstall it: {INSTALL_HINT}"]
    if not info["cuda_available"]:
        reason = info["cuda_warning"] or "no reason given"
        driver = info["driver_version"] or info["driver_error"]
        return [f"CUDA is not available to torch ({reason}). NVIDIA driver: {driver}. "
                f"Update the NVIDIA driver to {MIN_DRIVER} or newer (see docs/reproducibility.md)."]
    if info["gpu_op_error"]:
        return [f"A test operation on the GPU failed: {info['gpu_op_error']}"]
    return []


def format_report(info: dict, problems: list[str]) -> str:
    lines = [f"OS: {info['os']}", "Python: " + ".".join(str(p) for p in info["python"])]
    if info["torch_version"]:
        cuda = f" (CUDA {info['cuda_build']})" if info["cuda_build"] else ""
        lines.append(f"torch: {info['torch_version']}{cuda}")
    elif info["torch_load_error"]:
        lines.append("torch: installed but failed to load")
    else:
        lines.append("torch: not installed")
    lines.append(f"NVIDIA driver: {info['driver_version'] or info['driver_error']}")
    if info["gpu_name"] is not None:
        major, minor = info["compute_capability"]
        lines += [f"GPU: {info['gpu_name']}",
                  f"Compute capability: {major}.{minor} (sm_{major}{minor})",
                  f"VRAM: {info['vram_bytes'] / BYTES_PER_GIB:.1f} GiB",
                  f"bf16 supported: {'yes' if info['bf16_supported'] else 'no'} (info only)",
                  f"torch arch list: {' '.join(info['arch_list'])} (info only)"]
    lines.append("Packages:")
    lines += [f"  {name}: {version or 'not installed'}" for name, version in info["packages"].items()]
    lines.append("NOT READY" if problems else "READY")
    lines += [f"  - {problem}" for problem in problems]
    return "\n".join(lines)


def main() -> int:
    try:
        info = gather()
        problems = evaluate(info)
        print(format_report(info, problems))
    except Exception as exc:  # last resort: one friendly line, no traceback
        print(f"check_env could not finish: {type(exc).__name__}: {one_line(exc)}")
        return EXIT_NOT_READY
    return EXIT_NOT_READY if problems else EXIT_READY
