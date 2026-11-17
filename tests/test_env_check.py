"""Tests for turkish_asr_rdl.utils.env_check. Run: pytest tests/test_env_check.py"""
import contextlib
import copy
import io
import builtins
import importlib.metadata
import subprocess
import sys
import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

import turkish_asr_rdl.utils.env_check as check_env

READY = {
    "os": "Windows-11-10.0.26200-SP0",
    "python": (3, 13, 1),
    "packages": {name: "1.0" for name in
                 ("torch", "torchcodec", "transformers", "datasets", "accelerate",
                  "peft", "jiwer", "soundfile", "librosa")},
    "driver_version": "580.88",
    "driver_error": None,
    "torch_version": "2.14.1+cu130",
    "torch_load_error": None,
    "cuda_build": "13.0",
    "cuda_available": True,
    "cuda_warning": None,
    "gpu_name": "NVIDIA GeForce GTX 1650",
    "compute_capability": (7, 5),
    "vram_bytes": 4 * 1024 ** 3,
    "bf16_supported": False,
    "arch_list": ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"],
    "gpu_op_error": None,
}


def completed(returncode=0, stdout=""):
    return subprocess.CompletedProcess(["nvidia-smi"], returncode, stdout, "")


class EvaluateTests(unittest.TestCase):
    def test_ready_gpu_has_no_problems(self):
        self.assertEqual(check_env.evaluate(READY), [])

    def test_any_patch_of_required_python_is_fine(self):
        for patch in (0, 9):
            with self.subTest(patch=patch):
                self.assertEqual(check_env.evaluate({**READY, "python": (3, 13, patch)}), [])

    def test_older_python_is_one_problem(self):
        problems = check_env.evaluate({**READY, "python": (3, 12, 7)})
        self.assertEqual(len(problems), 1)
        self.assertIn("3.12.7", problems[0])
        self.assertIn("3.13", problems[0])

    def test_newer_python_is_one_problem(self):
        self.assertEqual(len(check_env.evaluate({**READY, "python": (3, 14, 0)})), 1)

    def test_python_and_torch_problems_are_both_reported(self):
        info = {**READY, "python": (3, 12, 7), "torch_version": None}
        self.assertEqual(len(check_env.evaluate(info)), 2)

    def test_torch_missing_is_only_problem(self):
        info = {**READY, "torch_version": None, "cuda_build": None, "cuda_available": False}
        problems = check_env.evaluate(info)
        self.assertEqual(len(problems), 1)
        self.assertIn("not installed", problems[0])

    def test_torch_load_failure_shows_message(self):
        info = {**READY, "torch_version": None, "torch_load_error": "DLL load failed xyz"}
        problems = check_env.evaluate(info)
        self.assertEqual(len(problems), 1)
        self.assertIn("DLL load failed xyz", problems[0])

    def test_cpu_build(self):
        info = {**READY, "cuda_build": None, "cuda_available": False}
        problems = check_env.evaluate(info)
        self.assertEqual(len(problems), 1)
        self.assertIn("CPU-only", problems[0])

    def test_cuda_unavailable_old_driver(self):
        info = {**READY, "cuda_available": False, "driver_version": "555.97",
                "cuda_warning": "The NVIDIA driver on your system is too old"}
        problems = check_env.evaluate(info)
        self.assertEqual(len(problems), 1)
        self.assertIn("555.97", problems[0])
        self.assertIn("580", problems[0])
        self.assertIn("too old", problems[0])

    def test_cuda_unavailable_without_nvidia_smi(self):
        info = {**READY, "cuda_available": False, "driver_version": None,
                "driver_error": "nvidia-smi not found (is the NVIDIA driver installed?)"}
        problems = check_env.evaluate(info)
        self.assertEqual(len(problems), 1)
        self.assertIn("nvidia-smi not found", problems[0])

    def test_gpu_op_failure(self):
        err = "CUDA error: no kernel image is available for execution on the device"
        problems = check_env.evaluate({**READY, "gpu_op_error": err})
        self.assertEqual(len(problems), 1)
        self.assertIn("no kernel image", problems[0])

    def test_sm89_gpu_missing_from_arch_list_is_ready(self):
        info = {**READY, "compute_capability": (8, 9)}
        self.assertNotIn("sm_89", info["arch_list"])
        self.assertEqual(check_env.evaluate(info), [])

    def test_arch_in_list_but_op_fails_is_not_ready(self):
        info = {**READY, "gpu_op_error": "out of memory"}
        self.assertIn("sm_75", info["arch_list"])
        self.assertEqual(len(check_env.evaluate(info)), 1)

    def test_does_not_mutate_input(self):
        info = {**READY, "python": (3, 12, 0), "gpu_op_error": "boom"}
        before = copy.deepcopy(info)
        check_env.evaluate(info)
        self.assertEqual(info, before)


class FakeTensor:
    def __matmul__(self, other):
        return self

    def sum(self):
        return self

    def item(self):
        return 0.0


def fake_torch(ones=lambda *args, **kwargs: FakeTensor(), synchronize=lambda: None):
    return SimpleNamespace(ones=ones, cuda=SimpleNamespace(synchronize=synchronize))


def raise_kernel_error(*args, **kwargs):
    raise RuntimeError("CUDA error: no kernel image is available\nCompile with TORCH_USE_CUDA_DSA")


class GpuOpTests(unittest.TestCase):
    def test_success_is_none(self):
        self.assertIsNone(check_env.run_gpu_op(fake_torch()))

    def test_error_becomes_first_line(self):
        self.assertEqual(check_env.run_gpu_op(fake_torch(ones=raise_kernel_error)),
                         "CUDA error: no kernel image is available")

    def test_error_reported_at_synchronize_is_caught(self):
        self.assertEqual(check_env.run_gpu_op(fake_torch(synchronize=raise_kernel_error)),
                         "CUDA error: no kernel image is available")


class QueryDriverTests(unittest.TestCase):
    def run_with(self, **kw):
        with mock.patch("subprocess.run", **kw):
            return check_env.query_driver()

    def test_success(self):
        self.assertEqual(self.run_with(return_value=completed(0, "555.97\n")), ("555.97", None))

    def test_first_non_empty_line(self):
        self.assertEqual(self.run_with(return_value=completed(0, "\n 555.97 \n560.1\n")),
                         ("555.97", None))

    def test_not_found(self):
        self.assertEqual(self.run_with(side_effect=FileNotFoundError()),
                         (None, "nvidia-smi not found (is the NVIDIA driver installed?)"))

    def test_timeout(self):
        err = subprocess.TimeoutExpired("nvidia-smi", 10)
        self.assertEqual(self.run_with(side_effect=err),
                         (None, "nvidia-smi did not answer within 10 s"))

    def test_nonzero_exit(self):
        self.assertEqual(self.run_with(return_value=completed(9)),
                         (None, "nvidia-smi failed with exit code 9"))

    def test_other_oserror(self):
        self.assertEqual(self.run_with(side_effect=PermissionError("denied")),
                         (None, "nvidia-smi could not run: denied"))

    def test_uses_list_args_and_timeout(self):
        with mock.patch("subprocess.run", return_value=completed(0, "1\n")) as run:
            check_env.query_driver()
        args, kwargs = run.call_args
        self.assertIsInstance(args[0], list)
        self.assertEqual(kwargs["timeout"], check_env.NVIDIA_SMI_TIMEOUT_S)


class FormatReportTests(unittest.TestCase):
    def test_ready_verdict(self):
        text = check_env.format_report(READY, [])
        self.assertIn("READY", text)
        self.assertNotIn("NOT READY", text)

    def test_not_ready_lists_problems(self):
        text = check_env.format_report(READY, ["first one", "second one"])
        self.assertIn("NOT READY", text)
        self.assertIn("  - first one", text)
        self.assertIn("  - second one", text)

    def test_gpu_facts(self):
        text = check_env.format_report(READY, [])
        self.assertIn("sm_75", text)
        self.assertIn("4.0 GiB", text)
        self.assertIn("7.5 (sm_75)", text)
        self.assertIn("580.88", text)
        self.assertIn("(CUDA 13.0)", text)

    def test_missing_torch_and_packages(self):
        info = {**READY, "torch_version": None, "cuda_build": None, "cuda_available": False,
                "gpu_name": None, "compute_capability": None, "vram_bytes": None, "arch_list": [],
                "packages": {**READY["packages"], "peft": None}}
        text = check_env.format_report(info, ["x"])
        self.assertIn("not installed", text)
        self.assertIn("  peft: not installed", text)
        self.assertNotIn("VRAM", text)

    def test_torch_load_failure_is_not_reported_as_missing(self):
        info = {**READY, "torch_version": None, "torch_load_error": "DLL load failed",
                "cuda_build": None, "cuda_available": False, "gpu_name": None}
        text = check_env.format_report(info, ["x"])
        self.assertIn("torch: installed but failed to load", text)
        self.assertNotIn("torch: not installed", text)


class MainTests(unittest.TestCase):
    def run_main(self, **patch_kw):
        out = io.StringIO()
        with mock.patch.object(check_env, "gather", **patch_kw), contextlib.redirect_stdout(out):
            code = check_env.main()
        return code, out.getvalue()

    def test_ready_returns_zero(self):
        self.assertEqual(self.run_main(return_value=READY)[0], 0)

    def test_broken_returns_one(self):
        code, out = self.run_main(return_value={**READY, "python": (3, 12, 0)})
        self.assertEqual(code, 1)
        self.assertIn("NOT READY", out)

    def test_unexpected_error_is_one_line(self):
        code, out = self.run_main(side_effect=ValueError("bad\nmore"))
        self.assertEqual(code, 1)
        self.assertEqual(out.strip().splitlines(), ["check_env could not finish: ValueError: bad"])


DRIVER_WARNING = ("The NVIDIA driver on your system is too old (found version 12050).\n"
                  "Please update your GPU driver")
TORCH_KEYS = {"torch_version", "torch_load_error", "cuda_build", "cuda_available", "cuda_warning",
              "gpu_name", "compute_capability", "vram_bytes", "bf16_supported", "arch_list",
              "gpu_op_error"}
VRAM = 4 * 1024 ** 3
ARCHES = ["sm_75", "sm_86"]


def forbidden_device_call(*args, **kwargs):
    raise AssertionError("device function must not be called")


def cuda_ones(shape, device=None):
    if device != "cuda":
        raise AssertionError("the test tensor must be created on the GPU")
    return FakeTensor()


def stand_in_torch(cuda_build="13.0", available=True, warning=None,
                   ones=cuda_ones, synchronize=lambda: None, forbid_device=False):
    def is_available():
        if warning:
            warnings.warn(warning)
        return available

    device = {
        "get_device_properties": lambda index: SimpleNamespace(name="Fake GPU", total_memory=VRAM),
        "get_device_capability": lambda index: (7, 5),
        # native bf16 is off on this GPU; only emulation would report True
        "is_bf16_supported": lambda including_emulation=True: including_emulation,
        "get_arch_list": lambda: list(ARCHES),
    }
    if forbid_device:
        device = dict.fromkeys(device, forbidden_device_call)
        ones = synchronize = forbidden_device_call
    cuda = SimpleNamespace(is_available=is_available, synchronize=synchronize, **device)
    return SimpleNamespace(__version__="2.14.1+cu130", version=SimpleNamespace(cuda=cuda_build),
                           ones=ones, cuda=cuda)


def facts_with(fake):
    with mock.patch.dict(sys.modules, {"torch": fake}):
        return check_env.torch_facts()


class TorchFactsTests(unittest.TestCase):
    def test_cuda_gpu_facts_are_all_reported(self):
        self.assertEqual(facts_with(stand_in_torch()), {
            "torch_version": "2.14.1+cu130", "torch_load_error": None, "cuda_build": "13.0",
            "cuda_available": True, "cuda_warning": None, "gpu_name": "Fake GPU",
            "compute_capability": (7, 5), "vram_bytes": VRAM, "bf16_supported": False,
            "arch_list": ARCHES, "gpu_op_error": None})

    def test_cpu_build_skips_device_queries(self):
        fake = stand_in_torch(cuda_build=None, available=False, forbid_device=True)
        facts = facts_with(fake)
        self.assertIsNone(facts["cuda_build"])
        self.assertIs(facts["cuda_available"], False)
        self.assertIsNone(facts["gpu_name"])
        self.assertEqual(facts["arch_list"], [])
        self.assertEqual(facts["torch_version"], "2.14.1+cu130")

    def test_unavailable_cuda_keeps_first_warning_line_and_prints_nothing(self):
        err, out = io.StringIO(), io.StringIO()
        fake = stand_in_torch(available=False, warning=DRIVER_WARNING, forbid_device=True)
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
            facts = facts_with(fake)
        self.assertIs(facts["cuda_available"], False)
        self.assertEqual(facts["cuda_warning"],
                         "The NVIDIA driver on your system is too old (found version 12050).")
        self.assertEqual((err.getvalue(), out.getvalue()), ("", ""))

    def test_warning_is_captured_even_when_caller_ignores_warnings(self):
        fake = stand_in_torch(available=False, warning=DRIVER_WARNING, forbid_device=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            facts = facts_with(fake)
        self.assertEqual(facts["cuda_warning"],
                         "The NVIDIA driver on your system is too old (found version 12050).")

    def test_gpu_op_failure_keeps_other_gpu_facts(self):
        facts = facts_with(stand_in_torch(ones=raise_kernel_error))
        self.assertEqual(facts["gpu_op_error"], "CUDA error: no kernel image is available")
        self.assertEqual((facts["gpu_name"], facts["vram_bytes"], facts["compute_capability"]),
                         ("Fake GPU", VRAM, (7, 5)))

    def test_missing_torch_is_not_a_load_error(self):
        facts = facts_with(None)
        self.assertIsNone(facts["torch_version"])
        self.assertIsNone(facts["torch_load_error"])

    def test_dll_failure_is_reported_as_first_line(self):
        real_import = builtins.__import__

        def failing_import(name, *args, **kwargs):
            if name == "torch":
                raise OSError("[WinError 126] The specified module could not be found. "
                              "Error loading c10.dll\nmore")
            return real_import(name, *args, **kwargs)

        with mock.patch.object(builtins, "__import__", failing_import):
            facts = check_env.torch_facts()
        self.assertEqual(facts["torch_load_error"], "[WinError 126] The specified module could "
                         "not be found. Error loading c10.dll")
        self.assertIsNone(facts["torch_version"])


class GatherTests(unittest.TestCase):
    def test_gather_combines_system_driver_packages_and_torch_facts(self):
        with (mock.patch.object(check_env, "query_driver", return_value=("580.88", None)),
              mock.patch.dict(sys.modules, {"torch": stand_in_torch()})):
            info = check_env.gather()
        self.assertEqual(set(info), {"os", "python", "packages", "driver_version",
                                     "driver_error"} | TORCH_KEYS)
        self.assertEqual(info["python"], tuple(sys.version_info[:3]))
        self.assertEqual((info["driver_version"], info["driver_error"]), ("580.88", None))
        self.assertEqual(set(info["packages"]), set(check_env.CORE_PACKAGES))
        self.assertEqual(check_env.torch_problems(info), [])


class PackageVersionsTests(unittest.TestCase):
    def test_installed_package_has_version_and_others_are_none(self):
        def version(name):
            if name == "jiwer":
                return "1.0"
            raise importlib.metadata.PackageNotFoundError(name)

        target = "turkish_asr_rdl.utils.env_check.importlib.metadata.version"
        with mock.patch(target, side_effect=version):
            versions = check_env.package_versions()
        self.assertEqual(list(versions), list(check_env.CORE_PACKAGES))
        self.assertEqual(versions["jiwer"], "1.0")
        self.assertEqual([n for n, v in versions.items() if v is not None], ["jiwer"])


class OneLineTests(unittest.TestCase):
    def test_empty_message_falls_back_to_type_name(self):
        self.assertEqual(check_env.one_line(RuntimeError()), "RuntimeError")

    def test_blank_lines_are_skipped(self):
        self.assertEqual(check_env.one_line(ValueError("\n  \nsecond")), "second")


if __name__ == "__main__":
    unittest.main()
