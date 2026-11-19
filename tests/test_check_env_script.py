"""Tests for the scripts/check_env.py entry point. Run: pytest tests/test_check_env_script.py"""
import contextlib
import io
import runpy
import sys
import unittest
from pathlib import Path
from unittest import mock

import turkish_asr_rdl.utils.env_check as check_env

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_env.py"


class CheckEnvScriptTests(unittest.TestCase):
    def run_script(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as caught:
            runpy.run_path(str(SCRIPT), run_name="__main__")
        return caught.exception.code, out.getvalue()

    def test_exit_code_comes_from_main(self):
        with mock.patch.object(check_env, "main", return_value=0):
            self.assertEqual(self.run_script()[0], 0)

    def test_missing_package_is_one_line_and_exit_one(self):
        with mock.patch.dict(sys.modules, {"turkish_asr_rdl.utils.env_check": None}):
            code, out = self.run_script()
        self.assertEqual(code, 1)
        lines = out.strip().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertIn("turkish_asr_rdl is not importable", lines[0])
        self.assertIn("pip install --no-deps -e .", lines[0])


if __name__ == "__main__":
    unittest.main()
