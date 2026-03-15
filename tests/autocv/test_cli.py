from __future__ import annotations

import io
import runpy
from unittest.mock import patch

from autocv import cli


def test_main_writes_runtime_report_to_stderr():
    stderr = io.StringIO()

    with (
        patch.object(cli._about, "__file__", r"C:\repo\autocv\_about.py", create=True),
        patch.object(cli._about, "__git_sha1__", "1234567890abcdef", create=True),
        patch.object(cli._about, "__version__", "9.9.9", create=True),
        patch("autocv.cli.platform.python_implementation", return_value="CPython"),
        patch("autocv.cli.platform.python_version", return_value="3.12.1"),
        patch("autocv.cli.platform.python_compiler", return_value="MSC v.1942"),
        patch("autocv.cli.platform.uname", return_value=("Windows", "host", "11", "22631", "AMD64", "")),
        patch("autocv.cli.sys.stderr", stderr),
    ):
        cli.main()

    assert stderr.getvalue().splitlines() == [
        "autocv (9.9.9) [12345678]",
        r"located at C:\repo\autocv",
        "CPython 3.12.1 MSC v.1942",
        "Windows host 11 22631 AMD64",
    ]


def test_python_m_autocv_invokes_cli_main():
    with patch("autocv.cli.main") as mock_main:
        runpy.run_module("autocv.__main__", run_name="__main__")

    mock_main.assert_called_once_with()
