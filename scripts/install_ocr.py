from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BackendConfig:
    extra: str
    runtime_package: str
    no_sources_packages: tuple[str, ...] = ()


BACKENDS: dict[str, BackendConfig] = {
    "cpu": BackendConfig(extra="paddle-cpu", runtime_package="paddlepaddle", no_sources_packages=("paddlepaddle-gpu",)),
    "gpu": BackendConfig(extra="paddle-gpu", runtime_package="paddlepaddle-gpu"),
    "gpu-nvidia": BackendConfig(extra="paddle-gpu-nvidia", runtime_package="paddlepaddle-gpu"),
}
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_command(config: BackendConfig, *, retry: bool) -> list[str]:
    command = ["uv", "sync", "--locked", "--extra", "ocr", "--extra", config.extra]
    for package_name in config.no_sources_packages:
        command.extend(["--no-sources-package", package_name])

    if retry:
        for package_name in ("paddleocr", config.runtime_package):
            command.extend(["--refresh-package", package_name, "--reinstall-package", package_name])

    return command


def install_ocr(backend: str, *, max_attempts: int) -> int:
    uv_path = shutil.which("uv")
    if uv_path is None:
        print("uv is required to install OCR dependencies.", file=sys.stderr)
        return 1

    config = BACKENDS[backend]
    for attempt in range(1, max_attempts + 1):
        command = build_command(config, retry=attempt > 1)
        command[0] = uv_path
        print(f"[autocv] OCR install attempt {attempt}/{max_attempts}: {' '.join(command)}")
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)  # noqa: S603
        if completed.returncode == 0:
            return 0
        if attempt < max_attempts:
            print(
                "[autocv] OCR install failed; retrying with a targeted Paddle/PaddleOCR reinstall.",
                file=sys.stderr,
            )

    return completed.returncode


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install optional AutoCV OCR dependencies with a targeted retry.")
    parser.add_argument(
        "--backend",
        choices=sorted(BACKENDS),
        default="cpu",
        help="OCR backend to install. Defaults to paddle CPU support.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="Maximum install attempts before failing. Defaults to 2.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.max_attempts < 1:
        print("--max-attempts must be at least 1.", file=sys.stderr)
        return 2
    return install_ocr(args.backend, max_attempts=args.max_attempts)


if __name__ == "__main__":
    raise SystemExit(main())
