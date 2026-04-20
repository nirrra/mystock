from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


class CommandFileError(RuntimeError):
    """Raised when the local command file cannot be used."""


def load_saved_command(command_file: Path) -> str:
    if not command_file.exists():
        raise CommandFileError(f"命令文件不存在：{command_file}")

    command = command_file.read_text(encoding="utf-8").strip()
    if not command:
        raise CommandFileError(f"命令文件为空：{command_file}")
    return command


def detect_powershell() -> str:
    for candidate in ("pwsh", "powershell"):
        executable = shutil.which(candidate)
        if executable:
            return executable
    raise CommandFileError("未找到 PowerShell 可执行文件，无法运行保存的命令。")


def run_saved_command(project_root: Path, shell_executable: str | None = None) -> int:
    root = project_root.resolve()
    command_file = root / "command.txt"
    command = load_saved_command(command_file)
    shell = shell_executable or detect_powershell()

    print(f"Executing: {command}", flush=True)
    completed = subprocess.run(
        [shell, "-NoProfile", "-Command", command],
        cwd=root,
        check=False,
    )
    return int(completed.returncode)


def main(project_root: Path) -> int:
    try:
        return run_saved_command(project_root)
    except CommandFileError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
