from __future__ import annotations

import subprocess
from pathlib import Path

from stocks_analyzer.runcmd import main, run_saved_command


def test_main_returns_error_when_command_file_is_missing(tmp_path: Path, capsys) -> None:
    exit_code = main(tmp_path)

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "命令文件不存在" in captured.err


def test_main_returns_error_when_command_file_is_empty(tmp_path: Path, capsys) -> None:
    (tmp_path / "command.txt").write_text(" \n", encoding="utf-8")

    exit_code = main(tmp_path)

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "命令文件为空" in captured.err


def test_run_saved_command_executes_saved_command_in_project_root(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    (tmp_path / "command.txt").write_text("Write-Output 'hello'", encoding="utf-8")
    called = {}

    def fake_run(args, cwd, check):
        called["args"] = args
        called["cwd"] = cwd
        called["check"] = check
        return subprocess.CompletedProcess(args=args, returncode=0)

    monkeypatch.setattr("stocks_analyzer.runcmd.subprocess.run", fake_run)
    monkeypatch.chdir(tmp_path.parent)

    exit_code = run_saved_command(tmp_path, shell_executable="powershell")

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Executing: Write-Output 'hello'" in captured.out
    assert called["args"] == ["powershell", "-NoProfile", "-Command", "Write-Output 'hello'"]
    assert called["cwd"] == tmp_path.resolve()
    assert called["check"] is False


def test_run_saved_command_propagates_command_exit_code(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "command.txt").write_text("exit 7", encoding="utf-8")

    def fake_run(args, cwd, check):
        return subprocess.CompletedProcess(args=args, returncode=7)

    monkeypatch.setattr("stocks_analyzer.runcmd.subprocess.run", fake_run)

    exit_code = run_saved_command(tmp_path, shell_executable="powershell")

    assert exit_code == 7
