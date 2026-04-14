# `runcmd` Design

## Goal

Provide a stable way to change the command executed from the terminal by editing a single text file in the project root. The user should only need to type one fixed command, `runcmd`, and have it execute the latest full shell command stored in the project.

## Scope

This design covers a minimal local command runner for the current project:

- A project-root command file that stores one full shell command.
- A Python runner script inside the project that reads and executes that command.
- A local PowerShell alias named `runcmd` that points to the runner script.

This design does not add scheduling, command history, command templating, multi-command batches, or safety restrictions such as a whitelist. The user explicitly wants the command file to allow arbitrary full shell commands.

## File Layout

The implementation will add the following files:

- `command.txt` at the project root. This file stores the current full command to run.
- `tools/runcmd.py` inside the project. This is the execution entrypoint used by the PowerShell alias.

The alias itself is user-machine configuration and is not stored as executable project logic. The project can still include a short setup note describing how to register the alias in the user's PowerShell profile.

## Runtime Behavior

`tools/runcmd.py` should behave as follows:

1. Resolve the project root relative to the script location, so behavior is stable even if the current terminal directory is elsewhere.
2. Read `command.txt` from the project root using UTF-8.
3. Trim leading and trailing whitespace from the file content before execution.
4. If the file is missing, empty, or contains only whitespace, print a clear error message and exit with a non-zero status.
5. Print the exact command that is about to run so the user can verify what the current file contains.
6. Execute the command in the project root directory using PowerShell, because the user wants arbitrary full shell commands instead of a restricted command subset.
7. Return the underlying command's exit code unchanged.
8. Leave `command.txt` untouched after execution so the last command remains visible and editable.

## Error Handling

The runner should distinguish between two failure classes:

- Pre-execution validation failure, such as a missing or empty `command.txt`. These should be reported by the runner itself with a non-zero exit code.
- Command execution failure, such as a bad CLI argument or a failing script. These should surface through the invoked process output, and the runner should propagate the same exit code.

The runner should not swallow stdout or stderr. The user's terminal should show the executed command output directly.

## Alias Setup

The target workflow is:

1. OpenClaw updates `command.txt` in the project root.
2. The user opens a terminal anywhere convenient.
3. The user types `runcmd`.
4. PowerShell invokes the project runner, which reads `command.txt` and runs the stored command in the project root.

The alias should point to the Python runner with an absolute path, so it remains stable regardless of the current directory.

## Constraints and Tradeoffs

The main tradeoff is deliberate: executing arbitrary full shell commands from a text file is flexible but unsafe by default. This is acceptable here because the user explicitly requested unrestricted command execution and intends to control the file content through their own workflow.

Using a Python runner instead of embedding the logic directly into the existing `mystock` CLI keeps responsibilities separate. The stock-analysis CLI remains focused on stock workflows, while `runcmd` remains a generic project-local command launcher.

## Testing

The implementation should include focused tests for:

- Missing `command.txt`
- Empty `command.txt`
- Successful execution of a simple command
- Execution from a terminal whose current directory is not the project root
- Exit-code propagation from a failing command

## Rollout

The minimal rollout should include:

1. Add `tools/runcmd.py`
2. Add or document `command.txt`
3. Add tests for the runner behavior
4. Add a short README section showing:
   - how to write the command into `command.txt`
   - how to configure the `runcmd` PowerShell alias
   - how to run the command
