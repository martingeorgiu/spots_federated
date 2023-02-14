import datetime
import os
import shlex
import subprocess
import sys
from pathlib import Path


def subprocess_cmd(command: str) -> bytes:
    kwargs = {}
    kwargs["stdout"] = subprocess.PIPE
    kwargs["stderr"] = subprocess.PIPE
    proc = subprocess.Popen(shlex.split(command), **kwargs)
    (stdout_str, stderr_str) = proc.communicate()
    return_code = proc.wait()
    return f"\n{command} - return_code {return_code}\n".encode("utf-8") + stdout_str + stderr_str


def save_setup(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)

    with open(f"{log_dir}/setup.txt", "wb") as f:
        hash = subprocess_cmd("git rev-parse --verify HEAD")
        status = subprocess_cmd("git status")

        start_time = datetime.datetime.now().replace(microsecond=0).isoformat()
        runner = f"Triggered by command\n{str(sys.argv)}\n\nTime started\n{start_time}\n"

        f.write(runner.encode("utf-8") + hash + status)
