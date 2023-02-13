import shlex
import subprocess
import sys


def subprocess_cmd(command: str) -> bytes:
    kwargs = {}
    kwargs["stdout"] = subprocess.PIPE
    kwargs["stderr"] = subprocess.PIPE
    proc = subprocess.Popen(shlex.split(command), **kwargs)
    (stdout_str, stderr_str) = proc.communicate()
    return_code = proc.wait()
    return f"\n{command} - return_code {return_code}\n".encode("utf-8") + stdout_str + stderr_str


def save_setup(log_dir: str) -> None:
    with open(f"{log_dir}/setup.txt", "wb") as f:
        hash = subprocess_cmd("git rev-parse --verify HEAD")
        status = subprocess_cmd("git status")

        runner = f"Triggered by command\n{str(sys.argv)}\n"
        f.write(runner.encode("utf-8") + hash + status)
