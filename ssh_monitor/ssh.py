from __future__ import annotations

import subprocess
from dataclasses import dataclass

from .config import HostConfig


class SSHError(RuntimeError):
    pass


@dataclass
class SSHResult:
    stdout: str
    stderr: str
    returncode: int


class SSHClient:
    def __init__(self, host: HostConfig, timeout: float = 15.0):
        self.host = host
        self.timeout = timeout

    def command(self, command: str, timeout: float | None = None) -> SSHResult:
        args = ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={int(self.timeout)}"]
        if self.host.port:
            args += ["-p", str(self.host.port)]
        if self.host.identity_file:
            args += ["-i", self.host.identity_file]
        args += [self.host.destination(), command]
        try:
            completed = subprocess.run(args, text=True, capture_output=True,
                                       timeout=timeout or self.timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SSHError(f"SSH transport failed: {exc}") from exc
        result = SSHResult(completed.stdout, completed.stderr, completed.returncode)
        if result.returncode:
            detail = result.stderr.strip() or "remote command failed"
            raise SSHError(f"{self.host.alias}: {detail}")
        return result
