from __future__ import annotations

import datetime as dt
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .ssh import SSHClient, SSHError


@dataclass
class CaptureState:
    kind: str
    remote_path: str
    process: subprocess.Popen[str]


class AuthorizedCapture:
    """Adapter for an explicitly installed remote capture helper.

    The helper must be configured per host and expose `capabilities`, `screenshot`,
    `start-recording`, and `stop-recording`. No generic shell execution is exposed.
    """

    def __init__(self, ssh: SSHClient, output_dir: Path):
        self.ssh, self.output_dir, self.active = ssh, output_dir, None

    def capability(self, name: str) -> bool:
        try:
            return name in self.ssh.command("ssh-monitor-agent capabilities").stdout.split()
        except SSHError:
            return False

    def screenshot(self) -> Path:
        if not self.capability("screenshot"):
            raise SSHError("authorized desktop screenshot capability is unavailable")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        remote = f"/tmp/ssh-monitor-{stamp}.png"
        self.ssh.command(f"ssh-monitor-agent screenshot --output {shlex.quote(remote)}")
        local = self.output_dir / f"{self.ssh.host.alias}-{stamp}.png"
        self._download(remote, local)
        if not local.is_file() or local.stat().st_size == 0:
            raise SSHError("capture transfer completed but output validation failed")
        return local

    def toggle_recording(self, kind: str) -> str:
        if self.active:
            if self.active.kind != kind:
                raise SSHError(f"{self.active.kind} recording is already active")
            self.ssh.command("ssh-monitor-agent stop-recording")
            self.active = None
            return "recording stopped; remote helper finalized the output"
        if not self.capability(kind):
            raise SSHError(f"authorized {kind} recording capability is unavailable")
        result = self.ssh.command(f"ssh-monitor-agent start-recording --kind {shlex.quote(kind)}").stdout.strip()
        if not result:
            raise SSHError("recording helper did not return an output path")
        self.active = CaptureState(kind, result, subprocess.Popen(["true"], text=True))
        return f"{kind} recording started"

    def browser_tab(self) -> str:
        if not self.capability("browser-tab"):
            raise SSHError("authorized browser-tab integration is unavailable")
        return self.ssh.command("ssh-monitor-agent browser-tab --format text").stdout.strip()

    def _download(self, remote: str, local: Path) -> None:
        args = ["scp", "-q"]
        if self.ssh.host.port:
            args += ["-P", str(self.ssh.host.port)]
        args += [f"{self.ssh.host.destination()}:{remote}", str(local)]
        try:
            completed = subprocess.run(args, text=True, capture_output=True,
                                       timeout=self.ssh.timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SSHError(f"capture transfer failed: {exc}") from exc
        if completed.returncode:
            raise SSHError(f"capture transfer failed: {completed.stderr.strip()}")
