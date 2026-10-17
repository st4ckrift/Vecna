from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class HostConfig:
    alias: str
    host: str | None = None
    user: str | None = None
    port: int | None = None
    identity_file: str | None = None
    capabilities: set[str] = field(default_factory=set)

    def destination(self) -> str:
        return f"{self.user}@{self.host or self.alias}" if self.user else (self.host or self.alias)


@dataclass(frozen=True)
class AppConfig:
    hosts: list[HostConfig]
    output_dir: Path
    command_timeout: float = 15.0


def default_config_path() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "ssh-monitor" / "config.toml"


def load_config(path: str | Path | None = None) -> AppConfig:
    config_path = Path(path or default_config_path()).expanduser()
    if not config_path.exists():
        return AppConfig([], Path.home() / "ssh-monitor-captures")
    try:
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"Unable to read configuration {config_path}: {exc}") from exc

    hosts: list[HostConfig] = []
    for index, raw in enumerate(data.get("host", [])):
        if not isinstance(raw, dict) or not raw.get("alias"):
            raise ValueError(f"host entry {index + 1} must define a non-empty alias")
        port = raw.get("port")
        if port is not None and (not isinstance(port, int) or not 1 <= port <= 65535):
            raise ValueError(f"host {raw['alias']!r} has an invalid port")
        hosts.append(HostConfig(
            alias=str(raw["alias"]), host=raw.get("host"), user=raw.get("user"), port=port,
            identity_file=raw.get("identity_file"), capabilities=set(raw.get("capabilities", [])),
        ))
    app = data.get("app", {})
    output_dir = Path(app.get("output_dir", "~/ssh-monitor-captures")).expanduser()
    timeout = float(app.get("command_timeout", 15.0))
    if timeout <= 0:
        raise ValueError("app.command_timeout must be positive")
    return AppConfig(hosts, output_dir, timeout)
