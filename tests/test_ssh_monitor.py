from pathlib import Path

import pytest

from ssh_monitor.config import HostConfig, load_config
from ssh_monitor.ssh import SSHClient, SSHError


def test_load_config_and_destination(tmp_path: Path):
    config = tmp_path / "config.toml"
    config.write_text('''[app]\noutput_dir = "~/captures"\n\n[[host]]\nalias = "lab"\nhost = "192.0.2.10"\nuser = "alice"\nport = 2222\ncapabilities = ["screenshot"]\n''')
    loaded = load_config(config)
    assert loaded.hosts[0].destination() == "alice@192.0.2.10"
    assert loaded.hosts[0].port == 2222
    assert "screenshot" in loaded.hosts[0].capabilities


def test_invalid_port_rejected(tmp_path: Path):
    config = tmp_path / "config.toml"
    config.write_text('[[host]]\nalias = "bad"\nport = 70000\n')
    with pytest.raises(ValueError, match="invalid port"):
        load_config(config)


def test_ssh_command_builds_secure_batch_mode(monkeypatch):
    calls = []

    class Result:
        stdout, stderr, returncode = "ok", "", 0

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return Result()

    monkeypatch.setattr("subprocess.run", fake_run)
    result = SSHClient(HostConfig("lab", host="example", user="alice"), 4).command("uname")
    assert result.stdout == "ok"
    assert calls[0][0][:4] == ["ssh", "-o", "BatchMode=yes", "-o"]
    assert calls[0][0][-2:] == ["alice@example", "uname"]


def test_ssh_failure_is_actionable(monkeypatch):
    class Result:
        stdout, stderr, returncode = "", "Permission denied", 255

    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: Result())
    with pytest.raises(SSHError, match="Permission denied"):
        SSHClient(HostConfig("lab"), 4).command("true")
