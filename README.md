
# SSH Monitor

Keyboard-driven, capability-aware monitoring for saved SSH hosts. Ordinary
network/process diagnostics use SSH commands. Desktop, audio, and browser
features require an explicitly authorized `ssh-monitor-agent` on the remote
host; the application never assumes SSH access grants access to another
user's graphical session or devices.

## Configure

Create `~/.config/ssh-monitor/config.toml`:

```toml
[app]
output_dir = "~/ssh-monitor-captures"
command_timeout = 15

[[host]]
alias = "lab"
host = "lab.example.org"
user = "alice"
port = 22
identity_file = "~/.ssh/id_ed25519"
capabilities = ["screenshot", "screen", "audio", "browser-tab"]
```

Host keys and authentication remain managed by OpenSSH. Do not put passwords
or private key contents in this file. The configured capabilities are labels;
the remote helper is queried before every capture request.

## Run and test

```bash
python -m ssh_monitor
pytest
```

Controls: `s` screenshot, `v` screen recording, `a` audio recording, `n`
network diagnostics, `p` processes, `t` authorized active-tab metadata, `b`
host selection, and `q` exit.

The remote helper protocol is intentionally explicit: `capabilities`,
`screenshot`, `start-recording`, `stop-recording`, and `browser-tab`. Installation
and authorization of that helper are outside this repository and must be done
by the device owner. Unsupported desktop environments, missing permissions,
and unavailable browser integrations are reported without unsafe fallbacks.
