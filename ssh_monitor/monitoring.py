from __future__ import annotations

from .ssh import SSHClient


def network_report(ssh: SSHClient) -> str:
    commands = {
        "interfaces": "ip -brief address 2>/dev/null || ifconfig 2>/dev/null",
        "listening ports": "ss -lntup 2>/dev/null || netstat -lntup 2>/dev/null",
        "connections": "ss -tun 2>/dev/null || netstat -tun 2>/dev/null",
        "traffic": "cat /proc/net/dev 2>/dev/null",
    }
    sections = []
    for title, command in commands.items():
        try:
            output = ssh.command(command).stdout.strip() or "(no data)"
        except Exception as exc:
            output = f"Unavailable: {exc}"
        sections.append(f"[{title}]\n{output}")
    return "\n\n".join(sections)


def process_report(ssh: SSHClient) -> str:
    command = "ps -eo pid,user,pcpu,pmem,state,comm --sort=-pcpu | head -n 31"
    try:
        return ssh.command(command).stdout.strip() or "(no processes reported)"
    except Exception as exc:
        return f"Process information unavailable: {exc}"
