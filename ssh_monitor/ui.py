from __future__ import annotations

import sys

from .capture import AuthorizedCapture
from .config import AppConfig
from .monitoring import network_report, process_report
from .ssh import SSHClient, SSHError


class TerminalUI:
    def __init__(self, config: AppConfig):
        self.config = config

    def run(self) -> int:
        try:
            while True:
                host = self._select_host()
                if host is None:
                    return 0
                self._host_menu(host)
        except KeyboardInterrupt:
            print("\nExiting.")
            return 130

    def _select_host(self):
        if not self.config.hosts:
            print("No hosts configured. Add [[host]] entries to your config file.")
            return None
        print("\nSaved SSH hosts")
        for index, host in enumerate(self.config.hosts, 1):
            print(f"  {index}) {host.alias} ({host.destination()})")
        print("  q) Exit")
        choice = input("Select host: ").strip().lower()
        if choice == "q":
            return None
        try:
            return self.config.hosts[int(choice) - 1]
        except (ValueError, IndexError):
            print("Invalid host selection.")
            return self._select_host()

    def _host_menu(self, host) -> None:
        ssh = SSHClient(host, self.config.command_timeout)
        capture = AuthorizedCapture(ssh, self.config.output_dir)
        print(f"\nConnected operations for {host.alias} (capabilities are verified per request)")
        while True:
            print("\n[s] screenshot  [v] screen recording  [a] audio recording")
            print("[n] network     [p] processes          [t] active browser tab")
            print("[b] host selection  [q] exit")
            key = input("Action: ").strip().lower()
            if key == "b":
                return
            if key == "q":
                raise SystemExit(0)
            try:
                if key == "s":
                    print(f"Screenshot saved: {capture.screenshot()}")
                elif key == "v":
                    print(capture.toggle_recording("screen"))
                elif key == "a":
                    print(capture.toggle_recording("audio"))
                elif key == "n":
                    print(network_report(ssh))
                elif key == "p":
                    print(process_report(ssh))
                elif key == "t":
                    print(capture.browser_tab())
                else:
                    print("Unknown action. Choose one of the displayed keys.")
            except SSHError as exc:
                print(f"Error: {exc}", file=sys.stderr)
