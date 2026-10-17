from .config import load_config
from .ui import TerminalUI


def main() -> int:
    try:
        return TerminalUI(load_config()).run()
    except ValueError as exc:
        print(f"Configuration error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
