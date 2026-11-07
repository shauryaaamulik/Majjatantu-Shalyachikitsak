"""Minimal CLI entrypoint."""

from __future__ import annotations

import argparse

from majjatantu import __version__


def main() -> None:
    parser = argparse.ArgumentParser(prog="majjatantu", description="MAJJATANTU toolkit")
    parser.add_argument("--version", action="version", version=f"majjatantu {__version__}")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("info", help="Print package status")
    args = parser.parse_args()

    if args.cmd == "info" or args.cmd is None:
        print(f"majjatantu {__version__}")
        print("Phase I: foundations — introspection, direction ablation, eval harness")
        print("Repairs are gradient-free. Forward-pass budgets are mandatory.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
