#!/usr/bin/env python3
"""
Main entry point for morphogenetic_engine.cli package.

This module allows the CLI to be run as:
    python -m morphogenetic_engine.cli sweep [args...]
    python -m morphogenetic_engine.cli reports [args...]
"""

import sys
from typing import List, Optional

from .reports import ReportsCLI
from .sweep import SweepCLI


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI package."""
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        print("Usage: python -m morphogenetic_engine.cli {sweep|reports} [args...]")
        print("Available commands:")
        print("  sweep   - Run hyperparameter sweeps")
        print("  reports - Generate and analyze sweep reports")
        return 1

    command = argv[0]
    args = argv[1:]

    if command == "sweep":
        cli = SweepCLI()
        return cli.main(args)
    elif command == "reports":
        cli = ReportsCLI()
        return cli.main(args)
    else:
        print(f"Unknown command: {command}")
        print("Available commands: sweep, reports")
        return 1


if __name__ == "__main__":
    sys.exit(main())
