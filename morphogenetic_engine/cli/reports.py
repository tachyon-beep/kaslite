"""
CLI for generating reports and analyzing sweep results.

This module provides Rich-powered dashboards and analysis tools
for experiment results.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

from rich.console import Console

from ..sweeps.results import ResultsAnalyzer, SweepResults

PATH_TO_SWEEP_RESULTS_DIR_HELP = "Path to sweep results directory"


class ReportsCLI:
    """Command-line interface for generating reports and analysis."""

    def __init__(self):
        """Initialize the reports CLI."""
        self.console = Console()

    def create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser for report commands."""
        parser = argparse.ArgumentParser(
            prog="morphogenetic-reports",
            description="Generate reports and analysis for morphogenetic experiments",
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Summary command
        summary_parser = subparsers.add_parser("summary", help="Show experiment summary")
        summary_parser.add_argument("--sweep-dir", "-d", type=Path, required=True, help=PATH_TO_SWEEP_RESULTS_DIR_HELP)
        summary_parser.add_argument(
            "--metric",
            "-m",
            type=str,
            default="val_acc",
            help="Metric to focus on (default: val_acc)",
        )
        summary_parser.add_argument("--top", "-t", type=int, default=10, help="Number of top results to show (default: 10)")

        # Analysis command
        analysis_parser = subparsers.add_parser("analysis", help="Detailed parameter analysis")
        analysis_parser.add_argument("--sweep-dir", "-d", type=Path, required=True, help=PATH_TO_SWEEP_RESULTS_DIR_HELP)
        analysis_parser.add_argument(
            "--metric",
            "-m",
            type=str,
            default="val_acc",
            help="Metric for analysis (default: val_acc)",
        )

        # Compare command
        compare_parser = subparsers.add_parser("compare", help="Compare multiple sweeps")
        compare_parser.add_argument(
            "--sweep-dirs",
            type=Path,
            nargs="+",
            required=True,
            help="Paths to sweep directories to compare",
        )
        compare_parser.add_argument(
            "--metric",
            "-m",
            type=str,
            default="val_acc",
            help="Metric for comparison (default: val_acc)",
        )

        # Export command
        export_parser = subparsers.add_parser("export", help="Export results to different formats")
        export_parser.add_argument("--sweep-dir", "-d", type=Path, required=True, help=PATH_TO_SWEEP_RESULTS_DIR_HELP)
        export_parser.add_argument(
            "--format",
            "-f",
            choices=["html", "csv", "json"],
            default="html",
            help="Export format (default: html)",
        )
        export_parser.add_argument("--output", "-o", type=Path, help="Output file path (default: auto-generated)")

        return parser

    def show_summary(self, args: argparse.Namespace) -> int:
        """Show a summary of sweep results."""
        try:
            if not args.sweep_dir.exists():
                self.console.print(f"[bold red]Sweep directory not found: {args.sweep_dir}[/bold red]")
                return 1

            # Load results
            results = SweepResults(args.sweep_dir)

            # Try to load existing results
            json_summary = args.sweep_dir / "sweep_summary.json"
            if json_summary.exists():
                with open(json_summary, "r", encoding="utf-8") as f:
                    summary_data = json.load(f)
                    results.results = summary_data.get("results", [])
            else:
                self.console.print("[yellow]No summary file found, looking for individual result files...[/yellow]")
                # Try to load individual result files
                result_files = list(args.sweep_dir.glob("result_*.json"))
                if result_files:
                    for result_file in result_files:
                        with open(result_file, "r", encoding="utf-8") as f:
                            result = json.load(f)
                            results.results.append(result)
                else:
                    self.console.print("[bold red]No result files found in directory[/bold red]")
                    return 1

            # Show analysis
            analyzer = ResultsAnalyzer(results)

            self.console.print("[bold green]Sweep Results Summary[/bold green]")
            self.console.print(f"Directory: {args.sweep_dir}")
            self.console.print(f"Total experiments: {len(results.results)}")
            self.console.print(f"Successful experiments: {len(results.get_successful_results())}")

            analyzer.print_summary_table(args.metric)
            analyzer.print_parameter_importance(args.metric)

            return 0

        except (ValueError, RuntimeError, ImportError, FileNotFoundError) as e:
            self.console.print(f"[bold red]Error loading results: {e}[/bold red]")
            return 1

    def show_analysis(self, args: argparse.Namespace) -> int:
        """Show detailed parameter analysis."""
        try:
            if not args.sweep_dir.exists():
                self.console.print(f"[bold red]Sweep directory not found: {args.sweep_dir}[/bold red]")
                return 1

            results = SweepResults(args.sweep_dir)

            # Load results (same logic as summary)
            json_summary = args.sweep_dir / "sweep_summary.json"
            if json_summary.exists():
                with open(json_summary, "r", encoding="utf-8") as f:
                    summary_data = json.load(f)
                    results.results = summary_data.get("results", [])

            analyzer = ResultsAnalyzer(results)

            self.console.print(f"[bold green]Parameter Analysis for {args.metric}[/bold green]")
            analyzer.print_parameter_importance(args.metric)

            # Show parameter analysis
            analysis = results.get_parameter_analysis()
            if analysis.get("parameter_correlations"):
                self.console.print("\n[bold cyan]Detailed Correlations:[/bold cyan]")
                for param, correlations in analysis["parameter_correlations"].items():
                    self.console.print(f"\n{param}:")
                    for metric, corr in correlations.items():
                        self.console.print(f"  {metric}: {corr:.3f}")

            return 0

        except (ValueError, RuntimeError, ImportError, FileNotFoundError) as e:
            self.console.print(f"[bold red]Error analyzing results: {e}[/bold red]")
            return 1

    def compare_sweeps(self, _args: argparse.Namespace) -> int:
        """Compare multiple sweep results."""
        self.console.print("[yellow]Sweep comparison not yet implemented[/yellow]")
        self.console.print("This feature will be added in a future update")
        return 1

    def export_results(self, _args: argparse.Namespace) -> int:
        """Export results to different formats."""
        self.console.print("[yellow]Export functionality not yet implemented[/yellow]")
        self.console.print("This feature will be added in a future update")
        return 1

    def main(self, argv: Optional[list] = None) -> int:
        """Main entry point for the reports CLI."""
        parser = self.create_parser()
        args = parser.parse_args(argv)

        if args.command is None:
            parser.print_help()
            return 1

        if args.command == "summary":
            return self.show_summary(args)
        elif args.command == "analysis":
            return self.show_analysis(args)
        elif args.command == "compare":
            return self.compare_sweeps(args)
        elif args.command == "export":
            return self.export_results(args)
        else:
            self.console.print(f"[bold red]Unknown command: {args.command}[/bold red]")
            return 1


def main():
    """Main entry point for the reports CLI."""
    cli = ReportsCLI()
    return cli.main()
