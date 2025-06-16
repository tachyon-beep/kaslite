#!/usr/bin/env python3
"""
Demo script for Phase 3: Hyperparameter Sweeps & Automated Optimization.

This script demonstrates the new sweep capabilities including:
- Enhanced grid search with parallel execution
- Rich-powered reporting and analysis
- CLI interfaces for sweep management
"""

import subprocess
import sys
from pathlib import Path
from rich.console import Console


def run_command(cmd, description, capture=False):
    """Run a command and optionally capture output."""
    console = Console()
    
    console.print(f"\n{'='*60}")
    console.print(f"STEP: {description}")
    console.print(f"COMMAND: {cmd}")
    console.print(f"{'='*60}")

    result = subprocess.run(cmd, shell=True, capture_output=capture, text=True, check=False)

    if not capture:
        if result.returncode != 0:
            console.print(f"⚠️  Command failed with return code {result.returncode}")
            return False, None
        else:
            console.print("✅ Command completed successfully")
            return True, None
    else:
        return result.returncode == 0, result.stdout


def main():
    """Demonstrate Phase 3 functionality."""
    console = Console()
    console.print("🚀 Phase 3: Hyperparameter Sweeps & Automated Optimization Demo")
    console.print("This demo showcases the new sweep capabilities and reporting tools")

    # Check if we're in the right directory
    if not Path("examples/quick_sweep.yaml").exists():
        console.print("❌ Error: quick_sweep.yaml not found. Please run this script from the project root.")
        sys.exit(1)

    # 1. Test CLI help commands
    if not run_command("python -m morphogenetic_engine.cli.sweep --help", "Test sweep CLI help")[0]:
        return

    if not run_command("python -m morphogenetic_engine.cli.reports --help", "Test reports CLI help")[0]:
        return

    # 2. Generate test data if needed
    if not Path("data/spirals.npz").exists():
        if not run_command(
            "python scripts/generate_data.py --output data/spirals.npz --problem spirals --samples 1000",
            "Generate test data for spirals problem"
        )[0]:
            return

    # 3. Run a quick grid search sweep
    console.print("\n[bold cyan]Running Quick Grid Search Sweep[/bold cyan]")
    if not run_command(
        "python -m morphogenetic_engine.cli.sweep grid --config examples/quick_sweep.yaml --parallel 2",
        "Execute grid search with parallel execution"
    )[0]:
        console.print("⚠️  Grid search failed, but continuing with demo...")

    # 4. Test the quick sweep command
    console.print("\n[bold cyan]Testing Quick Sweep Command[/bold cyan]")
    if not run_command(
        "python -m morphogenetic_engine.cli.sweep quick --problem spirals --trials 4",
        "Run quick validation sweep"
    )[0]:
        console.print("⚠️  Quick sweep failed, but continuing with demo...")

    # 5. Find and analyze results
    console.print("\n[bold cyan]Analyzing Sweep Results[/bold cyan]")
    
    # Find the most recent sweep directory
    results_dir = Path("results/sweeps")
    if results_dir.exists():
        sweep_dirs = list(results_dir.iterdir())
        if sweep_dirs:
            # Get the most recent directory
            latest_sweep = max(sweep_dirs, key=lambda x: x.stat().st_mtime)
            console.print(f"Found sweep results in: {latest_sweep}")
            
            # Generate summary report
            if not run_command(
                f"python -m morphogenetic_engine.cli.reports summary --sweep-dir {latest_sweep}",
                "Generate sweep summary report"
            )[0]:
                console.print("⚠️  Summary report failed")
            
            # Generate detailed analysis
            if not run_command(
                f"python -m morphogenetic_engine.cli.reports analysis --sweep-dir {latest_sweep}",
                "Generate detailed parameter analysis"
            )[0]:
                console.print("⚠️  Analysis report failed")
        else:
            console.print("⚠️  No sweep results found")
    else:
        console.print("⚠️  Results directory not found")

    # 6. Test Bayesian optimization (if Optuna is available)
    console.print("\n[bold cyan]Testing Bayesian Optimization[/bold cyan]")
    success, output = run_command(
        "python -c \"import optuna; print('Optuna available')\"",
        "Check Optuna availability",
        capture=True
    )
    
    if success:
        console.print("✅ Optuna is available, testing Bayesian optimization")
        if not run_command(
            "python -m morphogenetic_engine.cli.sweep bayesian --config examples/bayesian_sweep.yaml --trials 3",
            "Run Bayesian optimization sweep"
        )[0]:
            console.print("⚠️  Bayesian optimization failed")
    else:
        console.print("⚠️  Optuna not available, skipping Bayesian optimization test")
        console.print("Install with: pip install optuna")

    # 7. Show available sweep configurations
    console.print("\n[bold cyan]Available Sweep Configurations[/bold cyan]")
    examples_dir = Path("examples")
    if examples_dir.exists():
        sweep_configs = list(examples_dir.glob("*sweep*.yaml"))
        console.print("Found sweep configurations:")
        for config in sweep_configs:
            console.print(f"  📄 {config.name}")
            
            # Show first few lines of each config
            try:
                with open(config, 'r') as f:
                    lines = f.readlines()[:5]
                    console.print("     " + "".join(lines).rstrip())
                    if len(lines) == 5:
                        console.print("     ...")
                console.print()
            except Exception as e:
                console.print(f"     Error reading file: {e}")

    # 8. Summary
    console.print("\n[bold green]Phase 3 Demo Summary[/bold green]")
    console.print("✅ Enhanced CLI interfaces for sweep management")
    console.print("✅ Grid search with parallel execution support")
    console.print("✅ Rich-powered progress tracking and reporting")
    console.print("✅ Flexible YAML configuration system")
    console.print("✅ Results analysis and parameter importance")
    
    if success:  # Optuna was available
        console.print("✅ Bayesian optimization with Optuna integration")
    else:
        console.print("⚠️  Bayesian optimization requires Optuna (pip install optuna)")
    
    console.print("\n[bold cyan]Next Steps:[/bold cyan]")
    console.print("• Explore the examples/ directory for more sweep configurations")
    console.print("• Run larger sweeps with: morphogenetic-sweep grid --config examples/enhanced_sweep.yaml")
    console.print("• Generate detailed reports with: morphogenetic-reports summary --sweep-dir <path>")
    console.print("• Set up CI/CD with the provided .github/workflows/ci.yml")


if __name__ == "__main__":
    main()
