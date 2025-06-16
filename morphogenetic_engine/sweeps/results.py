"""
Results management and analysis for sweep experiments.

This module handles storing, loading, and analyzing results from
parameter sweep experiments.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from rich.console import Console
from rich.table import Table


class SweepResults:
    """Container for sweep experiment results with analysis capabilities."""
    
    def __init__(self, sweep_dir: Path):
        """Initialize with a sweep directory."""
        self.sweep_dir = Path(sweep_dir)
        self.results: List[Dict[str, Any]] = []
        self.console = Console()
        
        # Ensure directory exists
        self.sweep_dir.mkdir(parents=True, exist_ok=True)
    
    def add_result(self, result: Dict[str, Any]):
        """Add a single experiment result."""
        self.results.append(result)
        
        # Save incrementally to avoid losing data
        self._save_incremental(result)
    
    def _save_incremental(self, result: Dict[str, Any]):
        """Save a single result incrementally."""
        result_file = self.sweep_dir / f"result_{result.get('run_slug', 'unknown')}.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)
    
    def finalize(self):
        """Finalize results by creating summary files."""
        self._create_csv_summary()
        self._create_json_summary()
        self._create_analysis_report()
    
    def _create_csv_summary(self):
        """Create a CSV summary of all results."""
        if not self.results:
            return
        
        csv_path = self.sweep_dir / "sweep_summary.csv"
        
        # Collect all parameter names and result keys
        all_params = set()
        all_metrics = set()
        
        for result in self.results:
            if 'parameters' in result:
                all_params.update(result['parameters'].keys())
            
            # Collect metric keys (exclude metadata)
            for key, value in result.items():
                if key not in ['run_id', 'run_slug', 'parameters', 'stdout', 'stderr', 'error']:
                    if isinstance(value, (int, float, bool)):
                        all_metrics.add(key)
        
        # Write CSV
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['run_id', 'run_slug', 'success'] + sorted(all_params) + sorted(all_metrics)
            writer.writerow(header)
            
            # Data rows
            for result in self.results:
                row = [
                    result.get('run_id', ''),
                    result.get('run_slug', ''),
                    result.get('success', False)
                ]
                
                # Add parameter values
                params = result.get('parameters', {})
                for param in sorted(all_params):
                    row.append(params.get(param, ''))
                
                # Add metric values
                for metric in sorted(all_metrics):
                    row.append(result.get(metric, ''))
                
                writer.writerow(row)
    
    def _create_json_summary(self):
        """Create a complete JSON summary."""
        json_path = self.sweep_dir / "sweep_summary.json"
        
        summary = {
            'sweep_info': {
                'total_experiments': len(self.results),
                'successful_experiments': len([r for r in self.results if r.get('success', False)]),
                'sweep_directory': str(self.sweep_dir),
            },
            'results': self.results
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def _create_analysis_report(self):
        """Create a text analysis report."""
        report_path = self.sweep_dir / "analysis_report.txt"
        
        successful_results = [r for r in self.results if r.get('success', False)]
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Sweep Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total experiments: {len(self.results)}\n")
            f.write(f"Successful experiments: {len(successful_results)}\n")
            f.write(f"Failed experiments: {len(self.results) - len(successful_results)}\n\n")
            
            if successful_results:
                # Find metrics present in results
                metrics = set()
                for result in successful_results:
                    for key, value in result.items():
                        if key not in ['run_id', 'run_slug', 'parameters', 'stdout', 'stderr', 'error', 'success']:
                            if isinstance(value, (int, float)):
                                metrics.add(key)
                
                # Analyze each metric
                for metric in sorted(metrics):
                    values = [r.get(metric) for r in successful_results if metric in r]
                    if values:
                        f.write(f"Metric: {metric}\n")
                        f.write(f"  Mean: {np.mean(values):.4f}\n")
                        f.write(f"  Std:  {np.std(values):.4f}\n")
                        f.write(f"  Min:  {np.min(values):.4f}\n")
                        f.write(f"  Max:  {np.max(values):.4f}\n\n")
    
    def get_successful_results(self) -> List[Dict[str, Any]]:
        """Get only successful experiment results."""
        return [r for r in self.results if r.get('success', False)]
    
    def get_best_result(self, metric: str, direction: str = 'maximize') -> Optional[Dict[str, Any]]:
        """Get the best result for a given metric."""
        successful_results = self.get_successful_results()
        
        if not successful_results:
            return None
        
        # Filter results that have the metric
        metric_results = [r for r in successful_results if metric in r]
        
        if not metric_results:
            return None
        
        if direction == 'maximize':
            return max(metric_results, key=lambda x: x[metric])
        else:
            return min(metric_results, key=lambda x: x[metric])
    
    def get_parameter_analysis(self) -> Dict[str, Any]:
        """Analyze parameter importance and correlations."""
        successful_results = self.get_successful_results()
        
        if len(successful_results) < 2:
            return {}
        
        # Extract parameters and metrics
        all_params = set()
        all_metrics = set()
        
        for result in successful_results:
            if 'parameters' in result:
                all_params.update(result['parameters'].keys())
            
            for key, value in result.items():
                if key not in ['run_id', 'run_slug', 'parameters', 'stdout', 'stderr', 'error', 'success']:
                    if isinstance(value, (int, float)):
                        all_metrics.add(key)
        
        analysis = {
            'parameter_names': sorted(all_params),
            'metric_names': sorted(all_metrics),
            'parameter_correlations': {},
        }
        
        # Calculate correlations for numeric parameters
        for param in all_params:
            param_values = []
            for result in successful_results:
                param_value = result.get('parameters', {}).get(param)
                if isinstance(param_value, (int, float)):
                    param_values.append(param_value)
            
            if len(param_values) > 1:
                param_correlations = {}
                for metric in all_metrics:
                    metric_values = []
                    param_subset = []
                    
                    for result in successful_results:
                        param_value = result.get('parameters', {}).get(param)
                        metric_value = result.get(metric)
                        
                        if (isinstance(param_value, (int, float)) and 
                            isinstance(metric_value, (int, float))):
                            param_subset.append(param_value)
                            metric_values.append(metric_value)
                    
                    if len(metric_values) > 1:
                        correlation = np.corrcoef(param_subset, metric_values)[0, 1]
                        if not np.isnan(correlation):
                            param_correlations[metric] = correlation
                
                if param_correlations:
                    analysis['parameter_correlations'][param] = param_correlations
        
        return analysis


class ResultsAnalyzer:
    """Advanced analysis tools for sweep results."""
    
    def __init__(self, results: SweepResults):
        """Initialize with sweep results."""
        self.results = results
        self.console = Console()
    
    def print_summary_table(self, metric: str = 'val_acc'):
        """Print a rich table summarizing the results."""
        successful_results = self.results.get_successful_results()
        
        if not successful_results:
            self.console.print("[bold red]No successful results to display[/bold red]")
            return
        
        # Sort by metric (descending for maximize metrics)
        metric_results = [r for r in successful_results if metric in r]
        metric_results.sort(key=lambda x: x[metric], reverse=True)
        
        # Take top 10
        top_results = metric_results[:10]
        
        # Create table
        table = Table(title=f"Top 10 Results by {metric}")
        table.add_column("Rank", style="cyan")
        table.add_column("Run", style="magenta")
        table.add_column(metric, style="green")
        
        # Add parameter columns (limit to most important ones)
        param_names = set()
        for result in top_results:
            if 'parameters' in result:
                param_names.update(result['parameters'].keys())
        
        # Limit to 5 most varying parameters
        important_params = list(param_names)[:5]
        for param in important_params:
            table.add_column(param, style="yellow")
        
        # Add rows
        for i, result in enumerate(top_results):
            row = [
                str(i + 1),
                result.get('run_slug', 'unknown'),
                f"{result[metric]:.4f}"
            ]
            
            # Add parameter values
            params = result.get('parameters', {})
            for param in important_params:
                row.append(str(params.get(param, 'N/A')))
            
            table.add_row(*row)
        
        self.console.print(table)
    
    def print_parameter_importance(self, metric: str = 'val_acc'):
        """Print parameter importance analysis."""
        analysis = self.results.get_parameter_analysis()
        correlations = analysis.get('parameter_correlations', {})
        
        if not correlations:
            self.console.print("[yellow]Not enough data for parameter importance analysis[/yellow]")
            return
        
        # Create importance table
        table = Table(title=f"Parameter Correlations with {metric}")
        table.add_column("Parameter", style="cyan")
        table.add_column("Correlation", style="magenta")
        table.add_column("Strength", style="green")
        
        # Collect correlations for the target metric
        param_correlations = []
        for param, metrics in correlations.items():
            if metric in metrics:
                param_correlations.append((param, metrics[metric]))
        
        # Sort by absolute correlation
        param_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for param, corr in param_correlations:
            strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.3 else "Weak"
            table.add_row(param, f"{corr:.3f}", strength)
        
        self.console.print(table)
