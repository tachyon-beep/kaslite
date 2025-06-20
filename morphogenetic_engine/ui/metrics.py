"""
Metrics handling components for the UI dashboard.

This module handles metric history tracking, sparkline generation,
and metric-related calculations and visualizations.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from rich import box
from rich.panel import Panel
from rich.table import Table

from .config import DEFAULT_SPARKLINE_CONFIG, METRIC_TYPES


class MetricsManager:
    """Manages metric history, sparklines, and derived metric calculations."""

    def __init__(self):
        self.metrics_history: list[dict[str, Any]] = []
        self.metric_histories: dict[str, deque[float]] = {}
        self.latest_metrics: dict[str, Any] = {}
        self.previous_metrics: dict[str, Any] = {}

        # Timing tracking
        self.epoch_times: list[float] = []
        self.experiment_start_time: float = 0.0
        self.epoch_start_time: float | None = None
        self.total_planned_epochs: int | None = None

        # Configuration
        self.sparkline_config = DEFAULT_SPARKLINE_CONFIG.copy()
        self.metric_types = METRIC_TYPES.copy()

    def update_metrics(self, metrics: dict[str, Any], epoch: int, timestamp: float) -> None:
        """Update metrics and calculate derived values."""
        self.previous_metrics = self.latest_metrics.copy()
        self.latest_metrics = metrics.copy()

        # Add derived metrics
        self._add_derived_metrics(epoch, timestamp)

        # Update metric histories for sparklines
        for metric_key, value in self.latest_metrics.items():
            if isinstance(value, (int, float)):
                self._update_metric_history(metric_key, value)

        # Store in history for trend analysis
        metrics_with_metadata = {"epoch": epoch, "timestamp": timestamp, **metrics}
        self.metrics_history.append(metrics_with_metadata)
        # Keep only recent history
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]

    def _add_derived_metrics(self, epoch: int, timestamp: float) -> None:
        """Calculate and add derived metrics to latest_metrics."""
        # Timing metrics
        if self.epoch_times:
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
            self.latest_metrics["avg_epoch_time"] = avg_epoch_time

            # Estimate remaining time
            if self.total_planned_epochs and epoch > 0:
                remaining_epochs = self.total_planned_epochs - epoch
                eta_seconds = remaining_epochs * avg_epoch_time
                self.latest_metrics["eta_seconds"] = eta_seconds

        # Training progress metrics
        total_time = timestamp - self.experiment_start_time
        self.latest_metrics["total_time"] = total_time
        self.latest_metrics["epoch_num"] = epoch

        # Loss and accuracy derivatives if we have previous metrics
        if self.previous_metrics:
            # Loss change
            prev_loss = self.previous_metrics.get("train_loss")
            curr_loss = self.latest_metrics.get("train_loss")
            if prev_loss is not None and curr_loss is not None:
                loss_change = curr_loss - prev_loss
                self.latest_metrics["loss_change"] = loss_change
                self.latest_metrics["loss_improvement"] = -loss_change  # Negative change is improvement

            # Accuracy change
            prev_acc = self.previous_metrics.get("val_acc")
            curr_acc = self.latest_metrics.get("val_acc")
            if prev_acc is not None and curr_acc is not None:
                acc_change = curr_acc - prev_acc
                self.latest_metrics["acc_change"] = acc_change

    def _update_metric_history(self, metric_key: str, value: float) -> None:
        """Add new value to metric history with rolling window."""
        window_size = self.sparkline_config["window_size"]
        if not isinstance(window_size, int):
            window_size = 45  # Default fallback

        if metric_key not in self.metric_histories:
            self.metric_histories[metric_key] = deque(maxlen=window_size)

        if isinstance(value, (int, float)) and not math.isnan(value):
            self.metric_histories[metric_key].append(value)

    def create_metrics_table_panel(self) -> Panel:
        """Generate the panel for the live metrics table with comprehensive training data."""
        metrics_table = Table(show_header=True, header_style="bold magenta", box=box.MINIMAL, padding=(0, 0))
        metrics_table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
        metrics_table.add_column("Curr", justify="center", style="green")
        metrics_table.add_column("Prev", justify="center", style="yellow")
        metrics_table.add_column("Chg", justify="center", style="red")

        # Define metrics to display with their formatting and priority
        metrics_config = self._get_metrics_display_config()

        for metric_key, display_name, fmt in metrics_config:
            current_str, previous_str, change_str = self._format_metric_row(metric_key, fmt)
            metrics_table.add_row(display_name, current_str, previous_str, change_str)

        # Center the table both horizontally and vertically
        from rich.align import Align

        centered_table = Align.center(metrics_table, vertical="middle")
        return Panel(centered_table, title="Live Training Metrics", border_style="cyan")

    def _get_metrics_display_config(self) -> list[tuple[str, str, str]]:
        """Get the configuration for metrics display."""
        return [
            # Core training metrics
            ("train_loss", "Train Loss", ".4f"),
            ("val_loss", "Val Loss", ".4f"),
            ("val_acc", "Val Accuracy", ".4f"),
            ("best_acc", "Best Accuracy", ".4f"),
            # Training dynamics
            ("loss_change", "Loss Δ", "+.4f"),
            ("acc_change", "Acc Δ", "+.4f"),
            ("loss_improvement", "Loss ↗", "+.4f"),
            # Timing and performance
            ("avg_epoch_time", "Epoch Time (s)", ".2f"),
            ("samples_per_sec", "Samples/sec", ".1f"),
            ("total_time", "Total Time", ".0f"),
            ("eta_seconds", "ETA", ".0f"),
        ]

    def _format_metric_row(self, metric_key: str, fmt: str) -> tuple[str, str, str]:
        """Format a single metric row for display."""
        current_val = self.latest_metrics.get(metric_key)
        previous_val = self.previous_metrics.get(metric_key)

        # Format current value
        current_str = self._format_metric_value(current_val, fmt, metric_key)

        # For time metrics, don't show previous value or change
        if metric_key in ["total_time", "eta_seconds"]:
            previous_str = "-"
            change_str = "-"
        else:
            # Format previous value
            previous_str = self._format_metric_value(previous_val, fmt, metric_key)
            # Calculate and format change
            change_str = self._calculate_metric_change(current_val, previous_val, fmt, metric_key)

        return current_str, previous_str, change_str

    def _format_metric_value(self, value: Any, fmt: str, metric_key: str = "") -> str:
        """Format a single metric value."""
        if value is not None:
            # Special formatting for time metrics - mm:ss until 59:59, then hh:mm
            if metric_key in ["total_time", "eta_seconds"] and isinstance(value, (int, float)):
                if value < 3600:  # Less than 1 hour, show mm:ss
                    minutes = int(value // 60)
                    seconds = int(value % 60)
                    return f"{minutes:02d}:{seconds:02d}"
                else:  # 1 hour or more, show hh:mm
                    hours = int(value // 3600)
                    minutes = int((value % 3600) // 60)
                    return f"{hours:02d}:{minutes:02d}"
            elif isinstance(value, float):
                return f"{value:{fmt}}"
            return str(value)
        return "N/A"

    def _calculate_metric_change(self, current_val: Any, previous_val: Any, fmt: str, metric_key: str = "") -> str:
        """Calculate and format the change between two metric values."""
        if not (
            current_val is not None
            and previous_val is not None
            and isinstance(current_val, (int, float))
            and isinstance(previous_val, (int, float))
        ):
            return "N/A"

        change = current_val - previous_val
        if abs(change) <= 1e-6:  # No meaningful change
            return self._format_zero_change(fmt, metric_key)

        # Format meaningful changes
        return self._format_meaningful_change(change, fmt, metric_key)

    def _format_zero_change(self, fmt: str, metric_key: str) -> str:
        """Format zero change with appropriate precision."""
        if ".3f" in fmt:
            return "0.000"
        elif ".2f" in fmt:
            return "0.00"
        elif ".1f" in fmt:
            return "0.0"
        elif ".0f" in fmt:
            return "0"
        elif ".2e" in fmt or metric_key == "learning_rate":
            return "0.00e+00"
        else:
            return "0.000"

    def _format_meaningful_change(self, change: float, fmt: str, metric_key: str) -> str:
        """Format meaningful change values."""
        # Special handling for learning rate - show in exponential format
        if metric_key == "learning_rate":
            return f"{change:+.2e}"
        # Use the same format as the metric, but with + sign for positive changes
        elif fmt.startswith("+"):
            return f"{change:{fmt}}"
        else:
            return f"{change:+{fmt}}"

    def create_sparkline_panel(self) -> Panel:
        """Generate the panel for metric trends and sparklines."""
        sparkline_table = Table(show_header=True, header_style="bold cyan", expand=True, box=box.MINIMAL)
        sparkline_table.add_column("Metric", style="cyan", no_wrap=True)
        sparkline_table.add_column("Current", justify="center", style="green")
        sparkline_table.add_column("Change", justify="center", style="yellow")
        sparkline_table.add_column("Trend", style="dim", ratio=4)

        # Define metrics useful for tracking trends (updated per user request)
        trend_metrics = [
            ("train_loss", "Train Loss", ".3f"),
            ("val_loss", "Val Loss", ".3f"),
            ("val_acc", "Val Accuracy", ".3f"),
            ("loss_change", "Loss Δ", "+.3f"),
            ("acc_change", "Acc Δ", "+.3f"),
            ("loss_improvement", "Loss ↗", "+.3f"),
            ("samples_per_sec", "Samples/sec", ".1f"),
        ]

        for metric_key, display_name, fmt in trend_metrics:
            current_val = self.latest_metrics.get(metric_key)
            previous_val = self.previous_metrics.get(metric_key)

            # Format current value
            if current_val is not None:
                if isinstance(current_val, float):
                    current_str = f"{current_val:{fmt}}"
                else:
                    current_str = str(current_val)
            else:
                current_str = "N/A"

            # Calculate change
            change_str = self._calculate_metric_change(current_val, previous_val, fmt, metric_key)

            # Generate dynamic sparkline
            trend_indicator = self._generate_sparkline(metric_key)

            sparkline_table.add_row(display_name, current_str, change_str, trend_indicator)

        return Panel(sparkline_table, title="Trends", border_style="cyan")

    def _normalize_values(self, values: list[float], inverted: bool = False) -> list[int]:
        """Normalize values to 0-6 range for ASCII levels."""
        if len(values) < 2:
            return [3] * len(values)  # Middle level for insufficient data

        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            return [3] * len(values)  # All same value = middle level

        normalized = []
        for val in values:
            # Normalize to 0-1 range
            norm = (val - min_val) / (max_val - min_val)
            # Invert if needed (for loss metrics)
            if inverted:
                norm = 1 - norm
            # Convert to 0-6 index (7 levels total)
            level = int(norm * 6)
            normalized.append(min(6, max(0, level)))

        return normalized

    def _smooth_values(self, values: list[float], window: int = 3) -> list[float]:
        """Apply simple moving average smoothing."""
        if len(values) < window:
            return values

        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window // 2)
            end = min(len(values), i + window // 2 + 1)
            avg = sum(values[start:end]) / (end - start)
            smoothed.append(avg)

        return smoothed

    def _calculate_trend(self, recent_values: list[float]) -> float:
        """Calculate trend direction from recent values (-1 to 1)."""
        if len(recent_values) < 3:
            return 0.0

        # Simple linear trend calculation
        x_vals = list(range(len(recent_values)))
        y_vals = recent_values

        n = len(recent_values)
        sum_x = sum(x_vals)
        sum_y = sum(y_vals)
        sum_xy = sum(x * y for x, y in zip(x_vals, y_vals))
        sum_x2 = sum(x * x for x in x_vals)

        if n * sum_x2 - sum_x * sum_x == 0:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

        # Normalize slope to -1 to 1 range based on value scale
        value_range = max(y_vals) - min(y_vals)
        if value_range > 0:
            normalized_slope = slope / value_range
            return max(-1.0, min(1.0, normalized_slope))

        return 0.0

    def _generate_sparkline(self, metric_key: str) -> str:
        """Generate a sparkline for the given metric."""
        display_length = self.sparkline_config.get("display_length", 45)
        if not isinstance(display_length, int):
            display_length = 45

        if metric_key not in self.metric_histories:
            return "[dim]" + "─" * display_length + "[/dim]"

        history = list(self.metric_histories[metric_key])
        if len(history) < 2:
            return "[dim]" + "─" * display_length + "[/dim]"

        # Process the data
        processed_history = self._process_sparkline_data(history)
        sparkline_chars = self._create_sparkline_chars(processed_history)

        # Apply coloring
        return self._apply_sparkline_coloring(sparkline_chars, metric_key, history)

    def _process_sparkline_data(self, history: list[float]) -> list[int]:
        """Process raw history data into sparkline indices."""
        # Apply smoothing if enabled
        processed_history = history
        if self.sparkline_config["smoothing"]:
            processed_history = self._smooth_values(history)

        # Get metric configuration and normalize - fix: use empty string as fallback
        normalized = self._normalize_values(processed_history, False)  # Default to not inverted

        # Subsample or pad to display length
        display_length = self.sparkline_config.get("display_length", 45)
        if not isinstance(display_length, int):
            display_length = 45

        if len(normalized) > display_length:
            step = len(normalized) / display_length
            indices = [int(i * step) for i in range(display_length)]
            return [normalized[i] for i in indices]
        else:
            return normalized + [normalized[-1]] * (display_length - len(normalized))

    def _create_sparkline_chars(self, sparkline_data: list[int]) -> str:
        """Convert sparkline data indices to characters."""
        levels = self.sparkline_config.get("levels", "▁▂▃▄▅▆▇")
        if not isinstance(levels, str):
            levels = "▁▂▃▄▅▆▇"
        return "".join(levels[level] for level in sparkline_data)

    def _apply_sparkline_coloring(self, sparkline_chars: str, metric_key: str, history: list[float]) -> str:
        """Apply trend-based coloring to sparkline characters."""
        metric_config = self.metric_types.get(metric_key, {"inverted": False, "color": "white"})

        if not self.sparkline_config.get("color_trends", True) or len(history) < 5:
            color = metric_config.get("color", "white")
            return f"[{color}]{sparkline_chars}[/{color}]"

        recent_trend = self._calculate_trend(history[-5:])
        inverted = metric_config.get("inverted", False)
        if not isinstance(inverted, bool):
            inverted = False
        trend_color = self._get_trend_color(recent_trend, inverted)

        return f"[{trend_color}]{sparkline_chars}[/{trend_color}]"

    def _get_trend_color(self, trend: float, inverted: bool) -> str:
        """Determine color based on trend direction and metric type."""
        if trend > 0.1:  # Improving
            return "green" if not inverted else "red"
        elif trend < -0.1:  # Degrading
            return "red" if not inverted else "green"
        else:  # Stable
            return "dim"

    def configure_sparklines(self, **kwargs) -> None:
        """Allow runtime configuration of sparkline behavior."""
        self.sparkline_config.update(kwargs)

    def add_metric_type(self, metric_key: str, inverted: bool = False, color: str = "white") -> None:
        """Register a new metric type for sparkline rendering."""
        self.metric_types[metric_key] = {"inverted": inverted, "color": color}
