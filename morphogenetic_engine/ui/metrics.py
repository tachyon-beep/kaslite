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

from .config import DEFAULT_SPARKLINE_CONFIG, METRIC_TYPES, STYLE_BOLD_BLUE


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

    def create_metrics_table_panel(self, experiment_params: dict[str, Any]) -> Panel:
        """Generate the panel for current training metrics."""
        metrics_table = Table(show_header=False, expand=True, box=box.MINIMAL)
        metrics_table.add_column("Metric", style=STYLE_BOLD_BLUE, ratio=1)
        metrics_table.add_column("Value", ratio=1)

        # Calculate performance metrics
        avg_epoch_time = self.latest_metrics.get("avg_epoch_time")
        if avg_epoch_time:
            samples_per_epoch = experiment_params.get("n_samples", 1000)
            samples_per_sec = samples_per_epoch / avg_epoch_time
            self.latest_metrics["samples_per_sec"] = samples_per_sec

        # Core metrics with formatting
        metrics_display = [
            ("Train Loss", "train_loss", "{:.6f}"),
            ("Val Loss", "val_loss", "{:.6f}"),
            ("Val Accuracy", "val_acc", "{:.4f}"),
            ("Learning Rate", "lr", "{:.6f}"),
            ("Samples/Sec", "samples_per_sec", "{:.1f}"),
            ("Avg Epoch Time", "avg_epoch_time", "{:.2f}s"),
        ]

        for display_name, key, fmt in metrics_display:
            value = self.latest_metrics.get(key)
            if value is not None:
                formatted_value = fmt.format(value)
            else:
                formatted_value = "N/A"
            metrics_table.add_row(display_name, formatted_value)

        return Panel(metrics_table, title="Current Metrics", border_style="cyan")

    def create_sparkline_panel(self) -> Panel:
        """Generate the panel containing sparklines for key metrics."""
        sparkline_table = Table(show_header=False, expand=True, box=box.MINIMAL)
        sparkline_table.add_column("Metric", style=STYLE_BOLD_BLUE, ratio=1)
        sparkline_table.add_column("Trend", ratio=3)

        # Key metrics to show sparklines for
        sparkline_metrics = [
            ("Train Loss", "train_loss"),
            ("Val Loss", "val_loss"),
            ("Val Accuracy", "val_acc"),
            ("Loss Change", "loss_change"),
            ("Learning Rate", "learning_rate"),
        ]

        for display_name, metric_key in sparkline_metrics:
            sparkline = self._generate_sparkline(metric_key)
            sparkline_table.add_row(display_name, sparkline)

        return Panel(sparkline_table, title="Metric Trends", border_style="magenta")

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
