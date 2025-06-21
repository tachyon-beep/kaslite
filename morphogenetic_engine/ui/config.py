"""
Configuration and constants for the UI dashboard.

This module contains all constants, emoji mappings, styling configurations,
and other UI-related settings used throughout the dashboard components.
"""

from __future__ import annotations

from morphogenetic_engine.events import NetworkStrain, SeedState

# Grid Configuration
GRID_SIZE = 16

# Emoji Maps
SEED_EMOJI_MAP = {
    SeedState.DORMANT: "[grey50]⚪[/]",          # ⚪ White circle (dormant)
    SeedState.GERMINATED: "[bold green]🌱[/]",   # 🌱 Seedling (germinated)
    SeedState.TRAINING: "[bold yellow]🟢[/]",    # 🟢 Green circle (training)
    SeedState.BLENDING: "[bold cyan]🟡[/]",      # 🟡 Yellow circle (blending)
    SeedState.SHADOWING: "[bold bright_black]👻[/]",  # 👻 Ghost (shadowing)
    SeedState.PROBATIONARY: "[bold magenta]🧑‍⚖️[/]",  # 🧑‍⚖️ Judge (probationary)
    SeedState.FOSSILIZED: "[bold white]🦴[/]",   # 🦴 Bone (fossilized)
    SeedState.CULLED: "[bold red]🥀[/]",         # 🥀 Wilted flower (culled)
}

STRAIN_EMOJI_MAP = {
    NetworkStrain.NONE: "🔵",
    NetworkStrain.LOW: "🟢",
    NetworkStrain.MEDIUM: "🟡",
    NetworkStrain.HIGH: "🔴",
    NetworkStrain.FIRED: "💥",
}

# Common UI Elements
EMPTY_CELL_EMOJI = "⚫"
STYLE_BOLD_BLUE = "bold blue"

# Sparkline Configuration
DEFAULT_SPARKLINE_CONFIG = {
    "window_size": 45,  # Number of data points to track
    "display_length": 45,  # Characters to display
    "levels": "▁▂▃▄▅▆▇",
    "smoothing": True,
    "color_trends": True,
}

# Metric Type Configuration for Sparklines
METRIC_TYPES = {
    "train_loss": {"inverted": True, "color": "green"},  # Lower is better
    "val_loss": {"inverted": True, "color": "green"},
    "val_acc": {"inverted": False, "color": "cyan"},  # Higher is better
    "learning_rate": {"inverted": False, "color": "yellow"},
    "loss_change": {"inverted": True, "color": "magenta"},
    "acc_change": {"inverted": False, "color": "magenta"},
    "loss_improvement": {"inverted": False, "color": "green"},
}

# Event Log Configuration
MAX_EVENTS = 20
MAX_SEED_EVENTS = 200  # Increased to capture more seed events

# Layout Sizing Configuration
LAYOUT_SIZES = {
    "title_header": 3,
    "footer": 3,
    "quit_footer": 1,
    "top_left_area": 32,
    "sparkline_panel": 12,
    "top_row": 20,
    "tamiyo_panel": 13,
    "karn_panel": 9,
    "crucible_panel": 8,
    "legend_panel": 3,
    "seed_legend_panel": 4,
    "strain_legend_panel": 3,
}

# Column Ratios
COLUMN_RATIOS = {
    "left_column": 7,
    "center_column": 5,
    "right_column": 7,
    "info_panel": 2,
    "metrics_table_panel": 3,
}
