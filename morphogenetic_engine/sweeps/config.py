"""
Configuration parsing and validation for sweep experiments.

This module handles loading and validating YAML sweep configurations,
supporting both grid search and Bayesian optimization parameters.
"""

import itertools
from pathlib import Path
from typing import Any, Dict, List, Union, cast

import yaml


class SweepConfig:
    """Configuration for a parameter sweep experiment."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize from a configuration dictionary."""
        self.raw_config = config_dict
        self.sweep_type = config_dict.get('sweep_type', 'grid')
        self.experiment = config_dict.get('experiment', {})
        self.parameters = config_dict.get('parameters', {})
        self.execution = config_dict.get('execution', {})
        self.optimization = config_dict.get('optimization', {})
        
        self._validate()
    
    def _validate(self):
        """Validate the configuration."""
        if self.sweep_type not in ['grid', 'bayesian']:
            raise ValueError(f"Invalid sweep_type: {self.sweep_type}. Must be 'grid' or 'bayesian'")
        
        if not self.parameters:
            raise ValueError("Parameters section cannot be empty")
    
    def get_grid_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for grid search."""
        if self.sweep_type != 'grid':
            raise ValueError("Grid combinations only available for grid sweep type")
        
        # Parse all values into lists
        param_lists = {}
        for key, value in self.parameters.items():
            param_lists[key] = parse_value_list(value)
        
        # Merge with experiment fixed parameters
        for key, value in self.experiment.items():
            if key not in param_lists:
                param_lists[key] = [value]
        
        # Create cartesian product
        keys = list(param_lists.keys())
        value_combinations = itertools.product(*param_lists.values())
        
        # Convert to list of dictionaries
        combinations = []
        for combination in value_combinations:
            combo_dict = dict(zip(keys, combination))
            combinations.append(combo_dict)
        
        return combinations
    
    def get_bayesian_search_space(self) -> Dict[str, Any]:
        """Get the parameter search space for Bayesian optimization."""
        if self.sweep_type != 'bayesian':
            raise ValueError("Bayesian search space only available for bayesian sweep type")
        
        # Return parameters for Optuna integration
        # The BayesianSearchRunner will process these into proper Optuna distributions
        return dict(self.parameters)
    
    @property
    def max_parallel(self) -> int:
        """Maximum number of parallel experiments."""
        return int(self.execution.get('max_parallel', 1))
    
    @property 
    def timeout_per_trial(self) -> int:
        """Timeout per trial in seconds."""
        return int(self.execution.get('timeout_per_trial', 3600))
    
    @property
    def target_metric(self) -> str:
        """Target metric for optimization."""
        return cast(str, self.optimization.get('target_metric', 'val_acc'))
    
    @property
    def direction(self) -> str:
        """Optimization direction (maximize or minimize)."""
        return cast(str, self.optimization.get('direction', 'maximize'))


def parse_value_list(value: Any) -> List[Any]:
    """Parse a parameter value into a list of possible values."""
    if isinstance(value, list):
        return value
    elif isinstance(value, str):
        # Handle comma-separated values
        if ',' in value:
            return [item.strip() for item in value.split(',')]
        else:
            return [value]
    else:
        return [value]


def load_sweep_config(config_path: Union[str, Path]) -> SweepConfig:
    """Load a sweep configuration from a YAML file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Sweep config file not found: {config_path}")
    
    if config_path.suffix.lower() not in ['.yml', '.yaml']:
        raise ValueError(f"Sweep config file must have .yml or .yaml extension: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    return SweepConfig(config_dict)


def load_sweep_configs(config_path: Union[str, Path]) -> List[SweepConfig]:
    """Load sweep configuration(s) from a file or directory."""
    config_path = Path(config_path)
    configs = []
    
    if config_path.is_file():
        configs.append(load_sweep_config(config_path))
    elif config_path.is_dir():
        yaml_files = list(config_path.glob("*.yml")) + list(config_path.glob("*.yaml"))
        if not yaml_files:
            raise ValueError(f"No YAML files found in directory: {config_path}")
        for yaml_file in sorted(yaml_files):
            configs.append(load_sweep_config(yaml_file))
    else:
        raise ValueError(f"Sweep config path does not exist: {config_path}")
    
    return configs
