# Stage 3B: Reward System and Intensity Control Implementation Guide

## Overview

Stage 3B completes the policy network system by implementing a sophisticated reward calculation framework, hyperparameter tuning system, and intensity control mechanisms. This enables the policy network to learn effectively from experience.

## Comprehensive Reward System

### 1. Multi-Objective Reward Calculator

```python
# morphogenetic_engine/policy/rewards.py
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from ..telemetry.types import SeedTelemetry, HardwareContext

@dataclass
class RewardWeights:
    """Configurable weights for different reward components."""
    
    # Performance rewards
    accuracy_gain: float = 10.0
    loss_improvement: float = 5.0
    convergence_speed: float = 2.0
    
    # Efficiency penalties
    parameter_penalty: float = 0.01
    latency_penalty: float = 0.1
    memory_penalty: float = 0.05
    power_penalty: float = 0.02
    
    # Safety and stability rewards
    drift_bonus: float = 5.0
    stability_bonus: float = 3.0
    security_penalty: float = 20.0
    rollback_penalty: float = 15.0
    
    # Hardware efficiency rewards
    hardware_efficiency_bonus: float = 2.0
    utilization_bonus: float = 1.0
    
    # Exploration rewards
    novelty_bonus: float = 1.0
    diversity_bonus: float = 0.5

@dataclass
class RewardMetrics:
    """Metrics used for reward calculation."""
    
    # Performance metrics
    accuracy_pre: float
    accuracy_post: float
    loss_pre: float
    loss_post: float
    convergence_epochs: int
    
    # Resource metrics
    parameters_added: int
    latency_increase_ms: float
    memory_increase_mb: float
    power_increase_watts: float
    
    # Safety metrics
    drift_level: float
    stability_score: float
    security_violations: int
    rollback_occurred: bool
    
    # Hardware metrics
    hardware_utilization: float
    constraint_violations: int
    
    # Context metrics
    blueprint_novelty: float
    decision_diversity: float
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

class RewardCalculator:
    """Comprehensive reward calculation system for policy learning."""
    
    def __init__(self, weights: Optional[RewardWeights] = None):
        self.weights = weights or RewardWeights()
        self.reward_history = []
        self.component_history = {}
        
    def calculate_reward(self, 
                        metrics: RewardMetrics,
                        hardware_context: HardwareContext,
                        decision_context: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate comprehensive reward signal from metrics.
        
        Args:
            metrics: Performance and resource metrics
            hardware_context: Hardware constraints and capabilities
            decision_context: Context about the decision made
            
        Returns:
            Dictionary with total reward and component breakdowns
        """
        
        components = {}
        
        # Performance rewards
        components['accuracy_gain'] = self._calculate_accuracy_reward(metrics)
        components['loss_improvement'] = self._calculate_loss_reward(metrics)
        components['convergence_speed'] = self._calculate_convergence_reward(metrics)
        
        # Efficiency penalties
        components['parameter_penalty'] = self._calculate_parameter_penalty(metrics)
        components['latency_penalty'] = self._calculate_latency_penalty(metrics, hardware_context)
        components['memory_penalty'] = self._calculate_memory_penalty(metrics, hardware_context)
        components['power_penalty'] = self._calculate_power_penalty(metrics, hardware_context)
        
        # Safety and stability
        components['drift_bonus'] = self._calculate_drift_reward(metrics)
        components['stability_bonus'] = self._calculate_stability_reward(metrics)
        components['security_penalty'] = self._calculate_security_penalty(metrics)
        components['rollback_penalty'] = self._calculate_rollback_penalty(metrics)
        
        # Hardware efficiency
        components['hardware_efficiency'] = self._calculate_hardware_efficiency_reward(
            metrics, hardware_context
        )
        
        # Exploration rewards
        components['novelty_bonus'] = self._calculate_novelty_reward(metrics, decision_context)
        components['diversity_bonus'] = self._calculate_diversity_reward(decision_context)
        
        # Calculate total reward
        total_reward = sum(components.values())
        
        # Store in history
        reward_record = {
            'total_reward': total_reward,
            'components': components.copy(),
            'metrics': metrics,
            'timestamp': torch.tensor(len(self.reward_history), dtype=torch.float32)
        }
        
        self.reward_history.append(reward_record)
        
        # Update component history for analysis
        for component, value in components.items():
            if component not in self.component_history:
                self.component_history[component] = []
            self.component_history[component].append(value)
        
        return {
            'total_reward': total_reward,
            'components': components,
            'normalized_reward': self._normalize_reward(total_reward)
        }
    
    def _calculate_accuracy_reward(self, metrics: RewardMetrics) -> float:
        """Calculate reward based on accuracy improvement."""
        accuracy_gain = metrics.accuracy_post - metrics.accuracy_pre
        
        # Bonus for achieving high accuracy
        if metrics.accuracy_post > 0.95:
            accuracy_gain += 0.1
        
        # Penalty for accuracy degradation
        if accuracy_gain < 0:
            accuracy_gain *= 2.0  # Double penalty for regression
        
        return self.weights.accuracy_gain * accuracy_gain
    
    def _calculate_loss_reward(self, metrics: RewardMetrics) -> float:
        """Calculate reward based on loss improvement."""
        # Loss improvement (negative means better)
        loss_improvement = metrics.loss_pre - metrics.loss_post
        
        # Bonus for significant improvements
        if loss_improvement > 0.1:
            loss_improvement += 0.05
        
        return self.weights.loss_improvement * loss_improvement
    
    def _calculate_convergence_reward(self, metrics: RewardMetrics) -> float:
        """Calculate reward based on convergence speed."""
        # Reward faster convergence (fewer epochs needed)
        convergence_reward = max(0, 100 - metrics.convergence_epochs) / 100.0
        return self.weights.convergence_speed * convergence_reward
    
    def _calculate_parameter_penalty(self, metrics: RewardMetrics) -> float:
        """Calculate penalty for adding parameters."""
        # Penalty proportional to parameters added (in millions)
        param_penalty = metrics.parameters_added / 1e6
        return -self.weights.parameter_penalty * param_penalty
    
    def _calculate_latency_penalty(self, metrics: RewardMetrics, 
                                  hardware_context: HardwareContext) -> float:
        """Calculate penalty for latency increase."""
        # Normalize by target latency
        normalized_latency = metrics.latency_increase_ms / hardware_context.latency_target_ms
        
        # Exponential penalty for exceeding target
        if metrics.latency_increase_ms > hardware_context.latency_target_ms:
            normalized_latency *= 2.0
        
        return -self.weights.latency_penalty * normalized_latency
    
    def _calculate_memory_penalty(self, metrics: RewardMetrics,
                                 hardware_context: HardwareContext) -> float:
        """Calculate penalty for memory usage increase."""
        # Normalize by available memory
        normalized_memory = metrics.memory_increase_mb / (hardware_context.memory_gb * 1024)
        
        # Penalty increases sharply near memory limit
        if normalized_memory > 0.8:
            normalized_memory *= 3.0
        
        return -self.weights.memory_penalty * normalized_memory
    
    def _calculate_power_penalty(self, metrics: RewardMetrics,
                                hardware_context: HardwareContext) -> float:
        """Calculate penalty for power consumption increase."""
        normalized_power = metrics.power_increase_watts / hardware_context.power_budget_watts
        return -self.weights.power_penalty * normalized_power
    
    def _calculate_drift_reward(self, metrics: RewardMetrics) -> float:
        """Calculate reward for maintaining low drift."""
        # Reward low drift, penalty for high drift
        if metrics.drift_level < 0.05:
            return self.weights.drift_bonus
        elif metrics.drift_level > 0.15:
            return -self.weights.drift_bonus * 0.5
        else:
            return 0.0
    
    def _calculate_stability_reward(self, metrics: RewardMetrics) -> float:
        """Calculate reward for maintaining model stability."""
        return self.weights.stability_bonus * metrics.stability_score
    
    def _calculate_security_penalty(self, metrics: RewardMetrics) -> float:
        """Calculate penalty for security violations."""
        return -self.weights.security_penalty * metrics.security_violations
    
    def _calculate_rollback_penalty(self, metrics: RewardMetrics) -> float:
        """Calculate penalty for triggering rollbacks."""
        return -self.weights.rollback_penalty if metrics.rollback_occurred else 0.0
    
    def _calculate_hardware_efficiency_reward(self, metrics: RewardMetrics,
                                            hardware_context: HardwareContext) -> float:
        """Calculate reward for efficient hardware utilization."""
        # Reward optimal utilization (not too low, not too high)
        utilization = metrics.hardware_utilization
        
        if 0.7 <= utilization <= 0.9:
            efficiency_reward = 1.0
        elif 0.5 <= utilization < 0.7:
            efficiency_reward = 0.5
        elif utilization > 0.9:
            efficiency_reward = -0.5  # Over-utilization penalty
        else:
            efficiency_reward = 0.0  # Under-utilization
        
        # Penalty for constraint violations
        constraint_penalty = metrics.constraint_violations * 0.2
        
        return self.weights.hardware_efficiency_bonus * (efficiency_reward - constraint_penalty)
    
    def _calculate_novelty_reward(self, metrics: RewardMetrics,
                                 decision_context: Dict[str, Any]) -> float:
        """Calculate reward for trying novel blueprints."""
        return self.weights.novelty_bonus * metrics.blueprint_novelty
    
    def _calculate_diversity_reward(self, decision_context: Dict[str, Any]) -> float:
        """Calculate reward for maintaining decision diversity."""
        diversity_score = decision_context.get('diversity_score', 0.0)
        return self.weights.diversity_bonus * diversity_score
    
    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward to reasonable range."""
        if not self.reward_history:
            return reward
        
        # Use running statistics for normalization
        recent_rewards = [r['total_reward'] for r in self.reward_history[-100:]]
        mean_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards) + 1e-8
        
        return (reward - mean_reward) / std_reward
    
    def get_reward_statistics(self) -> Dict[str, Any]:
        """Get comprehensive reward statistics for analysis."""
        if not self.reward_history:
            return {}
        
        total_rewards = [r['total_reward'] for r in self.reward_history]
        
        stats = {
            'total_episodes': len(self.reward_history),
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'max_reward': np.max(total_rewards),
            'min_reward': np.min(total_rewards),
            'recent_mean': np.mean(total_rewards[-20:]) if len(total_rewards) >= 20 else np.mean(total_rewards),
            'component_means': {}
        }
        
        # Component statistics
        for component, values in self.component_history.items():
            stats['component_means'][component] = np.mean(values)
        
        return stats
    
    def update_weights(self, new_weights: RewardWeights):
        """Update reward weights (for hyperparameter tuning)."""
        self.weights = new_weights
    
    def reset_history(self):
        """Reset reward history (for new experiments)."""
        self.reward_history.clear()
        self.component_history.clear()
```

### 2. Hyperparameter Tuning Framework

```python
# morphogenetic_engine/policy/tuning.py
import optuna
from optuna.samplers import TPESampler
from typing import Dict, List, Callable, Any, Optional
import numpy as np
from .rewards import RewardWeights, RewardCalculator

class RewardWeightTuner:
    """Hyperparameter tuning for reward weights using Optuna."""
    
    def __init__(self, 
                 objective_function: Callable[[RewardWeights], float],
                 n_trials: int = 100,
                 sampler: Optional[optuna.samplers.BaseSampler] = None):
        
        self.objective_function = objective_function
        self.n_trials = n_trials
        self.sampler = sampler or TPESampler()
        
        # Create Optuna study
        self.study = optuna.create_study(
            direction='maximize',
            sampler=self.sampler,
            study_name='kasmina_reward_tuning'
        )
        
        self.best_weights = None
        self.tuning_history = []
    
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function for reward weight optimization."""
        
        # Define search space for reward weights
        weights = RewardWeights(
            accuracy_gain=trial.suggest_float('accuracy_gain', 1.0, 20.0),
            loss_improvement=trial.suggest_float('loss_improvement', 1.0, 10.0),
            convergence_speed=trial.suggest_float('convergence_speed', 0.5, 5.0),
            
            parameter_penalty=trial.suggest_float('parameter_penalty', 0.001, 0.1),
            latency_penalty=trial.suggest_float('latency_penalty', 0.01, 1.0),
            memory_penalty=trial.suggest_float('memory_penalty', 0.01, 0.5),
            power_penalty=trial.suggest_float('power_penalty', 0.001, 0.1),
            
            drift_bonus=trial.suggest_float('drift_bonus', 1.0, 10.0),
            stability_bonus=trial.suggest_float('stability_bonus', 0.5, 5.0),
            security_penalty=trial.suggest_float('security_penalty', 5.0, 50.0),
            rollback_penalty=trial.suggest_float('rollback_penalty', 5.0, 30.0),
            
            hardware_efficiency_bonus=trial.suggest_float('hardware_efficiency_bonus', 0.5, 5.0),
            utilization_bonus=trial.suggest_float('utilization_bonus', 0.1, 2.0),
            
            novelty_bonus=trial.suggest_float('novelty_bonus', 0.1, 2.0),
            diversity_bonus=trial.suggest_float('diversity_bonus', 0.1, 1.0)
        )
        
        # Evaluate performance with these weights
        performance = self.objective_function(weights)
        
        # Store trial information
        trial_info = {
            'trial_number': trial.number,
            'weights': weights,
            'performance': performance,
            'params': trial.params
        }
        
        self.tuning_history.append(trial_info)
        
        return performance
    
    def tune(self) -> RewardWeights:
        """Run hyperparameter tuning and return best weights."""
        
        print(f"Starting reward weight tuning with {self.n_trials} trials...")
        
        # Run optimization
        self.study.optimize(self.objective, n_trials=self.n_trials)
        
        # Extract best parameters
        best_params = self.study.best_params
        self.best_weights = RewardWeights(**best_params)
        
        print(f"Best performance: {self.study.best_value:.4f}")
        print("Best weights:")
        for param, value in best_params.items():
            print(f"  {param}: {value:.6f}")
        
        return self.best_weights
    
    def get_tuning_report(self) -> Dict[str, Any]:
        """Generate comprehensive tuning report."""
        
        if not self.tuning_history:
            return {"error": "No tuning trials completed"}
        
        performances = [trial['performance'] for trial in self.tuning_history]
        
        report = {
            'total_trials': len(self.tuning_history),
            'best_performance': max(performances),
            'worst_performance': min(performances),
            'mean_performance': np.mean(performances),
            'std_performance': np.std(performances),
            'best_trial': max(self.tuning_history, key=lambda x: x['performance']),
            'improvement_over_baseline': None,  # Would compare to default weights
            'convergence_analysis': self._analyze_convergence(),
            'parameter_importance': self._analyze_parameter_importance()
        }
        
        return report
    
    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analyze convergence behavior during tuning."""
        
        performances = [trial['performance'] for trial in self.tuning_history]
        
        # Calculate running best
        running_best = []
        best_so_far = float('-inf')
        
        for perf in performances:
            best_so_far = max(best_so_far, perf)
            running_best.append(best_so_far)
        
        # Detect convergence
        window_size = min(20, len(performances) // 4)
        if len(performances) >= window_size:
            recent_variance = np.var(running_best[-window_size:])
            converged = recent_variance < 0.01
        else:
            converged = False
        
        return {
            'converged': converged,
            'running_best': running_best,
            'final_improvement_rate': (running_best[-1] - running_best[-10]) / 10 if len(running_best) >= 10 else 0
        }
    
    def _analyze_parameter_importance(self) -> Dict[str, float]:
        """Analyze relative importance of different parameters."""
        
        if len(self.tuning_history) < 10:
            return {}
        
        # Simple correlation analysis between parameters and performance
        param_importance = {}
        performances = [trial['performance'] for trial in self.tuning_history]
        
        # Get all parameter names
        param_names = list(self.tuning_history[0]['params'].keys())
        
        for param_name in param_names:
            param_values = [trial['params'][param_name] for trial in self.tuning_history]
            
            # Calculate correlation with performance
            correlation = np.corrcoef(param_values, performances)[0, 1]
            param_importance[param_name] = abs(correlation) if not np.isnan(correlation) else 0.0
        
        return param_importance

class PolicyHyperparameterTuner:
    """Comprehensive hyperparameter tuning for the entire policy system."""
    
    def __init__(self, training_function: Callable[[Dict], float]):
        self.training_function = training_function
        
    def tune_policy_network(self, n_trials: int = 50) -> Dict[str, Any]:
        """Tune policy network hyperparameters."""
        
        def objective(trial: optuna.Trial) -> float:
            hyperparams = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'hidden_dim': trial.suggest_categorical('hidden_dim', [128, 256, 512]),
                'dropout': trial.suggest_float('dropout', 0.0, 0.5),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                'gamma': trial.suggest_float('gamma', 0.9, 0.999),
                'buffer_size': trial.suggest_categorical('buffer_size', [5000, 10000, 20000]),
            }
            
            return self.training_function(hyperparams)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'study': study
        }
```

### 3. Intensity Control System

```python
# morphogenetic_engine/policy/intensity.py
import torch
from torch import nn
from typing import Dict, List, Optional, Any
from ..telemetry.types import SeedTelemetry
from ..blueprints.base import Blueprint

class IntensityController:
    """Advanced intensity control system for seed learning rates and activation."""
    
    def __init__(self, base_learning_rate: float = 1e-3):
        self.base_learning_rate = base_learning_rate
        self.intensity_history = {}
        self.performance_history = {}
        
        # Blueprint-specific intensity mappings
        self.blueprint_intensity_configs = {
            'no_op': {'min_intensity': 0.0, 'max_intensity': 0.0, 'default': 0.0},
            'bottleneck_adapter': {'min_intensity': 0.1, 'max_intensity': 1.0, 'default': 0.5},
            'low_rank_residual': {'min_intensity': 0.2, 'max_intensity': 1.0, 'default': 0.6},
            'se_module': {'min_intensity': 0.1, 'max_intensity': 0.8, 'default': 0.4},
            'depthwise_conv': {'min_intensity': 0.2, 'max_intensity': 0.9, 'default': 0.5},
            'mini_self_attention': {'min_intensity': 0.05, 'max_intensity': 0.7, 'default': 0.3},
            'adapter': {'min_intensity': 0.1, 'max_intensity': 1.0, 'default': 0.5},
            'denoising_autoencoder': {'min_intensity': 0.3, 'max_intensity': 1.0, 'default': 0.7},
            'attention_filter': {'min_intensity': 0.1, 'max_intensity': 0.6, 'default': 0.3},
            'glu': {'min_intensity': 0.2, 'max_intensity': 0.8, 'default': 0.4},
            'residual_mlp': {'min_intensity': 0.1, 'max_intensity': 1.0, 'default': 0.5}
        }
        
        # Safety override protocols
        self.safety_overrides = {
            'drift_threshold': 0.15,
            'performance_drop_threshold': 0.05,
            'emergency_reduction_factor': 0.1
        }
    
    def calculate_intensity(self, 
                          blueprint_name: str,
                          policy_intensity: float,
                          telemetry: SeedTelemetry,
                          safety_scores: Dict[str, float]) -> float:
        """
        Calculate final intensity value considering policy, blueprint, and safety constraints.
        
        Args:
            blueprint_name: Name of the blueprint being activated
            policy_intensity: Raw intensity from policy network [0, 1]
            telemetry: Current seed telemetry
            safety_scores: Safety evaluation scores
            
        Returns:
            Final intensity value for learning rate scaling
        """
        
        # Get blueprint-specific constraints
        config = self.blueprint_intensity_configs.get(blueprint_name, {
            'min_intensity': 0.0, 'max_intensity': 1.0, 'default': 0.5
        })
        
        # Start with policy intensity
        intensity = policy_intensity
        
        # Apply blueprint-specific bounds
        intensity = max(config['min_intensity'], min(config['max_intensity'], intensity))
        
        # Apply safety overrides
        intensity = self._apply_safety_overrides(intensity, telemetry, safety_scores)
        
        # Apply adaptive adjustments based on history
        intensity = self._apply_adaptive_adjustments(blueprint_name, intensity, telemetry)
        
        # Store in history
        self._record_intensity_decision(blueprint_name, intensity, telemetry)
        
        return intensity
    
    def _apply_safety_overrides(self, 
                               intensity: float,
                               telemetry: SeedTelemetry,
                               safety_scores: Dict[str, float]) -> float:
        """Apply safety-based intensity reductions."""
        
        # High drift risk - reduce intensity
        if telemetry.interface_drift > self.safety_overrides['drift_threshold']:
            drift_factor = 1.0 - (telemetry.interface_drift - self.safety_overrides['drift_threshold']) * 2.0
            intensity *= max(0.1, drift_factor)
        
        # Security alert - emergency reduction
        if safety_scores.get('security_alert', 0.0) > 0.5:
            intensity *= self.safety_overrides['emergency_reduction_factor']
        
        # High rollback need - be conservative
        if safety_scores.get('rollback_need', 0.0) > 0.7:
            intensity *= 0.5
        
        return max(0.0, intensity)
    
    def _apply_adaptive_adjustments(self, 
                                   blueprint_name: str,
                                   intensity: float,
                                   telemetry: SeedTelemetry) -> float:
        """Apply adaptive adjustments based on historical performance."""
        
        if blueprint_name not in self.performance_history:
            return intensity
        
        # Get recent performance for this blueprint type
        recent_performance = self.performance_history[blueprint_name][-5:]
        
        if len(recent_performance) < 2:
            return intensity
        
        # Calculate performance trend
        performance_trend = np.mean([p['outcome'] for p in recent_performance[-3:]])
        
        # Adjust based on recent success/failure
        if performance_trend > 0.8:  # Recent successes
            intensity *= 1.2  # Increase intensity
        elif performance_trend < 0.3:  # Recent failures
            intensity *= 0.8  # Decrease intensity
        
        return intensity
    
    def _record_intensity_decision(self, 
                                  blueprint_name: str,
                                  intensity: float,
                                  telemetry: SeedTelemetry):
        """Record intensity decision for learning."""
        
        if blueprint_name not in self.intensity_history:
            self.intensity_history[blueprint_name] = []
        
        decision_record = {
            'intensity': intensity,
            'drift': telemetry.interface_drift,
            'utilization': telemetry.utilization_score,
            'age': telemetry.age_steps,
            'timestamp': len(self.intensity_history[blueprint_name])
        }
        
        self.intensity_history[blueprint_name].append(decision_record)
    
    def apply_intensity_to_optimizer(self, 
                                   optimizer: torch.optim.Optimizer,
                                   intensity: float,
                                   seed_parameters: List[torch.nn.Parameter]):
        """Apply intensity scaling to optimizer for specific seed parameters."""
        
        # Calculate effective learning rate
        effective_lr = self.base_learning_rate * intensity
        
        # Update learning rate for seed parameters
        for param_group in optimizer.param_groups:
            # Check if any parameters in this group belong to the seed
            group_params = set(param_group['params'])
            seed_param_set = set(seed_parameters)
            
            if group_params.intersection(seed_param_set):
                param_group['lr'] = effective_lr
    
    def create_gradual_activation_schedule(self, 
                                         intensity: float,
                                         warmup_steps: int = 50) -> Callable[[int], float]:
        """Create a gradual activation schedule for smooth seed integration."""
        
        def schedule(step: int) -> float:
            if step < warmup_steps:
                # Gradual ramp-up
                ramp_factor = step / warmup_steps
                return intensity * ramp_factor
            else:
                # Full intensity
                return intensity
        
        return schedule
    
    def record_performance_outcome(self, 
                                 blueprint_name: str,
                                 intensity_used: float,
                                 outcome_metrics: Dict[str, float]):
        """Record performance outcome for adaptive learning."""
        
        if blueprint_name not in self.performance_history:
            self.performance_history[blueprint_name] = []
        
        # Calculate overall outcome score
        accuracy_gain = outcome_metrics.get('accuracy_gain', 0.0)
        stability = 1.0 - outcome_metrics.get('drift_increase', 0.0)
        efficiency = 1.0 / max(outcome_metrics.get('latency_increase', 1.0), 0.1)
        
        outcome_score = (accuracy_gain + stability + efficiency) / 3.0
        
        outcome_record = {
            'intensity': intensity_used,
            'outcome': outcome_score,
            'metrics': outcome_metrics.copy(),
            'timestamp': len(self.performance_history[blueprint_name])
        }
        
        self.performance_history[blueprint_name].append(outcome_record)
    
    def get_intensity_statistics(self) -> Dict[str, Any]:
        """Get comprehensive intensity usage statistics."""
        
        stats = {
            'blueprint_stats': {},
            'overall_stats': {
                'total_decisions': sum(len(history) for history in self.intensity_history.values()),
                'blueprints_used': len(self.intensity_history),
                'avg_intensity': 0.0,
                'intensity_distribution': {}
            }
        }
        
        all_intensities = []
        
        # Per-blueprint statistics
        for blueprint_name, history in self.intensity_history.items():
            intensities = [record['intensity'] for record in history]
            
            stats['blueprint_stats'][blueprint_name] = {
                'usage_count': len(history),
                'avg_intensity': np.mean(intensities),
                'std_intensity': np.std(intensities),
                'min_intensity': np.min(intensities),
                'max_intensity': np.max(intensities),
                'recent_trend': self._calculate_intensity_trend(intensities)
            }
            
            all_intensities.extend(intensities)
        
        # Overall statistics
        if all_intensities:
            stats['overall_stats']['avg_intensity'] = np.mean(all_intensities)
            
            # Intensity distribution
            hist, bins = np.histogram(all_intensities, bins=10, range=(0, 1))
            stats['overall_stats']['intensity_distribution'] = {
                'histogram': hist.tolist(),
                'bins': bins.tolist()
            }
        
        return stats
    
    def _calculate_intensity_trend(self, intensities: List[float]) -> str:
        """Calculate trend direction for intensity values."""
        
        if len(intensities) < 5:
            return "insufficient_data"
        
        recent = intensities[-5:]
        early = intensities[-10:-5] if len(intensities) >= 10 else intensities[:-5]
        
        if not early:
            return "insufficient_data"
        
        recent_avg = np.mean(recent)
        early_avg = np.mean(early)
        
        if recent_avg > early_avg * 1.1:
            return "increasing"
        elif recent_avg < early_avg * 0.9:
            return "decreasing"
        else:
            return "stable"

class FailSafeGating:
    """Emergency fallback gating for seed activation failures."""
    
    def __init__(self, fallback_threshold: float = 0.1):
        self.fallback_threshold = fallback_threshold
        self.emergency_activations = {}
        
    def create_failsafe_gate(self, 
                           original_output: torch.Tensor,
                           seed_output: torch.Tensor,
                           confidence_score: float) -> torch.Tensor:
        """Create failsafe gating between original and seed output."""
        
        # Calculate gating weight based on confidence
        if confidence_score > self.fallback_threshold:
            gate_weight = confidence_score
        else:
            # Emergency fallback to original
            gate_weight = 0.0
            self._record_emergency_fallback()
        
        # Apply gating
        gated_output = gate_weight * seed_output + (1 - gate_weight) * original_output
        
        return gated_output
    
    def _record_emergency_fallback(self):
        """Record emergency fallback event."""
        timestamp = len(self.emergency_activations)
        self.emergency_activations[timestamp] = {
            'event': 'emergency_fallback',
            'timestamp': timestamp
        }
```

## Integration and Testing

### 1. Complete Policy System Integration

```python
# morphogenetic_engine/policy/complete_system.py
from typing import Dict, Any, Optional, Tuple
import torch
from .network import KasminaPolicyNetwork
from .rewards import RewardCalculator, RewardMetrics, RewardWeights
from .intensity import IntensityController, FailSafeGating
from .tuning import RewardWeightTuner
from ..telemetry.types import SeedTelemetry, HardwareContext

class CompletePolicySystem:
    """Complete integrated policy system with all components."""
    
    def __init__(self, 
                 hardware_context: HardwareContext,
                 reward_weights: Optional[RewardWeights] = None,
                 policy_config: Optional[Dict] = None):
        
        self.hardware_context = hardware_context
        
        # Initialize components
        self.policy_network = KasminaPolicyNetwork(**(policy_config or {}))
        self.reward_calculator = RewardCalculator(reward_weights)
        self.intensity_controller = IntensityController()
        self.failsafe_gating = FailSafeGating()
        
        # System state
        self.system_state = {
            'total_decisions': 0,
            'successful_activations': 0,
            'failed_activations': 0,
            'emergency_fallbacks': 0
        }
        
    def make_decision(self, 
                     global_telemetry: Dict[str, SeedTelemetry],
                     model_state: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """
        Make a complete decision about seed activation.
        
        Returns:
            Complete decision package or None if no action should be taken
        """
        
        # Get policy decision
        with torch.no_grad():
            policy_output = self.policy_network(
                global_telemetry, self.hardware_context, model_state
            )
        
        # Interpret timing decision
        timing_probs = torch.softmax(policy_output['timing_control'], dim=-1)
        timing_decision = torch.argmax(timing_probs).item()
        
        # Only proceed if timing says "now"
        if timing_decision != 0:
            return None
        
        # Extract decision components
        blueprint_probs = torch.softmax(policy_output['blueprint_choice'], dim=-1)
        blueprint_idx = torch.argmax(blueprint_probs).item()
        
        location_probs = torch.softmax(policy_output['location_selection'], dim=-1)
        location_idx = torch.argmax(location_probs).item()
        
        policy_intensity = policy_output['intensity_control'].item()
        
        # Map to actual blueprint name
        from ..blueprints.registry import BlueprintRegistry
        blueprint_names = BlueprintRegistry.list_blueprints()
        blueprint_name = blueprint_names[blueprint_idx] if blueprint_idx < len(blueprint_names) else "no_op"
        
        # Calculate final intensity with safety considerations
        # Use first available seed's telemetry for safety calculations
        representative_telemetry = next(iter(global_telemetry.values())) if global_telemetry else None
        
        if representative_telemetry:
            final_intensity = self.intensity_controller.calculate_intensity(
                blueprint_name, 
                policy_intensity,
                representative_telemetry,
                policy_output['safety_scores']
            )
        else:
            final_intensity = policy_intensity
        
        # Create complete decision package
        decision = {
            'blueprint_name': blueprint_name,
            'location_index': location_idx,
            'intensity': final_intensity,
            'policy_output': policy_output,
            'confidence_scores': {
                'blueprint': blueprint_probs[blueprint_idx].item(),
                'location': location_probs[location_idx].item(),
                'timing': timing_probs[timing_decision].item(),
                'overall': (blueprint_probs[blueprint_idx] * 
                           location_probs[location_idx] * 
                           timing_probs[timing_decision]).item()
            },
            'safety_assessment': policy_output['safety_scores'],
            'hardware_assessment': policy_output['hardware_scores']
        }
        
        self.system_state['total_decisions'] += 1
        
        return decision
    
    def execute_and_evaluate(self, 
                           decision: Dict[str, Any],
                           execution_function: Callable[[Dict], bool],
                           evaluation_function: Callable[[], Dict[str, float]]) -> Dict[str, Any]:
        """
        Execute decision and evaluate outcome for learning.
        
        Args:
            decision: Decision package from make_decision
            execution_function: Function to execute the germination
            evaluation_function: Function to evaluate post-execution metrics
            
        Returns:
            Comprehensive evaluation results
        """
        
        # Execute the decision
        execution_success = execution_function(decision)
        
        if execution_success:
            self.system_state['successful_activations'] += 1
        else:
            self.system_state['failed_activations'] += 1
            
        # Evaluate outcome
        outcome_metrics = evaluation_function()
        
        # Create reward metrics
        reward_metrics = RewardMetrics(
            accuracy_pre=outcome_metrics.get('accuracy_pre', 0.0),
            accuracy_post=outcome_metrics.get('accuracy_post', 0.0),
            loss_pre=outcome_metrics.get('loss_pre', 1.0),
            loss_post=outcome_metrics.get('loss_post', 1.0),
            convergence_epochs=outcome_metrics.get('convergence_epochs', 100),
            parameters_added=outcome_metrics.get('parameters_added', 0),
            latency_increase_ms=outcome_metrics.get('latency_increase_ms', 0.0),
            memory_increase_mb=outcome_metrics.get('memory_increase_mb', 0.0),
            power_increase_watts=outcome_metrics.get('power_increase_watts', 0.0),
            drift_level=outcome_metrics.get('drift_level', 0.0),
            stability_score=outcome_metrics.get('stability_score', 1.0),
            security_violations=outcome_metrics.get('security_violations', 0),
            rollback_occurred=outcome_metrics.get('rollback_occurred', False),
            hardware_utilization=outcome_metrics.get('hardware_utilization', 0.5),
            constraint_violations=outcome_metrics.get('constraint_violations', 0),
            blueprint_novelty=outcome_metrics.get('blueprint_novelty', 0.0),
            decision_diversity=outcome_metrics.get('decision_diversity', 0.0)
        )
        
        # Calculate reward
        reward_result = self.reward_calculator.calculate_reward(
            reward_metrics, self.hardware_context, decision
        )
        
        # Record performance for intensity controller
        self.intensity_controller.record_performance_outcome(
            decision['blueprint_name'],
            decision['intensity'],
            outcome_metrics
        )
        
        return {
            'execution_success': execution_success,
            'reward_result': reward_result,
            'outcome_metrics': outcome_metrics,
            'decision': decision
        }
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system performance statistics."""
        
        return {
            'system_state': self.system_state.copy(),
            'reward_stats': self.reward_calculator.get_reward_statistics(),
            'intensity_stats': self.intensity_controller.get_intensity_statistics(),
            'policy_performance': {
                'success_rate': self.system_state['successful_activations'] / max(self.system_state['total_decisions'], 1),
                'failure_rate': self.system_state['failed_activations'] / max(self.system_state['total_decisions'], 1),
                'emergency_rate': self.system_state['emergency_fallbacks'] / max(self.system_state['total_decisions'], 1)
            }
        }
    
    def optimize_reward_weights(self, 
                              validation_function: Callable[[RewardWeights], float],
                              n_trials: int = 50) -> RewardWeights:
        """Optimize reward weights using the validation function."""
        
        tuner = RewardWeightTuner(validation_function, n_trials)
        optimized_weights = tuner.tune()
        
        # Update reward calculator with optimized weights
        self.reward_calculator.update_weights(optimized_weights)
        
        return optimized_weights
```

### 2. Testing Suite

```python
# tests/test_complete_policy_system.py
import pytest
import torch
from morphogenetic_engine.policy.complete_system import CompletePolicySystem
from morphogenetic_engine.policy.rewards import RewardWeights, RewardMetrics
from morphogenetic_engine.telemetry.types import SeedTelemetry, HardwareContext

class TestCompletePolicySystem:
    
    @pytest.fixture
    def hardware_context(self):
        return HardwareContext(
            device_type="gpu", memory_gb=24.0, flops_per_ms=1e9,
            bandwidth_gbps=1008.0, power_budget_watts=450.0, latency_target_ms=10.0
        )
    
    @pytest.fixture
    def sample_telemetry(self):
        return {
            'seed1': SeedTelemetry(
                seed_id='seed1',
                activation_variance=0.5,
                interface_drift=0.1,
                gradient_norm=2.0,
                utilization_score=0.8
            )
        }
    
    @pytest.fixture
    def policy_system(self, hardware_context):
        return CompletePolicySystem(hardware_context)
    
    def test_complete_decision_making(self, policy_system, sample_telemetry):
        """Test complete decision making pipeline."""
        
        model_state = {
            'accuracy': 0.85,
            'loss': 0.5,
            'epoch': 100,
            'learning_rate': 1e-3
        }
        
        decision = policy_system.make_decision(sample_telemetry, model_state)
        
        # Decision might be None if timing says "later"
        if decision is not None:
            # Check decision structure
            assert 'blueprint_name' in decision
            assert 'location_index' in decision
            assert 'intensity' in decision
            assert 'confidence_scores' in decision
            assert 'safety_assessment' in decision
            assert 'hardware_assessment' in decision
            
            # Check value ranges
            assert 0.0 <= decision['intensity'] <= 1.0
            assert decision['location_index'] >= 0
    
    def test_reward_calculation(self, policy_system):
        """Test reward calculation system."""
        
        metrics = RewardMetrics(
            accuracy_pre=0.80,
            accuracy_post=0.85,
            loss_pre=0.6,
            loss_post=0.5,
            convergence_epochs=50,
            parameters_added=10000,
            latency_increase_ms=2.0,
            memory_increase_mb=10.0,
            power_increase_watts=5.0,
            drift_level=0.05,
            stability_score=0.9,
            security_violations=0,
            rollback_occurred=False,
            hardware_utilization=0.7,
            constraint_violations=0,
            blueprint_novelty=0.5,
            decision_diversity=0.6
        )
        
        reward_result = policy_system.reward_calculator.calculate_reward(
            metrics, policy_system.hardware_context, {}
        )
        
        assert 'total_reward' in reward_result
        assert 'components' in reward_result
        assert 'normalized_reward' in reward_result
        
        # Check that components exist
        components = reward_result['components']
        assert 'accuracy_gain' in components
        assert 'parameter_penalty' in components
        assert 'drift_bonus' in components
    
    def test_intensity_control(self, policy_system, sample_telemetry):
        """Test intensity control system."""
        
        blueprint_name = "bottleneck_adapter"
        policy_intensity = 0.7
        safety_scores = {'security_alert': 0.0, 'rollback_need': 0.2}
        
        telemetry = list(sample_telemetry.values())[0]
        
        final_intensity = policy_system.intensity_controller.calculate_intensity(
            blueprint_name, policy_intensity, telemetry, safety_scores
        )
        
        assert 0.0 <= final_intensity <= 1.0
        
        # Test safety override
        high_drift_telemetry = SeedTelemetry(
            seed_id='test',
            interface_drift=0.25,  # High drift
            activation_variance=0.5,
            gradient_norm=2.0,
            utilization_score=0.8
        )
        
        reduced_intensity = policy_system.intensity_controller.calculate_intensity(
            blueprint_name, policy_intensity, high_drift_telemetry, safety_scores
        )
        
        assert reduced_intensity < final_intensity  # Should be reduced due to high drift
    
    def test_system_statistics(self, policy_system):
        """Test system statistics collection."""
        
        stats = policy_system.get_system_statistics()
        
        assert 'system_state' in stats
        assert 'reward_stats' in stats
        assert 'intensity_stats' in stats
        assert 'policy_performance' in stats
        
        # Check system state structure
        system_state = stats['system_state']
        assert 'total_decisions' in system_state
        assert 'successful_activations' in system_state
        assert 'failed_activations' in system_state
```

## Deliverables Checklist for Stage 3B

- [ ] Multi-objective reward calculation system
- [ ] Hyperparameter tuning framework with Optuna
- [ ] Advanced intensity control with safety overrides
- [ ] Blueprint-specific intensity configurations
- [ ] Gradual activation scheduling system
- [ ] Failsafe gating mechanisms
- [ ] Performance outcome tracking and adaptation
- [ ] Complete policy system integration
- [ ] Comprehensive testing suite
- [ ] Statistics and monitoring framework
- [ ] Documentation and usage examples

## Stage 3B Success Criteria

1. **Functional Requirements**
   - Reward system produces stable, meaningful signals
   - Intensity control respects safety constraints
   - Hyperparameter tuning improves performance
   - Integration with Stage 3A components seamless

2. **Performance Requirements**
   - Reward calculation <5ms per evaluation
   - Intensity control decisions <2ms
   - Tuning converges within reasonable iterations
   - No memory leaks or performance degradation

3. **Quality Requirements**
   - Reward components are interpretable and balanced
   - Safety overrides trigger appropriately
   - Statistical tracking provides actionable insights
   - System handles edge cases gracefully

This completes Stage 3B, providing a comprehensive reward and control system that enables the policy network to learn effectively and make safe, intelligent decisions about blueprint deployment.
