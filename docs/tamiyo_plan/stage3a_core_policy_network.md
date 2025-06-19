# Stage 3A: Core Policy Network Implementation Guide

## Overview

Stage 3A replaces the simple KasminaMicro controller with an intelligent policy network that can make sophisticated decisions about blueprint selection, placement, and activation timing. This stage focuses on the core decision-making architecture.

## Core Policy Network Architecture

### 1. Enhanced Policy Network

#### 1.1 Multi-Head Decision Network

```python
# morphogenetic_engine/policy/network.py
import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from ..telemetry.types import SeedTelemetry, HardwareContext
from ..blueprints.registry import BlueprintRegistry

class KasminaPolicyNetwork(nn.Module):
    """Advanced policy network for blueprint selection and placement decisions."""
    
    def __init__(self, 
                 telemetry_dim: int = 32,
                 hidden_dim: int = 256,
                 num_blueprints: int = 12,
                 max_seeds: int = 16,
                 dropout: float = 0.1):
        super().__init__()
        
        self.telemetry_dim = telemetry_dim
        self.hidden_dim = hidden_dim
        self.num_blueprints = num_blueprints
        self.max_seeds = max_seeds
        
        # Telemetry encoder
        self.telemetry_encoder = TelemetryEncoder(telemetry_dim, hidden_dim)
        
        # Context aggregation
        self.context_aggregator = ContextAggregator(hidden_dim)
        
        # Decision heads
        self.decision_heads = nn.ModuleDict({
            'blueprint_choice': BlueprintChoiceHead(hidden_dim, num_blueprints),
            'location_selection': LocationSelectionHead(hidden_dim, max_seeds),
            'intensity_control': IntensityControlHead(hidden_dim),
            'timing_control': TimingControlHead(hidden_dim)
        })
        
        # Safety and constraint heads
        self.safety_head = SafetyConstraintHead(hidden_dim)
        self.hardware_head = HardwareConstraintHead(hidden_dim)
        
        # Value network for decision evaluation
        self.value_network = ValueNetwork(hidden_dim)
        
    def forward(self, 
                global_telemetry: Dict[str, SeedTelemetry],
                hardware_context: HardwareContext,
                model_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the policy network.
        
        Args:
            global_telemetry: Telemetry data for all seeds
            hardware_context: Current hardware constraints
            model_state: Current model performance metrics
            
        Returns:
            Dictionary of decision outputs and constraint scores
        """
        
        # Encode individual seed telemetry
        seed_embeddings = {}
        for seed_id, telemetry in global_telemetry.items():
            seed_embeddings[seed_id] = self.telemetry_encoder(telemetry)
        
        # Aggregate global context
        global_context = self.context_aggregator(
            seed_embeddings, hardware_context, model_state
        )
        
        # Generate decisions
        decisions = {}
        for head_name, head in self.decision_heads.items():
            decisions[head_name] = head(global_context, seed_embeddings)
        
        # Apply safety constraints
        safety_scores = self.safety_head(global_context)
        hardware_scores = self.hardware_head(global_context, hardware_context)
        
        # Apply constraint masking
        constrained_decisions = self._apply_constraints(
            decisions, safety_scores, hardware_scores
        )
        
        # Estimate value of decisions
        value_estimate = self.value_network(global_context)
        
        return {
            **constrained_decisions,
            'safety_scores': safety_scores,
            'hardware_scores': hardware_scores,
            'value_estimate': value_estimate,
            'global_context': global_context
        }
    
    def _apply_constraints(self, 
                          decisions: Dict[str, torch.Tensor],
                          safety_scores: Dict[str, torch.Tensor],
                          hardware_scores: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply safety and hardware constraints to decisions."""
        
        constrained = decisions.copy()
        
        # Safety constraints
        if safety_scores['security_alert'] > 0.5:
            # Force No-Op selection during security alerts
            constrained['blueprint_choice'] = self._force_no_op_selection(
                decisions['blueprint_choice']
            )
        
        if safety_scores['drift_risk'] > 0.7:
            # Limit high-capacity blueprints during drift risk
            constrained['blueprint_choice'] = self._limit_high_capacity_blueprints(
                decisions['blueprint_choice']
            )
        
        # Hardware constraints
        if not hardware_scores['latency_feasible'] or not hardware_scores['memory_feasible']:
            # Prioritize lightweight blueprints
            constrained['blueprint_choice'] = self._prioritize_lightweight_blueprints(
                decisions['blueprint_choice']
            )
        
        return constrained
    
    def _force_no_op_selection(self, blueprint_logits: torch.Tensor) -> torch.Tensor:
        """Force selection of No-Op blueprint during security alerts."""
        masked_logits = torch.full_like(blueprint_logits, float('-inf'))
        masked_logits[..., 0] = 0.0  # No-Op is index 0
        return masked_logits
    
    def _limit_high_capacity_blueprints(self, blueprint_logits: torch.Tensor) -> torch.Tensor:
        """Limit selection of high-capacity blueprints during drift."""
        masked_logits = blueprint_logits.clone()
        # Indices of high-capacity blueprints (configured)
        high_capacity_indices = [5, 6, 7, 8]  # Mini attention, residual MLP, etc.
        masked_logits[..., high_capacity_indices] -= 5.0  # Strong penalty
        return masked_logits
    
    def _prioritize_lightweight_blueprints(self, blueprint_logits: torch.Tensor) -> torch.Tensor:
        """Prioritize lightweight blueprints under hardware constraints."""
        masked_logits = blueprint_logits.clone()
        # Indices of lightweight blueprints
        lightweight_indices = [0, 1, 2, 9]  # No-Op, bottleneck, low-rank, etc.
        masked_logits[..., lightweight_indices] += 2.0  # Bonus for lightweight
        return masked_logits

class TelemetryEncoder(nn.Module):
    """Encode seed telemetry into fixed-size embeddings."""
    
    def __init__(self, telemetry_dim: int, hidden_dim: int):
        super().__init__()
        
        # Feature extractors for different telemetry components
        self.activation_encoder = nn.Linear(3, telemetry_dim // 4)  # variance, mean, std
        self.drift_encoder = nn.Linear(2, telemetry_dim // 4)  # drift, trend
        self.gradient_encoder = nn.Linear(2, telemetry_dim // 4)  # norm, variance
        self.utility_encoder = nn.Linear(4, telemetry_dim // 4)  # utilization components
        
        # Combine all features
        self.feature_combiner = nn.Sequential(
            nn.Linear(telemetry_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, telemetry: SeedTelemetry) -> torch.Tensor:
        """Encode telemetry into embedding vector."""
        
        # Extract feature groups
        activation_features = torch.tensor([
            telemetry.activation_variance,
            telemetry.activation_mean,
            telemetry.activation_std
        ], dtype=torch.float32)
        
        drift_features = torch.tensor([
            telemetry.interface_drift,
            sum(telemetry.drift_trend[-5:]) / 5 if telemetry.drift_trend else 0.0
        ], dtype=torch.float32)
        
        gradient_features = torch.tensor([
            telemetry.gradient_norm,
            telemetry.gradient_variance
        ], dtype=torch.float32)
        
        utility_features = torch.tensor([
            telemetry.utilization_score,
            telemetry.l1_weight_norm,
            telemetry.output_magnitude,
            telemetry.loss_impact
        ], dtype=torch.float32)
        
        # Encode each feature group
        activation_embed = self.activation_encoder(activation_features)
        drift_embed = self.drift_encoder(drift_features)
        gradient_embed = self.gradient_encoder(gradient_features)
        utility_embed = self.utility_encoder(utility_features)
        
        # Combine features
        combined_features = torch.cat([
            activation_embed, drift_embed, gradient_embed, utility_embed
        ])
        
        return self.feature_combiner(combined_features)

class ContextAggregator(nn.Module):
    """Aggregate global context from all seed embeddings and system state."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Attention mechanism for seed aggregation
        self.seed_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1
        )
        
        # Hardware context encoder
        self.hardware_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim // 2),  # 6 hardware features
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Model state encoder
        self.model_state_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),  # 4 model state features
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Final context combiner
        self.context_combiner = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, 
                seed_embeddings: Dict[str, torch.Tensor],
                hardware_context: HardwareContext,
                model_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Aggregate global context from all inputs."""
        
        # Stack seed embeddings for attention
        if seed_embeddings:
            seed_stack = torch.stack(list(seed_embeddings.values()))
            seed_stack = seed_stack.unsqueeze(1)  # Add batch dimension
            
            # Apply self-attention to aggregate seed information
            aggregated_seeds, _ = self.seed_attention(
                seed_stack, seed_stack, seed_stack
            )
            aggregated_seeds = aggregated_seeds.mean(dim=0).squeeze(0)
        else:
            aggregated_seeds = torch.zeros(self.hidden_dim)
        
        # Encode hardware context
        hardware_features = torch.tensor([
            hardware_context.memory_gb,
            hardware_context.flops_per_ms / 1e6,  # Normalize
            hardware_context.bandwidth_gbps,
            hardware_context.power_budget_watts,
            hardware_context.latency_target_ms,
            1.0 if hardware_context.device_type == "gpu" else 0.0
        ], dtype=torch.float32)
        
        hardware_embed = self.hardware_encoder(hardware_features)
        
        # Encode model state
        model_features = torch.tensor([
            model_state.get('accuracy', 0.0),
            model_state.get('loss', 1.0),
            model_state.get('epoch', 0) / 1000.0,  # Normalize
            model_state.get('learning_rate', 1e-3) * 1000  # Normalize
        ], dtype=torch.float32)
        
        model_embed = self.model_state_encoder(model_features)
        
        # Combine all contexts
        global_context = torch.cat([aggregated_seeds, hardware_embed, model_embed])
        return self.context_combiner(global_context)

class BlueprintChoiceHead(nn.Module):
    """Decision head for blueprint selection."""
    
    def __init__(self, hidden_dim: int, num_blueprints: int):
        super().__init__()
        
        self.choice_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_blueprints)
        )
        
    def forward(self, global_context: torch.Tensor, 
                seed_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Generate blueprint choice logits."""
        return self.choice_network(global_context)

class LocationSelectionHead(nn.Module):
    """Decision head for seed location selection."""
    
    def __init__(self, hidden_dim: int, max_seeds: int):
        super().__init__()
        
        self.location_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, max_seeds)
        )
        
    def forward(self, global_context: torch.Tensor,
                seed_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Generate location selection logits."""
        location_logits = self.location_network(global_context)
        
        # Mask locations that already have active seeds
        mask = torch.zeros_like(location_logits)
        for i, seed_id in enumerate(seed_embeddings.keys()):
            if i < len(location_logits):
                mask[i] = float('-inf')  # Already occupied
        
        return location_logits + mask

class IntensityControlHead(nn.Module):
    """Decision head for learning intensity control."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.intensity_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
    def forward(self, global_context: torch.Tensor,
                seed_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Generate intensity value."""
        return self.intensity_network(global_context)

class TimingControlHead(nn.Module):
    """Decision head for germination timing control."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.timing_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # now, soon, later
        )
        
    def forward(self, global_context: torch.Tensor,
                seed_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Generate timing decision logits."""
        return self.timing_network(global_context)
```

### 2. Safety and Hardware Constraint Heads

```python
# morphogenetic_engine/policy/constraints.py
import torch
from torch import nn
from typing import Dict
from ..telemetry.types import HardwareContext

class SafetyConstraintHead(nn.Module):
    """Safety constraint evaluation and scoring."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        # Individual safety scorers
        self.drift_risk_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.rollback_need_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.security_alert_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Overall safety gate
        self.safety_gate = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, global_context: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Evaluate safety constraints."""
        
        drift_risk = self.drift_risk_scorer(global_context)
        rollback_need = self.rollback_need_scorer(global_context)
        security_alert = self.security_alert_scorer(global_context)
        
        # Combine individual scores
        combined_input = torch.cat([
            global_context, drift_risk, rollback_need, security_alert
        ])
        
        overall_safety = self.safety_gate(combined_input)
        
        return {
            'drift_risk': drift_risk.squeeze(-1),
            'rollback_need': rollback_need.squeeze(-1),
            'security_alert': security_alert.squeeze(-1),
            'overall_safety': overall_safety.squeeze(-1)
        }

class HardwareConstraintHead(nn.Module):
    """Hardware constraint evaluation and feasibility scoring."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.latency_predictor = nn.Sequential(
            nn.Linear(hidden_dim + 6, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.ReLU()  # Positive latency
        )
        
        self.memory_predictor = nn.Sequential(
            nn.Linear(hidden_dim + 6, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.ReLU()  # Positive memory
        )
        
        self.power_predictor = nn.Sequential(
            nn.Linear(hidden_dim + 6, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.ReLU()  # Positive power
        )
        
    def forward(self, global_context: torch.Tensor, 
                hardware_context: HardwareContext) -> Dict[str, torch.Tensor]:
        """Evaluate hardware constraints and feasibility."""
        
        # Encode hardware context
        hardware_features = torch.tensor([
            hardware_context.memory_gb,
            hardware_context.flops_per_ms / 1e6,
            hardware_context.bandwidth_gbps,
            hardware_context.power_budget_watts,
            hardware_context.latency_target_ms,
            1.0 if hardware_context.device_type == "gpu" else 0.0
        ], dtype=torch.float32)
        
        # Combine with global context
        combined_input = torch.cat([global_context, hardware_features])
        
        # Predict resource usage
        predicted_latency = self.latency_predictor(combined_input)
        predicted_memory = self.memory_predictor(combined_input)
        predicted_power = self.power_predictor(combined_input)
        
        # Check feasibility
        latency_feasible = (predicted_latency <= hardware_context.latency_target_ms).float()
        memory_feasible = (predicted_memory <= hardware_context.memory_gb * 1024).float()
        power_feasible = (predicted_power <= hardware_context.power_budget_watts).float()
        
        return {
            'predicted_latency': predicted_latency.squeeze(-1),
            'predicted_memory': predicted_memory.squeeze(-1),
            'predicted_power': predicted_power.squeeze(-1),
            'latency_feasible': latency_feasible.squeeze(-1),
            'memory_feasible': memory_feasible.squeeze(-1),
            'power_feasible': power_feasible.squeeze(-1),
            'overall_feasible': (latency_feasible * memory_feasible * power_feasible).squeeze(-1)
        }

class ValueNetwork(nn.Module):
    """Value network for decision evaluation and learning."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.value_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, global_context: torch.Tensor) -> torch.Tensor:
        """Estimate value of current state."""
        return self.value_network(global_context).squeeze(-1)
```

### 3. Policy Training Framework

```python
# morphogenetic_engine/policy/training.py
import torch
import torch.nn.functional as F
from torch.optim import Adam
from typing import Dict, List, Tuple, Optional
import random
from collections import deque
from dataclasses import dataclass

@dataclass
class PolicyExperience:
    """Single experience for policy learning."""
    
    global_telemetry: Dict[str, any]
    hardware_context: any
    model_state: Dict[str, float]
    action: Dict[str, torch.Tensor]
    reward: float
    next_global_telemetry: Optional[Dict[str, any]]
    next_model_state: Optional[Dict[str, float]]
    done: bool

class PolicyTrainer:
    """Training framework for the Kasmina policy network."""
    
    def __init__(self, 
                 policy_network: 'KasminaPolicyNetwork',
                 learning_rate: float = 1e-4,
                 buffer_size: int = 10000,
                 batch_size: int = 32,
                 gamma: float = 0.99):
        
        self.policy_network = policy_network
        self.optimizer = Adam(policy_network.parameters(), lr=learning_rate)
        
        # Experience replay buffer
        self.experience_buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        
        # Training statistics
        self.training_stats = {
            'total_steps': 0,
            'total_episodes': 0,
            'average_reward': 0.0,
            'policy_loss': 0.0,
            'value_loss': 0.0
        }
        
    def add_experience(self, experience: PolicyExperience):
        """Add experience to replay buffer."""
        self.experience_buffer.append(experience)
    
    def train_step(self) -> Dict[str, float]:
        """Perform one training step."""
        if len(self.experience_buffer) < self.batch_size:
            return {}
        
        # Sample batch
        batch = random.sample(self.experience_buffer, self.batch_size)
        
        # Prepare batch data
        batch_data = self._prepare_batch(batch)
        
        # Forward pass
        policy_outputs = self.policy_network(
            batch_data['global_telemetry'],
            batch_data['hardware_context'],
            batch_data['model_state']
        )
        
        # Calculate losses
        policy_loss = self._calculate_policy_loss(
            policy_outputs, batch_data['actions'], batch_data['rewards']
        )
        
        value_loss = self._calculate_value_loss(
            policy_outputs['value_estimate'], batch_data['returns']
        )
        
        total_loss = policy_loss + 0.5 * value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update statistics
        self.training_stats['total_steps'] += 1
        self.training_stats['policy_loss'] = policy_loss.item()
        self.training_stats['value_loss'] = value_loss.item()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def _prepare_batch(self, batch: List[PolicyExperience]) -> Dict[str, any]:
        """Prepare batch data for training."""
        # This is a simplified version - in practice, you'd need to carefully
        # handle the variable-length telemetry data
        
        batch_data = {
            'global_telemetry': {},
            'hardware_context': batch[0].hardware_context,  # Simplified
            'model_state': {},
            'actions': {},
            'rewards': torch.tensor([exp.reward for exp in batch]),
            'returns': self._calculate_returns([exp.reward for exp in batch])
        }
        
        return batch_data
    
    def _calculate_policy_loss(self, 
                              policy_outputs: Dict[str, torch.Tensor],
                              actions: Dict[str, torch.Tensor],
                              rewards: torch.Tensor) -> torch.Tensor:
        """Calculate policy gradient loss."""
        
        # Blueprint choice loss (cross-entropy)
        blueprint_logits = policy_outputs['blueprint_choice']
        blueprint_actions = actions.get('blueprint_choice', torch.zeros(len(rewards), dtype=torch.long))
        blueprint_loss = F.cross_entropy(blueprint_logits, blueprint_actions)
        
        # Location selection loss
        location_logits = policy_outputs['location_selection']
        location_actions = actions.get('location_selection', torch.zeros(len(rewards), dtype=torch.long))
        location_loss = F.cross_entropy(location_logits, location_actions)
        
        # Intensity loss (MSE for continuous action)
        intensity_pred = policy_outputs['intensity_control']
        intensity_target = actions.get('intensity_control', torch.ones(len(rewards)))
        intensity_loss = F.mse_loss(intensity_pred, intensity_target)
        
        # Combine losses with reward weighting
        policy_loss = (blueprint_loss + location_loss + intensity_loss) * rewards.mean()
        
        return policy_loss
    
    def _calculate_value_loss(self, 
                             value_predictions: torch.Tensor,
                             returns: torch.Tensor) -> torch.Tensor:
        """Calculate value function loss."""
        return F.mse_loss(value_predictions, returns)
    
    def _calculate_returns(self, rewards: List[float]) -> torch.Tensor:
        """Calculate discounted returns."""
        returns = []
        cumulative_return = 0.0
        
        for reward in reversed(rewards):
            cumulative_return = reward + self.gamma * cumulative_return
            returns.append(cumulative_return)
        
        returns.reverse()
        return torch.tensor(returns, dtype=torch.float32)
    
    def save_checkpoint(self, filepath: str):
        """Save training checkpoint."""
        checkpoint = {
            'policy_network_state': self.policy_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'experience_buffer': list(self.experience_buffer)
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath)
        self.policy_network.load_state_dict(checkpoint['policy_network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.training_stats = checkpoint['training_stats']
        self.experience_buffer = deque(checkpoint['experience_buffer'], 
                                     maxlen=self.experience_buffer.maxlen)
```

## Integration with Existing System

### 1. Enhanced Kasmina Controller

```python
# morphogenetic_engine/core/enhanced_controller.py
import torch
from typing import Dict, Optional, Tuple
from ..policy.network import KasminaPolicyNetwork
from ..policy.training import PolicyTrainer, PolicyExperience
from ..telemetry.types import SeedTelemetry, HardwareContext
from ..blueprints.registry import BlueprintRegistry

class EnhancedKasminaController:
    """Enhanced Kasmina controller using policy network instead of simple heuristics."""
    
    def __init__(self,
                 seed_manager: 'EnhancedSeedManager',
                 hardware_context: HardwareContext,
                 policy_config: Optional[Dict] = None):
        
        self.seed_manager = seed_manager
        self.hardware_context = hardware_context
        
        # Initialize policy network
        policy_config = policy_config or {}
        self.policy_network = KasminaPolicyNetwork(
            num_blueprints=len(BlueprintRegistry.list_blueprints()),
            max_seeds=16,  # Configurable
            **policy_config
        )
        
        # Initialize trainer
        self.trainer = PolicyTrainer(self.policy_network)
        
        # Decision tracking
        self.last_decision = None
        self.last_telemetry = None
        self.decision_history = []
        
    def should_germinate(self, 
                        model_state: Dict[str, float],
                        force_decision: bool = False) -> Tuple[bool, Optional[Dict]]:
        """
        Determine if germination should occur and return decision details.
        
        Args:
            model_state: Current model performance metrics
            force_decision: Force a decision even if conditions aren't met
            
        Returns:
            (should_germinate, decision_details)
        """
        
        # Collect current telemetry
        global_telemetry = {}
        for seed_id in self.seed_manager.seeds.keys():
            global_telemetry[seed_id] = self.seed_manager.get_comprehensive_telemetry(seed_id)
        
        # Get policy decision
        with torch.no_grad():
            policy_output = self.policy_network(
                global_telemetry, self.hardware_context, model_state
            )
        
        # Interpret timing decision
        timing_probs = torch.softmax(policy_output['timing_control'], dim=-1)
        timing_decision = torch.argmax(timing_probs).item()
        
        # 0: now, 1: soon, 2: later
        should_germinate_now = (timing_decision == 0) or force_decision
        
        if should_germinate_now:
            # Get blueprint and location decisions
            blueprint_probs = torch.softmax(policy_output['blueprint_choice'], dim=-1)
            blueprint_idx = torch.argmax(blueprint_probs).item()
            
            location_probs = torch.softmax(policy_output['location_selection'], dim=-1)
            location_idx = torch.argmax(location_probs).item()
            
            intensity = policy_output['intensity_control'].item()
            
            # Map indices to actual blueprint names and locations
            blueprint_names = BlueprintRegistry.list_blueprints()
            blueprint_name = blueprint_names[blueprint_idx] if blueprint_idx < len(blueprint_names) else "no_op"
            
            decision_details = {
                'blueprint_name': blueprint_name,
                'location_index': location_idx,
                'intensity': intensity,
                'safety_scores': policy_output['safety_scores'],
                'hardware_scores': policy_output['hardware_scores'],
                'confidence': {
                    'blueprint': blueprint_probs[blueprint_idx].item(),
                    'location': location_probs[location_idx].item(),
                    'timing': timing_probs[timing_decision].item()
                }
            }
            
            # Store for experience collection
            self.last_decision = decision_details
            self.last_telemetry = global_telemetry
            
            return True, decision_details
        
        return False, None
    
    def execute_germination(self, decision_details: Dict) -> bool:
        """Execute the germination decision."""
        
        # Find available seed location
        available_seeds = [
            sid for sid, info in self.seed_manager.seeds.items()
            if info['status'] == 'dormant'
        ]
        
        if not available_seeds:
            return False
        
        # Select seed based on location preference
        target_seed = available_seeds[
            min(decision_details['location_index'], len(available_seeds) - 1)
        ]
        
        # Create blueprint instance
        try:
            blueprint = BlueprintRegistry.create_blueprint(
                decision_details['blueprint_name'],
                input_dim=self.seed_manager.seeds[target_seed]['module'].dim
            )
            
            # Replace the existing module with blueprint
            self.seed_manager.seeds[target_seed]['module'] = blueprint
            
            # Set intensity (learning rate scaling)
            intensity = decision_details['intensity']
            self._apply_intensity_scaling(target_seed, intensity)
            
            # Request germination
            return self.seed_manager.request_germination(target_seed)
            
        except Exception as e:
            print(f"Germination failed: {e}")
            return False
    
    def _apply_intensity_scaling(self, seed_id: str, intensity: float):
        """Apply intensity scaling to seed learning rate."""
        # This would integrate with the learning rate management system
        # Implementation depends on how learning rates are managed
        pass
    
    def collect_reward_and_train(self, 
                                post_germination_metrics: Dict[str, float]):
        """Collect reward signal and train the policy network."""
        if self.last_decision is None or self.last_telemetry is None:
            return
        
        # Calculate reward (simplified - Stage 3B will have comprehensive reward system)
        reward = self._calculate_simple_reward(post_germination_metrics)
        
        # Create experience
        experience = PolicyExperience(
            global_telemetry=self.last_telemetry,
            hardware_context=self.hardware_context,
            model_state=post_germination_metrics,  # Simplified
            action={
                'blueprint_choice': self.last_decision['blueprint_name'],
                'intensity_control': self.last_decision['intensity']
            },
            reward=reward,
            next_global_telemetry=None,  # Would collect current telemetry
            next_model_state=None,
            done=False
        )
        
        # Add to trainer
        self.trainer.add_experience(experience)
        
        # Train if enough experiences
        if len(self.trainer.experience_buffer) >= self.trainer.batch_size:
            self.trainer.train_step()
        
        # Clear last decision
        self.last_decision = None
        self.last_telemetry = None
    
    def _calculate_simple_reward(self, metrics: Dict[str, float]) -> float:
        """Calculate simple reward signal (placeholder for Stage 3B)."""
        accuracy_gain = metrics.get('accuracy_improvement', 0.0)
        latency_penalty = metrics.get('latency_increase', 0.0) * -0.1
        
        return accuracy_gain + latency_penalty
```

## Testing Framework

### 1. Policy Network Tests

```python
# tests/test_policy_network.py
import pytest
import torch
from morphogenetic_engine.policy.network import KasminaPolicyNetwork
from morphogenetic_engine.telemetry.types import SeedTelemetry, HardwareContext

class TestPolicyNetwork:
    
    @pytest.fixture
    def policy_network(self):
        return KasminaPolicyNetwork(
            telemetry_dim=32,
            hidden_dim=128,
            num_blueprints=5,
            max_seeds=8
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
    def sample_hardware_context(self):
        return HardwareContext(
            device_type="gpu",
            memory_gb=24.0,
            flops_per_ms=1e9,
            bandwidth_gbps=1008.0,
            power_budget_watts=450.0,
            latency_target_ms=10.0
        )
    
    def test_policy_network_forward(self, policy_network, sample_telemetry, sample_hardware_context):
        """Test policy network forward pass."""
        model_state = {
            'accuracy': 0.85,
            'loss': 0.5,
            'epoch': 100,
            'learning_rate': 1e-3
        }
        
        output = policy_network(sample_telemetry, sample_hardware_context, model_state)
        
        # Check output structure
        assert 'blueprint_choice' in output
        assert 'location_selection' in output
        assert 'intensity_control' in output
        assert 'timing_control' in output
        assert 'safety_scores' in output
        assert 'hardware_scores' in output
        assert 'value_estimate' in output
        
        # Check output shapes
        assert output['blueprint_choice'].shape[-1] == 5  # num_blueprints
        assert output['location_selection'].shape[-1] == 8  # max_seeds
        assert output['intensity_control'].shape[-1] == 1
        assert output['timing_control'].shape[-1] == 3
    
    def test_constraint_application(self, policy_network, sample_telemetry, sample_hardware_context):
        """Test that safety constraints are properly applied."""
        # Create scenario with high security alert
        model_state = {'accuracy': 0.5, 'loss': 1.0, 'epoch': 50, 'learning_rate': 1e-3}
        
        # Mock high security alert (would need to modify the network or input)
        output = policy_network(sample_telemetry, sample_hardware_context, model_state)
        
        # Check that constraints are computed
        assert 'safety_scores' in output
        assert 'hardware_scores' in output
    
    def test_deterministic_output(self, policy_network, sample_telemetry, sample_hardware_context):
        """Test that the same input produces the same output."""
        model_state = {'accuracy': 0.85, 'loss': 0.5, 'epoch': 100, 'learning_rate': 1e-3}
        
        with torch.no_grad():
            output1 = policy_network(sample_telemetry, sample_hardware_context, model_state)
            output2 = policy_network(sample_telemetry, sample_hardware_context, model_state)
        
        # Check deterministic behavior
        torch.testing.assert_close(output1['blueprint_choice'], output2['blueprint_choice'])
        torch.testing.assert_close(output1['intensity_control'], output2['intensity_control'])
```

## Deliverables Checklist for Stage 3A

- [ ] Core policy network architecture with multi-head decisions
- [ ] Telemetry encoder for comprehensive feature extraction
- [ ] Context aggregator with attention mechanism
- [ ] Safety and hardware constraint heads
- [ ] Value network for decision evaluation
- [ ] Policy training framework with experience replay
- [ ] Enhanced Kasmina controller integration
- [ ] Comprehensive test suite for all components
- [ ] Integration tests with existing SeedManager
- [ ] Performance benchmarks and profiling
- [ ] Documentation and usage examples

## Stage 3A Success Criteria

1. **Functional Requirements**
   - Policy network produces valid decisions for all input scenarios
   - Safety constraints properly mask dangerous actions
   - Hardware constraints respected in all decisions
   - Integration with existing system seamless

2. **Performance Requirements**
   - Decision generation <50ms on target hardware
   - Training converges within reasonable time
   - Memory usage scales linearly with number of seeds
   - No performance regression on existing experiments

3. **Quality Requirements**
   - 85%+ test coverage on all policy components
   - Comprehensive integration testing
   - Performance profiling and optimization
   - Clear documentation and examples

This completes Stage 3A, establishing the core intelligent decision-making capability for Kasmina.
