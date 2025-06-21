# Technical Implementation Specification: Morphogenetic Seed Lifecycle

**Document Type:** Technical Specification  
**Status:** Ready for Implementation  
**Target Version:** 2.0  
**Dependencies:** Implementation Plan Reviewed  

---

## API Specifications

### 1. Updated SeedState Enum

```python
# morphogenetic_engine/events.py
class SeedState(Enum):
    """Enumeration of possible states for a morphogenetic seed."""
    
    DORMANT = "dormant"          # Seed monitors activations, identity pass-through
    GERMINATED = "germinated"    # Seed initialized, waiting in training queue
    TRAINING = "training"        # Child learns reconstruction via MSE loss
    GRAFTING = "grafting"        # Alpha blending: (1-α)*x + α*child(x)
    STABILIZATION = "stabilization"  # Hold α=1.0, freeze parameters
    FINE_TUNING = "fine_tuning"     # Child learns task loss, α=1.0
    FOSSILIZED = "fossilized"    # Permanently integrated
    CULLED = "culled"           # Failed validation, under embargo
```

### 2. Enhanced BlendingConfig (→ GraftingConfig)

```python
# morphogenetic_engine/core.py
@dataclass(frozen=True)
class GraftingConfig:
    """Configuration for grafting strategies and lifecycle phases."""
    
    # Grafting phase configuration
    fixed_steps: int = 30
    high_drift_threshold: float = 0.12
    low_health_threshold: float = 1e-3
    performance_loss_factor: float = 0.8
    grad_norm_lower: float = 0.1
    grad_norm_upper: float = 1.0
    
    # NEW: Stabilization phase configuration
    stabilization_epochs: int = 5
    
    # NEW: Fine-tuning phase configuration
    fine_tuning_max_epochs: int = 20
    fine_tuning_patience: int = 5
    fine_tuning_min_improvement: float = 0.01
```

### 3. Extended SentinelSeed Interface

```python
# morphogenetic_engine/components.py
class SentinelSeed(nn.Module):
    """Enhanced SentinelSeed with new lifecycle phases."""
    
    def __init__(self, ...):
        # ... existing parameters ...
        
        # NEW: Fine-tuning infrastructure
        self.label_buffer: list[torch.Tensor] = []
        self.task_loss_history: list[float] = []
        self.stabilization_counter: int = 0
        
    def train_child_step_task_loss(
        self, 
        inputs: torch.Tensor, 
        labels: torch.Tensor, 
        criterion: nn.Module,
        epoch: int | None = None
    ) -> None:
        """Train child using task loss instead of reconstruction loss."""
        
    def append_to_buffer_with_labels(
        self, 
        x: torch.Tensor, 
        labels: torch.Tensor
    ) -> None:
        """Store input-label pairs for fine-tuning phase."""
        
    def enter_stabilization_phase(self) -> None:
        """Initialize stabilization phase with frozen parameters."""
        
    def enter_fine_tuning_phase(self) -> None:
        """Initialize fine-tuning phase with task-aligned training."""
        
    def assess_fine_tuning_performance(self) -> tuple[bool, dict]:
        """Assess whether fine-tuning improves global performance."""
```

---

## State Transition Logic

### 1. Forward Pass Behavior by State

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """State-dependent forward pass behavior."""
    
    match self.state:
        case SeedState.DORMANT.value:
            # Buffer activations, pass through unchanged
            self.seed_manager.append_to_buffer(self.seed_id, x)
            return x
            
        case SeedState.GERMINATED.value | SeedState.TRAINING.value:
            # Pass through unchanged, training happens separately
            return x
            
        case SeedState.GRAFTING.value:
            # Alpha blending between original and child paths
            child_out = self.child(x)
            output = (1 - self.alpha) * x + self.alpha * child_out
            self._monitor_drift(x, output)
            return output
            
        case SeedState.STABILIZATION.value | SeedState.FINE_TUNING.value:
            # Full residual connection
            child_out = self.child(x)
            output = x + child_out
            self._monitor_drift(x, output)
            return output
            
        case SeedState.FOSSILIZED.value:
            # Permanent residual connection
            child_out = self.child(x)
            return x + child_out
            
        case SeedState.CULLED.value:
            # Pass through unchanged, seed is disabled
            return x
            
        case _:
            # Fallback for unknown states
            return x
```

### 2. State Transition Rules

```python
# In KasminaMicro.assess_and_update_seeds()
def _handle_state_transitions(self, seed_id: tuple[int, int], epoch: int) -> None:
    """Handle state transitions for a specific seed."""
    
    info = self.seed_manager.seeds[seed_id]
    module = info["module"]
    
    match module.state:
        case SeedState.TRAINING.value:
            if self._training_soft_test_passed(seed_id):
                self._transition_to_grafting(seed_id, epoch)
            elif self._training_failed(seed_id):
                self._transition_to_culled(seed_id, epoch)
                
        case SeedState.GRAFTING.value:
            if module.alpha >= 1.0:
                self._transition_to_stabilization(seed_id, epoch)
                
        case SeedState.STABILIZATION.value:
            if self._stabilization_complete(seed_id):
                self._transition_to_fine_tuning(seed_id, epoch)
                
        case SeedState.FINE_TUNING.value:
            performance_improved, metrics = module.assess_fine_tuning_performance()
            if performance_improved:
                self._transition_to_fossilized(seed_id, epoch)
            elif self._fine_tuning_failed(seed_id):
                self._transition_to_culled(seed_id, epoch)
```

---

## Fine-Tuning Infrastructure

### 1. Label Management System

```python
class LabeledActivationBuffer:
    """Manages input-label pairs for fine-tuning."""
    
    def __init__(self, max_size: int = 1000):
        self.inputs: list[torch.Tensor] = []
        self.labels: list[torch.Tensor] = []
        self.max_size = max_size
        
    def append(self, x: torch.Tensor, labels: torch.Tensor) -> None:
        """Add input-label pair to buffer."""
        self.inputs.append(x.detach().clone())
        self.labels.append(labels.detach().clone())
        
        # Maintain buffer size
        if len(self.inputs) > self.max_size:
            self.inputs.pop(0)
            self.labels.pop(0)
            
    def sample_batch(self, batch_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of input-label pairs."""
        if len(self.inputs) < batch_size:
            batch_size = len(self.inputs)
            
        indices = random.sample(range(len(self.inputs)), batch_size)
        
        batch_inputs = torch.cat([self.inputs[i] for i in indices], dim=0)
        batch_labels = torch.cat([self.labels[i] for i in indices], dim=0)
        
        return batch_inputs, batch_labels
```

### 2. Task Loss Training Method

```python
def train_child_step_task_loss(
    self, 
    inputs: torch.Tensor, 
    labels: torch.Tensor, 
    criterion: nn.Module,
    epoch: int | None = None
) -> float:
    """Train child network using task loss."""
    
    if self.state != SeedState.FINE_TUNING.value:
        raise ValueError(f"Task loss training only available in FINE_TUNING state, current: {self.state}")
        
    if self.child_optim is None:
        raise RuntimeError("Child optimizer not initialized.")
        
    # Forward pass through child
    self.child_optim.zero_grad(set_to_none=True)
    
    # Get child's prediction for these inputs
    child_prediction = self.child(inputs)
    
    # Compute task loss (e.g., CrossEntropy for classification)
    task_loss = criterion(child_prediction, labels)
    
    # Backward pass and optimization
    task_loss.backward()
    
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(self.child.parameters(), max_norm=1.0)
    
    self.child_optim.step()
    
    # Update metrics
    loss_value = task_loss.item()
    self.task_loss_history.append(loss_value)
    
    seed_info = self.seed_manager.seeds[self.seed_id]
    seed_info["current_task_loss"] = loss_value
    seed_info["fine_tuning_steps"] = seed_info.get("fine_tuning_steps", 0) + 1
    
    return loss_value
```

### 3. Performance Assessment Framework

```python
def assess_fine_tuning_performance(self) -> tuple[bool, dict]:
    """Assess whether fine-tuning is improving global performance."""
    
    if self.state != SeedState.FINE_TUNING.value:
        return False, {}
        
    seed_info = self.seed_manager.seeds[self.seed_id]
    
    # Get baseline performance (before fine-tuning started)
    baseline_loss = seed_info.get("fine_tuning_baseline_loss")
    baseline_acc = seed_info.get("fine_tuning_baseline_acc")
    
    # Get current global performance
    current_loss = seed_info.get("current_global_loss", float('inf'))
    current_acc = seed_info.get("current_global_acc", 0.0)
    
    # Calculate improvements
    loss_improvement = (baseline_loss - current_loss) if baseline_loss else 0.0
    acc_improvement = (current_acc - baseline_acc) if baseline_acc else 0.0
    
    # Decision criteria
    config = self.parent_net_ref().blend_cfg
    min_improvement = config.fine_tuning_min_improvement
    
    improved = (
        loss_improvement > min_improvement or 
        acc_improvement > min_improvement
    )
    
    metrics = {
        "loss_improvement": loss_improvement,
        "acc_improvement": acc_improvement,
        "baseline_loss": baseline_loss,
        "baseline_acc": baseline_acc,
        "current_loss": current_loss,
        "current_acc": current_acc,
        "fine_tuning_steps": seed_info.get("fine_tuning_steps", 0)
    }
    
    return improved, metrics
```

---

## Grafting Strategy Specifications

### 1. Enhanced Strategy Interface

```python
# morphogenetic_engine/grafting.py (renamed from blending.py)
class GraftingStrategy(ABC):
    """Abstract base class for all grafting strategies."""
    
    def __init__(self, seed: SentinelSeed, config: GraftingConfig):
        self.seed = seed
        self.config = config
        
    @abstractmethod
    def update(self) -> float:
        """Calculate next alpha value for grafting phase."""
        raise NotImplementedError
        
    def can_be_used_for_seed(self, seed: SentinelSeed) -> bool:
        """Check if this strategy is suitable for the given seed."""
        return True  # Default: all strategies can be used
```

### 2. Strategy Selection Algorithm

```python
def _choose_graft_strategy(self, seed_id: tuple[int, int]) -> str:
    """Intelligent strategy selection based on real-time telemetry."""
    
    info = self.seed_manager.seeds[seed_id]
    module = info["module"]
    telemetry = info.get("telemetry", {})
    
    # Collect decision factors
    drift = telemetry.get("drift", 0.0)
    health_signal = module.get_health_signal()
    grad_norm = info.get("avg_grad_norm", 0.0)
    training_stability = self._assess_training_stability(seed_id)
    
    cfg = self.blend_cfg
    
    # Decision tree for strategy selection
    if drift > cfg.high_drift_threshold:
        strategy = "DRIFT_CONTROLLED"
        reason = f"High drift detected: {drift:.4f}"
        
    elif grad_norm > cfg.grad_norm_upper or grad_norm < cfg.grad_norm_lower:
        strategy = "GRAD_NORM_GATED"
        reason = f"Unstable gradients: {grad_norm:.4f}"
        
    elif health_signal < cfg.low_health_threshold:
        strategy = "FIXED_RAMP"
        reason = f"Low health signal: {health_signal:.6f}"
        
    elif training_stability < 0.8:  # Less than 80% stable
        strategy = "DRIFT_CONTROLLED"
        reason = f"Training instability: {training_stability:.2f}"
        
    else:
        strategy = "FIXED_RAMP"
        reason = "Stable conditions, using reliable default"
    
    # Log the decision with telemetry
    self._log_strategy_selection(seed_id, strategy, reason, {
        "drift": drift,
        "health_signal": health_signal,
        "grad_norm": grad_norm,
        "training_stability": training_stability
    })
    
    return strategy

def _assess_training_stability(self, seed_id: tuple[int, int]) -> float:
    """Assess training stability based on recent loss history."""
    info = self.seed_manager.seeds[seed_id]
    module = info["module"]
    
    if len(module.loss_history) < 10:
        return 1.0  # Assume stable if not enough data
        
    recent_losses = module.loss_history[-10:]
    loss_variance = np.var(recent_losses)
    loss_mean = np.mean(recent_losses)
    
    # Coefficient of variation as stability metric
    if loss_mean > 1e-6:
        cv = np.sqrt(loss_variance) / loss_mean
        stability = max(0.0, 1.0 - cv)  # Lower CV = higher stability
    else:
        stability = 1.0
        
    return stability
```

---

## Integration Points

### 1. Training Loop Integration

```python
# In training.py - Enhanced per-step updates
def _perform_per_step_seed_updates(
    seed_manager: "SeedManager", 
    device: torch.device, 
    epoch: int | None,
    current_batch_labels: torch.Tensor | None = None,
    criterion: nn.Module | None = None
):
    """Enhanced seed updates with fine-tuning support."""
    
    for seed_id, info in seed_manager.seeds.items():
        module = info["module"]
        
        # Existing reconstruction training
        if module.state == SeedState.TRAINING.value:
            batch_data = _get_seed_training_batch(info, device)
            if batch_data is not None:
                module.train_child_step(batch_data, epoch)
                
        # NEW: Task loss training for fine-tuning seeds
        elif module.state == SeedState.FINE_TUNING.value:
            if current_batch_labels is not None and criterion is not None:
                # Get labeled batch from seed's buffer
                try:
                    inputs, labels = module.labeled_buffer.sample_batch()
                    if inputs.numel() > 0:
                        module.train_child_step_task_loss(
                            inputs.to(device), 
                            labels.to(device), 
                            criterion, 
                            epoch
                        )
                except Exception as e:
                    logging.warning(f"Fine-tuning step failed for seed {seed_id}: {e}")
                    
        # Grafting strategy updates
        elif module.state == SeedState.GRAFTING.value:
            strategy_name = info.get("graft_strategy")
            if strategy_name:
                from .grafting import get_strategy
                strategy = get_strategy(strategy_name, module, seed_manager.graft_config)
                new_alpha = strategy.update()
                module.alpha = new_alpha
```

### 2. Data Flow Architecture

```python
# Enhanced BaseNet to support label propagation
def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
    """Forward pass with optional label propagation for fine-tuning."""
    
    x = self.input_activation(self.input_layer(x))
    
    for i in range(self.num_layers):
        x = self.activations[i](self.layers[i](x))
        
        # Apply seeds for this layer
        layer_seeds = self.get_seeds_for_layer(i)
        
        for seed in layer_seeds:
            # Propagate labels to seeds that need them
            if labels is not None and hasattr(seed, 'append_to_buffer_with_labels'):
                seed.append_to_buffer_with_labels(x, labels)
            
            x = seed(x)
            
    return self.out(x)
```

---

## Migration and Compatibility

### 1. Backward Compatibility Strategy

```python
# Compatibility layer for old state names
class StateCompatibility:
    """Handles migration from old to new state names."""
    
    OLD_TO_NEW = {
        "blending": "grafting",
        "shadowing": "stabilization", 
        "probationary": "fine_tuning"
    }
    
    NEW_TO_OLD = {v: k for k, v in OLD_TO_NEW.items()}
    
    @classmethod
    def migrate_state(cls, old_state: str) -> str:
        """Convert old state name to new state name."""
        return cls.OLD_TO_NEW.get(old_state, old_state)
        
    @classmethod
    def is_deprecated_state(cls, state: str) -> bool:
        """Check if state name is deprecated."""
        return state in cls.OLD_TO_NEW
```

### 2. Configuration Migration

```python
def migrate_config(old_config: dict) -> dict:
    """Migrate old configuration to new format."""
    
    new_config = old_config.copy()
    
    # Rename blending-related keys
    if "blend_steps" in new_config:
        new_config["fixed_steps"] = new_config.pop("blend_steps")
        warnings.warn("'blend_steps' renamed to 'fixed_steps'", DeprecationWarning)
        
    # Add new configuration defaults
    new_config.setdefault("stabilization_epochs", 5)
    new_config.setdefault("fine_tuning_max_epochs", 20)
    new_config.setdefault("fine_tuning_patience", 5)
    new_config.setdefault("fine_tuning_min_improvement", 0.01)
    
    return new_config
```

---

## Error Handling and Validation

### 1. State Transition Validation

```python
def validate_state_transition(
    current_state: SeedState, 
    new_state: SeedState
) -> bool:
    """Validate that a state transition is allowed."""
    
    valid_transitions = {
        SeedState.DORMANT: {SeedState.GERMINATED},
        SeedState.GERMINATED: {SeedState.TRAINING, SeedState.CULLED},
        SeedState.TRAINING: {SeedState.GRAFTING, SeedState.CULLED},
        SeedState.GRAFTING: {SeedState.STABILIZATION, SeedState.CULLED},
        SeedState.STABILIZATION: {SeedState.FINE_TUNING, SeedState.CULLED},
        SeedState.FINE_TUNING: {SeedState.FOSSILIZED, SeedState.CULLED},
        SeedState.FOSSILIZED: set(),  # Terminal state
        SeedState.CULLED: {SeedState.DORMANT}  # After embargo
    }
    
    return new_state in valid_transitions.get(current_state, set())
```

### 2. Runtime Validation

```python
def validate_seed_configuration(seed: SentinelSeed) -> list[str]:
    """Validate seed configuration and return list of issues."""
    
    issues = []
    
    # Check required attributes
    required_attrs = ["seed_id", "dim", "alpha", "state"]
    for attr in required_attrs:
        if not hasattr(seed, attr):
            issues.append(f"Missing required attribute: {attr}")
            
    # Validate alpha range
    if hasattr(seed, "alpha") and not (0.0 <= seed.alpha <= 1.0):
        issues.append(f"Alpha value {seed.alpha} outside valid range [0.0, 1.0]")
        
    # Validate state-specific requirements
    if seed.state == SeedState.FINE_TUNING.value:
        if not hasattr(seed, "labeled_buffer"):
            issues.append("FINE_TUNING state requires labeled_buffer")
            
    return issues
```

---

This technical specification provides the detailed implementation guidance needed to successfully execute the morphogenetic seed lifecycle migration. All APIs, state transitions, and integration points are clearly defined with concrete code examples.
