### Grafting Strategy Enhancement Plan (Revised)

This document details the end-to-end plan for enhancing the morphogenetic engine’s `GRAFTING` phase. It outlines the current foundational system and the future "Phase 2" goal of implementing a dynamic, heuristic-based strategy selector.

-----

### 1. Overview & Phased Rollout

1. **Phase 1: Foundation (Complete)**

      * **Goal**: Establish a robust, pluggable system for grafting strategies.
      * **Components**:
          * A new `morphogenetic_engine/blending.py` file containing a `BlendingStrategy` abstract base class and concrete strategy implementations.
          * A centralized `BlendingConfig` dataclass to manage all related hyperparameters.
          * A factory function, `get_strategy`, to select and instantiate strategies based on a configuration string.
          * New event types (`GRAFT_STRATEGY_CHOSEN`, `GRAFT_COMPLETED`) for logging and monitoring.

2. **Phase 2: Heuristic Selector (Future Work)**

      * **Goal**: Build an intelligent agent that dynamically selects the best grafting strategy at runtime based on live model telemetry.
      * **Components**:
          * Implementation of a `_choose_graft_strategy` method within the controller (`KasminaMicro`).
          * Enhanced telemetry collection to provide the necessary inputs for the heuristic rules.
          * Comprehensive testing and analytics to validate the effectiveness of the selector.

-----

### 2. Configuration & Event Contracts

*These components are part of the completed Phase 1 foundation.*

**Configuration (`morphogenetic_engine/core.py`):**
A `BlendingConfig` dataclass centralizes all thresholds and step counts, making experiments easily configurable without code edits.

**Event Contracts (`morphogenetic_engine/events.py`):**
New event types and `TypedDict` payloads provide detailed, structured logging for when a strategy is chosen and when a graft is completed. (Note: `BLEND_` prefixes should be renamed to `GRAFT_` for consistency).

-----

### 3. Foundational Strategies (Phase 1)

The following strategies have been implemented in `morphogenetic_engine/blending.py` and are selectable via configuration.

* `FixedRampBlending`: A simple, predictable, time-based ramp-up.
* `DriftControlledBlending`: A stability-gated strategy that pauses grafting if weight drift is too high.
* `GradNormGatedBlending`: A stability-gated strategy that pauses grafting if gradients are unstable.
* `PerformanceLinkedBlending`:
  * **Current Implementation**: Dynamically adjusts speed based on **reconstruction loss**.
  * **Critical Flaw**: As implemented, this strategy reinforces the autoencoder objective rather than the main task goal.
  * **Recommendation for Future**: This strategy should be disabled or redesigned to use a more meaningful metric, such as the global task validation loss, but only *after* the fine-tuning phase has begun.

-----

### 4. Heuristic Strategy Selection (Phase 2 Goal)

This section describes the future implementation of the smart selector.

**File:** `morphogenetic_engine/core.py` → `KasminaMicro._choose_graft_strategy`

The goal is to implement this function, which was outlined in the original plan. It will contain the rule-based logic to select a strategy name.

```python
# Future Implementation
def _choose_graft_strategy(self, seed_id: tuple[int, int]) -> str:
    """Dynamically selects the best grafting strategy based on live telemetry."""
    info = self.seed_manager.seeds[seed_id]
    telemetry = info.get("telemetry", {})
    drift = telemetry.get("drift", 0.0)
    # ... other telemetry checks ...
    cfg = self.blend_cfg

    # Heuristic rules:
    if drift > cfg.high_drift_threshold:
        # If the model is unstable, use a cautious strategy
        return "DRIFT_CONTROLLED"
    # Add more rules based on health, performance, etc.
    
    # Default to the safest option
    return "FIXED_RAMP"
```

This method will be called once when a seed enters the `GRAFTING` state. The returned strategy name will then be fed into the existing `get_strategy` factory.

-----

### 5. Grafting Execution & The New Lifecycle

This section is updated to reflect the new lifecycle: `GRAFTING` → `FINE_TUNING`.

**File:** `morphogenetic_engine/components.py`

1. **On Entering `GRAFTING`**:

      * The controller calls `_choose_graft_strategy` to get the strategy name for this seed.
      * A `GRAFT_STRATEGY_CHOSEN` event is logged with the chosen strategy and current telemetry.
      * Initial metrics like `graft_start_epoch` are stored.

2. **During `GRAFTING`**:

      * On each step, the chosen strategy's `update()` method is called to get the new `alpha`.

3. **On Completion of `GRAFTING` (`alpha >= 1.0`)**:

      * A `GRAFT_COMPLETED` event is logged with final metrics.
      * The seed's state transitions to **`PROBATIONARY & FINE_TUNING`**, not `SHADOWING`.

-----

### 6. Testing & Documentation

* **Phase 1 Testing (Complete)**: Unit tests for each individual strategy and the factory function are in place.
* **Phase 2 Testing (Future)**:
  * **Selector Logic Tests**: Mock telemetry data and assert that `_choose_graft_strategy` returns the correct strategy name.
  * **Integration Tests**: Simulate a full seed lifecycle to ensure the dynamically chosen strategy is correctly applied and that the transition to `PROBATIONARY & FINE_TUNING` occurs as expected.
* **Documentation**:
  * The `SEED_LIFECYCLE.md` document should be considered the single source of truth for the lifecycle stages.
  * A new document, `GRAFTING_STRATEGIES.md`, should detail the behavior of each implemented strategy and the future plan for the heuristic selector.
