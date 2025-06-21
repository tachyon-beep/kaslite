# Blending Strategy Enhancement Plan

This document details the end-to-end plan for refactoring the morphogenetic engine’s BLENDING phase into a dynamic, pluggable system—with robust logging, metrics collection, and UI support—while incorporating best practices for configuration, factory patterns, backward compatibility, and phased delivery.

---

## 1. Overview & Dependencies

1. **Phased Rollout**

   * **Phase 1:** Event types, config objects, fixed‐ramp strategy moved to `blending.py`, and core logging wiring.
   * **Phase 2:** Additional strategies, strategy‐factory injection, selector logic, metrics logging, dashboard panels.

2. **New Dependencies**

   * **`morphogenetic_engine/blending.py`** (new file)
   * **`morphogenetic_engine/events.py`** updates (new enums & payloads)
   * Optional: a lightweight **`BlendingConfig`** dataclass in `core.py`

3. **Backward Compatibility**

   * All existing experiments without new config or UI panels must continue working.
   * New code paths guarded by presence of config flags or logger availability.

---

## 2. Configuration Centralisation

**File:** `morphogenetic_engine/core.py`

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class BlendingConfig:
    fixed_steps: int = 30
    high_drift_threshold: float = 0.12
    low_health_threshold: float = 1e-3
    performance_loss_factor: float = 0.8
    grad_norm_lower: float = 0.1
    grad_norm_upper: float = 1.0

# In KasminaMicro.__init__, accept a blending_config: BlendingConfig
# and store as self.blend_cfg
```

All thresholds and step counts now live in one place, making experiments and tests configurable without code edits.

---

## 3. Event Contracts

**File:** `morphogenetic_engine/events.py`

1. **Extend `EventType`:**

   ```python
   class EventType(Enum):
       # … existing …
       BLEND_STRATEGY_CHOSEN = "blend_strategy_chosen"
       BLEND_COMPLETED        = "blend_completed"
   ```

2. **New Payload Definitions:**

   ```python
   class BlendStrategyChosenPayload(TypedDict):
       seed_id: tuple[int, int]
       epoch: int
       strategy_name: str
       telemetry: dict[str, float]  # e.g. {'health_signal':0.002,'drift':0.05,'baseline_loss':1.2,'current_loss':0.8}
       timestamp: float

   class BlendCompletedPayload(TypedDict):
       seed_id: tuple[int, int]
       epoch: int
       strategy_name: str
       duration_epochs: int
       initial_loss: float
       final_loss: float
       initial_drift: float
       final_drift: float
       timestamp: float
   ```

3. **Register Payloads** in the `EventPayload` union so downstream code can dispatch them cleanly.

---

## 4. Strategy Implementation

**File:** `morphogenetic_engine/blending.py` (new)

1. **Abstract Base:**

   ```python
   from abc import ABC, abstractmethod

   class BlendingStrategy(ABC):
       def __init__(self, cfg: BlendingConfig): 
           self.cfg = cfg

       @abstractmethod
       def update(self, seed: "SentinelSeed") -> float:
           """Return new alpha for this step/epoch."""
   ```

2. **Concrete Strategies:**

   ```python
   class FixedRampBlending(BlendingStrategy):
       def update(self, seed):
           return min(1.0, seed.alpha + 1/self.cfg.fixed_steps)

   class PerformanceLinkedBlending(BlendingStrategy):
       def update(self, seed):
           start = seed.seed_info["blend_initial_loss"]
           curr  = seed.validate_on_holdout()
           drop  = max(0.0, start-curr)
           return min(1.0, drop/(start+1e-12))

   class DriftControlledBlending(BlendingStrategy):
       def update(self, seed):
           info = seed.seed_info
           drifts = info.setdefault("drift_window", deque(maxlen=5))
           avg = sum(drifts)/len(drifts) if drifts else 0.0
           if avg < 0.5*self.cfg.high_drift_threshold:
               return min(1.0, seed.alpha + 2/self.cfg.fixed_steps)
           if avg < self.cfg.high_drift_threshold:
               return min(1.0, seed.alpha + 1/self.cfg.fixed_steps)
           return seed.alpha  # hold

   class GradNormGatedBlending(BlendingStrategy):
       def update(self, seed):
           gn = seed.seed_info.get("avg_grad_norm", 0.0)
           if self.cfg.grad_norm_lower < gn < self.cfg.grad_norm_upper:
               return min(1.0, seed.alpha + 1/self.cfg.fixed_steps)
           return seed.alpha
   ```

3. **Factory Function:**

   ```python
   STRATEGIES = {
       "FIXED_RAMP": FixedRampBlending,
       "PERFORMANCE_LINKED": PerformanceLinkedBlending,
       "DRIFT_CONTROLLED": DriftControlledBlending,
       "GRAD_NORM_GATED": GradNormGatedBlending,
   }

   def get_strategy(name: str, cfg: BlendingConfig) -> BlendingStrategy:
       cls = STRATEGIES.get(name, FixedRampBlending)
       return cls(cfg)
   ```

---

## 5. Strategy Selection

**File:** `morphogenetic_engine/core.py` → `KasminaMicro._choose_blend_strategy`

```python
def _choose_blend_strategy(self, seed_id):
    info = self.seed_manager.seeds[seed_id]
    module = info["module"]
    hs = module.get_health_signal()
    drift = info["telemetry"]["drift"]
    curr = info.get("current_loss", float('inf'))
    base = info.get("baseline_loss", float('inf'))
    cfg = self.blend_cfg  # from constructor

    if drift > cfg.high_drift_threshold:
        return "DRIFT_CONTROLLED"
    if hs < cfg.low_health_threshold:
        return "PERFORMANCE_LINKED"
    if curr < base * cfg.performance_loss_factor:
        return "GRAD_NORM_GATED"
    return "FIXED_RAMP"
```

**Log the choice immediately** in `assess_and_update_seeds`:

```python
if module.state == SeedState.BLENDING.value and "blend_strategy" not in info:
    strat = self._choose_blend_strategy(seed_id)
    info["blend_strategy"] = strat
    if self.logger:
        self.logger.log_seed_event_detailed(
            epoch=epoch,
            event_type=EventType.BLEND_STRATEGY_CHOSEN,
            payload=BlendStrategyChosenPayload(
                seed_id=seed_id, epoch=epoch,
                strategy_name=strat,
                telemetry={
                    "health_signal": hs,
                    "drift": drift,
                    "baseline_loss": base,
                    "current_loss": curr
                },
                timestamp=time.time(),
            ),
        )
```

---

## 6. Blending Execution & Metrics

**File:** `morphogenetic_engine/components.py`

1. **On Entering BLENDING** (e.g. in `_handle_training_transition` just before `_set_state(BLENDING)`):

   * Store `seed_info["blend_start_epoch"] = epoch`
   * Store `seed_info["blend_initial_loss"]  = seed.validate_on_holdout()`
   * Store `seed_info["blend_initial_drift"] = compute current drift via cosine‐sim code`

2. **Modify `update_blending`:**

   ```python
   def update_blending(self, epoch):
       if self.state != SeedState.BLENDING.value: return
       info = self.seed_manager.seeds[self.seed_id]
       strat_name = info["blend_strategy"]
       strategy = get_strategy(strat_name, self.seed_manager.blend_cfg)
       self.alpha = strategy.update(self)
       info["alpha"] = self.alpha
   ```

3. **On Completion** (`_handle_blending_transition`, when `alpha >= 0.99`):

   * Compute:

     ```python
     start_e = info.pop("blend_start_epoch")
     start_l = info.pop("blend_initial_loss")
     start_d = info.pop("blend_initial_drift")
     end_l   = self.validate_on_holdout()
     end_d   = compute cosine‐sim drift now
     epochs  = epoch - start_e + 1
     ```

   * Log `BLEND_COMPLETED`:

     ```python
     if self.seed_manager.logger:
         self.seed_manager.logger.log_seed_event_detailed(
             epoch=epoch,
             event_type=EventType.BLEND_COMPLETED,
             payload=BlendCompletedPayload(
                 seed_id=self.seed_id, epoch=epoch,
                 strategy_name=strat_name,
                 duration_epochs=epochs,
                 initial_loss=start_l,
                 final_loss=end_l,
                 initial_drift=start_d,
                 final_drift=end_d,
                 timestamp=time.time(),
             ),
         )
     ```

   * Then call `_set_state(SeedState.SHADOWING, epoch)`.

---

## 7. Dashboard & Analytics

1. **Seed‐State Grid** (`ui_dashboard.py`):

   * Add columns **“Blend Strategy”** and **“Blend α”** to the per‐seed table. Pull from `seed_states[layer][index].metrics["blend_strategy"]` and `.metrics["alpha"]`.

2. **New Analytics Panel** (`ui/panels.py`):

   * Implement `create_blending_analytics_panel()` that reads **both** the event stream **and** the in‐memory aggregates in an `EventManager` or new `BlendingStatsManager`.
   * Show per‐strategy summaries (count, avg duration, avg loss improvement, success/failure).

3. **Event Aggregator** (`ui/events.py` or `logger.py`):

   * In the code that dispatches events to dashboards, add handlers for `BLEND_COMPLETED` to accumulate stats in a `defaultdict(list)` keyed by `strategy_name`.

4. **Layout Update** (`dashboard.py`):

   * Replace or augment the existing event log panel with the blending analytics panel in the TUI/GUI layout.

---

## 8. Testing

**File:** `tests/test_blending_strategies.py`

1. **Unit Tests for Strategies:**

   * Instantiate each `BlendingStrategy` with a `BlendingConfig` and a dummy `SentinelSeed` (with fake `alpha`, `validate_on_holdout`, etc.), assert correct `alpha` updates.

2. **Selector Logic Tests:**

   * Mock seed\_info with different drift/health/loss values, call `_choose_blend_strategy`, assert correct strategy string.

3. **Integration Tests:**

   * Simulate a seed going from TRAINING→BLENDING, assert that:

     * `BLEND_STRATEGY_CHOSEN` event is emitted with expected telemetry.
     * After “n” calls to `update_blending`, a `BLEND_COMPLETED` event appears with accurate metrics.

4. **UI Rendering Tests (optional):**

   * Use a dummy `EventManager` with pre‐recorded `BlendCompletedPayload`s, call `create_blending_analytics_panel`, verify table contents.

---

## 9. Documentation & Next Steps

* **README / CONTRIBUTING:** Document the new `blending.py`, the `BlendingConfig`, event types, and how to hook into the analytics panel.
* **Example Config:** Provide a YAML snippet showing how to override `blend_cfg` thresholds.
* **CI:** Update pipelines to install any new dependencies (none expected), and run the new test suite.

With this phased, config-driven, factory-based design—and full logging/analytics—you’ll have a highly adaptive, transparent blending system ready for any morphogenetic experiment.
