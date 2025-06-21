## Migration and Enhancement Plan: Morphogenetic Seed Lifecycle

This document outlines a phased implementation plan to migrate the Kaslite morphogenetic engine to a new, more robust seed lifecycle. The plan also includes an overhaul of the grafting strategies and fixes for critical bugs in the current implementation.

### **Executive Summary**

The primary goals of this migration are:

1. **Fix Critical Bugs:** Correct the forward pass logic to ensure the `alpha` grafting factor is properly applied, making the grafting strategies functional.
2. **Streamline the Seed Lifecycle:** Migrate from the complex `BLENDING` → `SHADOWING` → `PROBATIONARY` path to a more intuitive `GRAFTING` → `STABILIZATION` → `FINE-TUNING` lifecycle.
3. **Enhance Grafting Strategies:** Improve the intelligence of the grafting process by fixing flawed strategies and implementing a dynamic, heuristic-based strategy selector.

The plan is divided into five phases, starting with critical fixes and progressively refactoring the system.

### **Phase 1: Critical Bug Fixes & Foundational Refactoring**

This phase addresses the most critical issues preventing the current system from functioning as intended.

#### **1.1. Fix the Forward Pass Grafting Logic**

The bug where the `alpha` value is calculated but not used in the forward pass is the highest priority.

* **File to Modify:** `morphogenetic_engine/components.py`

* **Class to Modify:** `BaseNet`

* **Action:** Modify the `forward` method in `BaseNet`. The current implementation replaces the main layer's output with the seed's output. It needs to be updated to correctly graft the two.

    **Current (Bugged) Logic:**

    ```python
    # The output of the main layer is REPLACED by the seed output.
    # If multiple seeds, their outputs are averaged.
    x = layer_seeds[0](x)
    ```

    **Proposed (Corrected) Logic:**

    ```python
    # In BaseNet.forward method, within the layer loop

    # After applying the main layer and activation
    x_main = self.activations[i](self.layers[i](x))

    # Get the seed for this layer (assuming one for simplicity)
    seed = self.get_seeds_for_layer(i)[0]

    # The SentinelSeed's forward pass will handle its internal state logic.
    # We retrieve the alpha value from the seed itself.
    x_seed_out = seed(x_main) 

    # THE FIX: Graft the main output with the seed's output using the seed's alpha.
    # The SentinelSeed's forward pass should be modified to simply return its child's
    # output during GRAFTING, and the grafting happens here in the parent network.
    # However, a simpler approach is to let the seed handle its own grafting logic
    # and return the correctly grafted output, which it already does. The issue is that
    # the BaseNet was not using the main layer's output at all.

    # Corrected approach: The seed's forward pass should take the main activation
    # as input and return the grafted result.
    x = seed(x_main)
    ```

    To be more explicit, the `SentinelSeed.forward` method should be adjusted to correctly handle the grafting based on its state. The current `SentinelSeed.forward` logic is already mostly correct, the main bug is in `BaseNet.forward`.

#### **1.2. Update the SeedState Enum**

* **File to Modify:** `morphogenetic_engine/events.py`

* **Action:** Update the `SeedState` enum to reflect the new lifecycle.

    **Current Enum:**

    ```python
    class SeedState(Enum):
        DORMANT = "dormant"
        GERMINATED = "germinated"
        TRAINING = "training"
        BLENDING = "blending"
        SHADOWING = "shadowing"
        PROBATIONARY = "probationary"
        FOSSILIZED = "fossilized"
        CULLED = "culled"
    ```

    **New Enum:**

    ```python
    class SeedState(Enum):
        DORMANT = "dormant"
        GERMINATED = "germinated"
        TRAINING = "training"
        GRAFTING = "grafting" # Replaces BLENDING
        STABILIZATION = "stabilization" # New state
        FINE_TUNING = "fine_tuning" # New state
        FOSSILIZED = "fossilized"
        CULLED = "culled"
    ```

### **Phase 2: Lifecycle Migration**

This phase implements the new lifecycle states and transitions within the `SentinelSeed` and `KasminaMicro` controller.

* **File to Modify:** `morphogenetic_engine/components.py` (class `SentinelSeed`)
* **File to Modify:** `morphogenetic_engine/core.py` (class `KasminaMicro`)

#### **2.1. Implement the `GRAFTING` State**

* **Transition:** Occurs after a successful soft-test at the end of the `TRAINING` state.
* **Behavior:** The `alpha` graft factor ramps from 0 to 1. The `forward` pass now computes: `output = (1 - alpha) * x + alpha * child(x)`. This logic is already present in `SentinelSeed.forward` under the `BLENDING` state; it just needs to be triggered by the `GRAFTING` state instead.

#### **2.2. Implement the `STABILIZATION` State**

* **Transition:** Occurs after `GRAFTING` is complete (`alpha` ≈ 1.0).
* **Behavior:** Hold `alpha = 1.0` for a fixed number of epochs with no parameter updates. This allows residual grafting artifacts to settle. The `SentinelSeed`'s `forward` pass will simply be `output = x + child(x)`.

#### **2.3. Implement the `FINE-TUNING` State**

* **Transition:** Occurs after the `STABILIZATION` period.
* **Behavior:**
  * The `forward` pass remains `output = x + child(x)`.
  * The child network's training objective switches from reconstruction (MSE loss) to the model's primary task loss (e.g., Cross-Entropy). This is a critical change. The `train_child_step` method in `SentinelSeed` will need to be modified to accept the main task's labels and compute the task loss.
  * The `KasminaMicro` controller will monitor global performance metrics (`val_acc`, `val_loss`) during this phase to make the final `FOSSILIZED` or `CULLED` decision.

#### **2.4. Update `KasminaMicro` Controller**

* The controller's logic in `assess_and_update_seeds` must be updated to manage these new state transitions.

### **Phase 3: Grafting Strategy Overhaul**

This phase refactors the grafting strategy system to align with the new lifecycle and improve its intelligence.

* **File to Modify:** `morphogenetic_engine/grafting.py`
* **File to Modify:** `morphogenetic_engine/core.py` (class `KasminaMicro`)

#### **3.1. Rename "Blending" to "Grafting"**

* Rename `BlendingStrategy` to `GraftingStrategy`.
* Rename `blending.py` to `grafting.py`.
* Update all references, including the factory function `get_strategy`.

#### **3.2. Fix `PerformanceLinkedGrafting`**

* **The Flaw:** As noted in `grafting_strategy_implementation_plan.md`, this strategy currently links grafting speed to reconstruction loss, which is counterproductive.
* **The Fix:** This strategy should be disabled or redesigned. The recommended fix is to link it to the global validation loss *after* the `FINE-TUNING` phase has begun. For the `GRAFTING` phase, this strategy is not suitable. A new strategy could be created for the `FINE-TUNING` phase that adjusts the learning rate based on task performance.

#### **3.3. Implement Heuristic Strategy Selector**

* **Goal:** Implement the `_choose_graft_strategy` method in `KasminaMicro` as outlined in `blending_strategy_phase2_implementation_plan.md`.

* **Logic:** This method will use live telemetry (e.g., drift, health signal) to dynamically select the most appropriate grafting strategy (`FIXED_RAMP`, `DRIFT_CONTROLLED`, etc.) when a seed enters the `GRAFTING` state.

    ```python
    # In KasminaMicro
    def _choose_graft_strategy(self, seed_id: tuple[int, int]) -> str:
        """Dynamically selects the best grafting strategy based on live telemetry."""
        info = self.seed_manager.seeds[seed_id]
        telemetry = info.get("telemetry", {})
        drift = telemetry.get("drift", 0.0)
        cfg = self.graft_cfg

        # Heuristic rules:
        if drift > cfg.high_drift_threshold:
            # If the model is unstable, use a cautious strategy
            return "DRIFT_CONTROLLED"
        
        # Add more rules based on health, performance, etc.
        
        # Default to the safest option
        return "FIXED_RAMP"
    ```

### **Phase 4: Testing and Validation**

A comprehensive testing suite is required to validate the new lifecycle and bug fixes.

* **Unit Tests:**
  * Verify that the `BaseNet.forward` method correctly grafts outputs with `alpha`.
  * Test each state transition in `SentinelSeed`'s lifecycle.
  * Test each `GraftingStrategy` to ensure it calculates `alpha` correctly.
  * Test the `_choose_graft_strategy` heuristic selector with mocked telemetry data.
* **Integration Tests:**
  * Run a full experiment simulation to ensure a seed correctly transitions through the entire new lifecycle from `DORMANT` to `FOSSILIZED` or `CULLED`.
  * Verify that the `FINE-TUNING` phase correctly uses the main task loss.

### **Phase 5: Documentation Update**

Update all relevant project documentation to reflect these changes.

* **`README.md`:** Update the main README to describe the new lifecycle and features.
* **`docs/2. Architecture/kasima/SEED_LIFECYCLE.md`:** This document should be updated to be the new single source of truth for the lifecycle.
* **`docs/2. Architecture/kasima/grafting_strategy_implementation_plan.md`:** Rename to `grafting_strategy_implementation_plan.md` and update its contents.

By following this phased plan, the `kaslite` morphogenetic engine can be systematically and safely migrated to the new, more robust and functional architecture.
