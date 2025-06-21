### Project Breakdown into Implementation Stages

The architectural refactoring can be executed in five sequential stages. Each stage builds upon the last, ensuring a logical and manageable workflow from initial restructuring to final UI integration.

* **Stage 1: Foundational Restructuring of Core Components.** This initial stage focuses on creating the new class structure and redefining roles. We will rename the controller and create the new `KasminaHeuristics` class, establishing the correct ownership relationship with `SentinelSeed`.
* **Stage 2: Delegation of Lifecycle Logic.** With the new structure in place, this stage involves migrating the state machine and all post-germination lifecycle methods from `SentinelSeed` into the `KasminaHeuristics` class.
* **Stage 3: Re-wiring the Control and Data Flow.** This stage redefines how the components communicate. We will modify the `TamiyoController` to issue high-level commands and create a new update loop in the training script to drive the local Kasmina instances.
* **Stage 4: Implementing the Complete Post-Germination Lifecycle.** This stage adds the new features requested for the local lifecycle, including the `EVALUATING`, `FOSSILIZED`, and `REJECTED` states and their associated logic.
* **Stage 5: UI and Telemetry Integration.** The final stage is to update the `RichDashboard`, logging, and monitoring systems to accurately reflect the new architecture, states, and component names.

---

### Detailed Tasking Statement

#### **Stage 1: Foundational Restructuring of Core Components**

* **Objective:** To establish the new architectural pattern by renaming the global controller and creating the new local `KasminaHeuristics` class, ensuring the correct class relationships are in place before logic is moved.

* **Tasks:**
    1. **In `core.py`:**
        * Rename the class `KasminaMicro` to `TamiyoController`. Update all internal references and type hints accordingly.
    2. **In `components.py`:**
        * Create a new class file or section for `class KasminaHeuristics`.
        * Define its `__init__` method, which must accept a `seed: "SentinelSeed"` argument to establish a reference to its parent.
        * In the `SentinelSeed.__init__` method, instantiate the new local heuristics manager by adding the line: `self.kasmina = KasminaHeuristics(self, **kwargs)`.
    3. **In `training.py` and `main.py`:**
        * Update the type hints and instantiation calls to use the new `TamiyoController` name instead of `KasminaMicro`.

* **Acceptance Criteria:**
  * The `TamiyoController` class exists and has replaced `KasminaMicro`.
  * The `KasminaHeuristics` class exists.
  * Every `SentinelSeed` instance now creates and holds its own `KasminaHeuristics` instance.
  * The application will be broken at this stage, but the new class structure is correctly implemented.

---

#### **Stage 2: Delegation of Lifecycle Logic**

* **Objective:** To move all local lifecycle management logic out of `SentinelSeed` and into the new `KasminaHeuristics` class, making the `SentinelSeed` a simpler container.

* **Tasks:**
    1. **In `components.py` (`SentinelSeed` to `KasminaHeuristics`):**
        * Migrate the state-related attributes (`self.state`, `self.alpha`, `self.training_progress`, `self.step_counter`) from `SentinelSeed` to `KasminaHeuristics`.
        * Migrate the lifecycle management methods (`_set_state`, `train_child_step`, `update_blending`) from `SentinelSeed` into `KasminaHeuristics`.
        * Update these migrated methods to use `self.seed.child` to access the neural network and `self.seed.seed_manager` for logging.
        * The `SentinelSeed.forward()` method must be refactored to delegate its logic to a new `self.kasmina.forward(x)` method.

* **Acceptance Criteria:**
  * The `SentinelSeed` class no longer contains state machine logic (e.g., `self.state`, `self.alpha`).
  * The `KasminaHeuristics` class now contains all methods and attributes for managing the training, blending, and state of a single seed.
  * The `SentinelSeed.forward()` call is correctly delegated to its `KasminaHeuristics` instance.

---

#### **Stage 3: Re-wiring the Control and Data Flow**

* **Objective:** To update the high-level control flow so that `TamiyoController` issues commands and a new training loop drives the local `KasminaHeuristics` instances.

* **Tasks:**
    1. **In `core.py`:**
        * Modify `TamiyoController.step()`: The method should no longer directly alter a seed's state. Its sole purpose upon a successful trigger is to call `self.seed_manager.request_germination(seed_id, blueprint)`.
        * Modify `SeedManager.request_germination()`: This method should now find the correct `seed_module` and call a new method on its local heuristics instance, e.g., `seed_module.kasmina.start_germination(blueprint)`.
    2. **In `training.py`:**
        * Delete the `handle_seed_training` function.
        * Create a new function: `update_all_seeds_lifecycle(seed_manager: SeedManager, device: torch.device)`.
        * This new function must iterate through all registered seeds in the `seed_manager` and, for any seed not in a `DORMANT` or `FOSSILIZED` state, call a new `step(device)` method on its `kasmina` instance (e.g., `seed.kasmina.step(device)`).
        * In `train_epoch`, replace the old call to `handle_seed_training` with the new `update_all_seeds_lifecycle` call.

* **Acceptance Criteria:**
  * `TamiyoController` no longer implements lifecycle logic; it only sends a germination signal.
  * The training loop correctly calls the new `update_all_seeds_lifecycle` function each iteration.
  * This new function correctly steps through the state machine of each active seed via its `KasminaHeuristics.step()` method.

---

#### **Stage 4: Implementing the Complete Post-Germination Lifecycle**

* **Objective:** To implement the full, user-requested lifecycle within `KasminaHeuristics`, including the evaluation, fossilization, and rejection states.

* **Tasks:**
    1. **In a shared `events.py` or similar location:**
        * Update the `SeedState` enum to include `EVALUATING`, `FOSSILIZED`, and `REJECTED`.
    2. **In `components.py` (`KasminaHeuristics`):**
        * In the `_blend_step` method, change the final transition from `-> "active"` to `-> SeedState.EVALUATING`.
        * Implement the logic for the `EVALUATING` state within the `KasminaHeuristics.step()` method. This should use a simple step counter (`self.step_counter >= self.eval_steps`) as a placeholder for a real performance check.
        * Based on a placeholder `is_successful` flag, transition the state to either `SeedState.FOSSILIZED` or `SeedState.REJECTED`.
    3. **In `components.py` (`SentinelSeed`):**
        * Create the `fossilize()` method. This method must iterate through `self.child.parameters()` and set `p.requires_grad = False`.
        * Create or refactor the `reset_to_dormant()` method. This method must re-initialize the child network to a near-identity function and freeze its parameters. This method will be called when the `REJECTED` state is entered.

* **Acceptance Criteria:**
  * The `KasminaHeuristics` state machine correctly transitions through `TRAINING` -> `BLENDING` -> `EVALUATING`.
  * From `EVALUATING`, the state correctly transitions to either `FOSSILIZED` or `REJECTED`.
  * Entering the `FOSSILIZED` state successfully makes the child network's parameters non-trainable.
  * Entering the `REJECTED` state successfully resets the seed to its initial, dormant configuration.

---

#### **Stage 5: UI and Telemetry Integration**

* **Objective:** To update all user-facing components, including the dashboard, logs, and metrics, to accurately reflect the new architecture and provide clear feedback on the new lifecycle states.

* **Tasks:**
    1. **In `ui_dashboard.py`:**
        * Update the `SEED_EMOJI_MAP` dictionary to include entries for the new states: `EVALUATING` (e.g., 🔬), `FOSSILIZED` (e.g., 🦴), and `REJECTED` (e.g., 🔥).
        * In the `_setup_layout` method, rename the panel titles to reflect the new architecture. For example, change the `"kasima_panel"` title to `"Seed Lifecycle Monitor"` and ensure the `"tamiyo_panel"` is populated with telemetry from the `TamiyoController`.
        * Ensure the `_create_seeds_training_table` and `_create_seed_box_panel` correctly render the new states and their corresponding emojis.
    2. **In `core.py` and `components.py`:**
        * Review all calls to the `ExperimentLogger` to ensure state transitions are logged with the new `SeedState` enum values and clear, descriptive messages.
    3. **In `monitoring.py`:**
        * Update the `PrometheusMonitor`'s `state_map` to include integer representations for the new states.
        * Rename any metrics related to `kasmina_` to `tamiyo_` (e.g., `KASMINA_PLATEAU_COUNTER` should become `TAMIYO_PLATEAU_COUNTER`).

* **Acceptance Criteria:**
  * The `RichDashboard` correctly displays the new panel names.
  * The seed state grid and tables correctly display the new emojis and state names for `EVALUATING`, `FOSSILIZED`, and `REJECTED`.
  * Prometheus metrics are correctly named and report the new states.
  * All system logs accurately describe the actions of the `TamiyoController` and the local `KasminaHeuristics` instances.
