### The Authoritative Guide to the Morphogenetic Seed Lifecycle

This document outlines the step-by-step journey of a `SentinelSeed` from its creation as a dormant placeholder to its final state as either a permanent part of the network or a culled failure.

**Lifecycle Path:**
`DORMANT` → `GERMINATED` → `TRAINING` → `BLENDING` → `SHADOWING` → `PROBATIONARY` → (`FOSSILIZED` or `CULLED`)

---

#### 1. State: DORMANT (⚪)

* **Trigger:** This is the initial state of a `SentinelSeed` when the parent `BaseNet` is first created, or after a previously `CULLED` seed's embargo period has expired.
* **Process:**
  * The seed is inert and its `forward` pass acts as an identity function, simply returning its input `x` unmodified.
  * It is in a passive monitoring mode, collecting all activation data that passes through it and storing it in its buffer. This buffer will later be used for training.
  * The `TamiyoController` continuously polls the seed's "health signal" (activation variance) to assess it as a potential candidate for germination.
* **Next State: `GERMINATED`**
  * This transition is triggered when the controller detects a global performance plateau and selects this seed as the best candidate to fix the bottleneck. It then calls `request_germination` via the `SeedManager`.

#### 2. State: GERMINATED (🌱)

* **Trigger:** A successful `request_germination` call from the controller.
* **Process:**
  * This state functions as a **training queue or "parking lot."** This is critical because `Tamiyo` might "burst" several seeds at once, and this queue prevents them from all training simultaneously and interfering with each other.
  * The seed's child network is initialized with proper weights, and its parameters are made trainable (`requires_grad = True`).
  * The seed waits in this state until the `SeedManager` confirms that no other seed is currently being trained.
* **Next State: `TRAINING`**
  * When a training "slot" is free, the `SeedManager`'s `start_training_next_seed` method promotes the oldest seed from the `GERMINATED` queue into the `TRAINING` state.

#### 3. State: TRAINING (🟢)

* **Trigger:** Promotion from the `GERMINATED` state by the `SeedManager`.
* **Process:**
  * The main training loop, via `handle_seed_training`, now continuously feeds batches of buffered data to the `TRAINING` seed's `train_child_step` method.
  * The seed's child network learns to reconstruct the data, and its `training_progress` increases.
  * During this phase, the seed's main `forward` pass remains an identity function (`return x`). This is crucial as it isolates the training process, preventing the partially-trained child from disrupting the parent network's overall function.
* **Next State: `BLENDING`**
  * This transition occurs automatically once `training_progress` surpasses its predefined threshold, indicating the local training is sufficiently complete.

#### 4. State: BLENDING (🟡)

* **Trigger:** The `training_progress` threshold is met in the `TRAINING` state.
* **Process:**
  * The `update_blending` method is now called on each step, gradually increasing the `alpha` parameter from 0 to 1.
  * The seed's `forward` pass now becomes active, smoothly mixing the original input with the child network's output: `output = (1 - alpha) * x + alpha * child_out`. This ensures a gradual, non-disruptive introduction of the new functionality.
* **Next State: `SHADOWING`**
  * Occurs when `alpha` reaches `~1.0`, signifying the blending is complete.

#### 5. State: SHADOWING (👻) - Stage 1 Validation

* **Trigger:** Completion of the `BLENDING` phase.
* **Process:**
  * This is the first of two validation gates, designed to verify the module's internal stability.
  * The seed's `forward` pass becomes **inert**; it returns the input `x` unmodified, protecting the parent network from any potential instability.
  * In the background, the system can perform stability checks on the child network using live data.
* **Next State: `PROBATIONARY` or `CULLED`**
  * If the seed remains stable, it proceeds. If not, it is culled.

#### 6. State: PROBATIONARY (🧑‍⚖️) - Stage 2 Validation

* **Trigger:** After successfully passing the shadowing phase.
* **Process:**
  * This is the second validation gate, testing for systemic impact.
  * The seed's `forward` pass is now **fully live**, returning `x + child_out`. The graft is active and influencing the entire network's behavior.
  * The controller monitors global performance metrics (`val_acc`, `val_loss`) to ensure the change is not detrimental to the system as a whole.
* **Next State: `FOSSILIZED` or `CULLED`**
  * The probationary period ends. The `_evaluate_and_complete` method is called to make the final judgment.

#### 7. Final State: FOSSILIZED (🦴) - Success

* **Trigger:** The module successfully passes its probation by demonstrating a sufficient performance improvement.
* **Process:**
  * This is a terminal success state.
  * The `_fossilize_seed` method is called, which instructs the `BaseNet` to **physically replace** the `SentinelSeed` with a permanent, frozen `GraftedChild` module. The new network graph is then saved to disk.
* **Next State:** None (Lifecycle Complete).

#### 8. Final State: CULLED (🥀 / 🔒) - Failure and Embargo

* **Trigger:** The module fails its validation in either the `SHADOWING` or `PROBATIONARY` state.
* **Process:**
  * This is a terminal failure state that also functions as a **timed embargo** to prevent thrashing.
  * The `_cull_seed` method sets the seed's state to `CULLED` and records the current epoch (`culling_epoch`) in the `SeedManager`.
  * The failed `SentinelSeed` module remains in the network—inactive and frozen—but its slot is now under embargo. The `TamiyoController` is blocked from selecting this slot for germination.
* **Next State: `DORMANT`**
  * The lifecycle for this specific seed *instance* is over. However, the architectural *slot* will become available again.
  * An epoch-level process (`handle_cooldowns`) monitors all culled seeds. Once the embargo duration has passed (`current_epoch > culling_epoch + embargo_duration`), this process will reset the seed module to its original `DORMANT` state, making it eligible for a new germination attempt in the future.
