### Guide to Morphogenetic Grafting Strategies

This document provides a detailed overview of the pluggable grafting strategies used within the morphogenetic engine. The choice of strategy dictates how a new seed's influence (`alpha`) is increased during the `GRAFTING` phase of its lifecycle. Each strategy offers a different balance of speed, predictability, and safety.

All strategies inherit from the `GraftingStrategy` abstract base class and implement an `update()` method that calculates the new `alpha` value on each step.

---

### 1. Fixed Ramp Grafting (`FixedRampGrafting`)

This is the most basic and predictable strategy. It increases the `alpha` value linearly over a pre-configured number of steps.

* **Mechanism**: On each call to `update()`, the `alpha` value is incremented by a fixed amount: `1.0 / config.fixed_steps`.
* **Behavior**: This creates a smooth, constant-speed ramp-up of the seed's influence.
* **Use Case**: Ideal for scenarios where predictability is paramount or as a stable baseline for experiments. It is not responsive to the model's performance or stability.
* **Safety**: It is the default strategy used by the factory function if a specified strategy name is not found, ensuring robust behavior.

### 2. Performance-Linked Grafting (`PerformanceLinkedGrafting`)

This strategy dynamically adjusts the grafting speed based on the performance improvement of the seed.

* **Mechanism**: It measures performance by checking the seed's reconstruction loss on a holdout set of data (`self.seed.validate_on_holdout()`). The rate of `alpha` increase is then scaled by the `improvement_ratio`, which is the percentage of loss reduction from a baseline captured at the start of the phase.
* **Behavior**: Grafting accelerates when the seed shows significant improvement in its reconstruction task and proceeds at a base rate otherwise. The speed is capped between 0.5x and 3x the base step size.
* **Use Case**: Intended to quickly integrate "good" seeds.
* **Critical Caveat**: This strategy links grafting speed to **reconstruction performance (MSE loss)**, not the main model's task performance (e.g., classification accuracy). This can be counterproductive, as it incentivizes the seed to become a better autoencoder, which may not contribute positively to the overall goal.

### 3. Drift-Controlled Grafting (`DriftControlledGrafting`)

This is a stability-gated strategy that modulates the grafting speed based on the measured drift of the model's weights.

* **Mechanism**: It maintains a moving average of weight drift measurements over a small window. The grafting speed is adjusted based on this average drift relative to a configured `high_drift_threshold`.
* **Behavior**:
  * **Low Drift**: If the model is very stable, grafting speed is doubled (2x).
  * **Moderate Drift**: If the model is stable, grafting proceeds at the normal base rate (1x).
  * **High Drift**: If the model is unstable (drift exceeds the threshold), grafting is **paused** (`step_size = 0.0`) until stability returns.
* **Use Case**: A crucial safety measure to prevent a new seed from destabilizing the parent network. It ensures that grafting only proceeds when the model is in a stable state.

### 4. Gradient Norm-Gated Grafting (`GradNormGatedGrafting`)

This is another stability-gated strategy that uses the norm of the model's gradients as a health signal.

* **Mechanism**: It checks if the average gradient norm falls within a pre-configured stable range (`grad_norm_lower` and `grad_norm_upper`).
* **Behavior**:
  * **Stable Gradients**: If the gradient norm is within the acceptable range, grafting proceeds at the normal base rate.
  * **Unstable Gradients**: If the gradients are exploding (too high) or vanishing (too low), grafting is **paused**, and the current `alpha` value is held until the gradients stabilize.
* **Use Case**: Prevents grafting during periods of training instability, such as when encountering difficult batches of data or when the learning rate might be too high.

---

### Strategy Selection

The system uses a factory function, `get_strategy()`, to instantiate the desired grafting strategy at the beginning of an experiment. This allows for easy configuration and experimentation with different grafting behaviors simply by changing a name in the experiment's configuration file.
