# The Authoritative Guide to the Morphogenetic Seed Lifecycle

This document traces each `SentinelSeed` from its inert birth to its ultimate fate—**FOSSILIZED** or **CULLED**.

---

## Lifecycle Path

```text
DORMANT 
  → GERMINATED 
    → TRAINING 
      → (soft-test) → GRAFTING 
        → STABILIZATION 
          → FINE-TUNING 
            → (FOSSILIZED or CULLED)
```

---

### 1. DORMANT (⚪)

* **Trigger**
  Initial state at network creation or when a culled seed’s embargo expires.

* **Behavior**

  * Identity pass-through: `output = x`.
  * Buffers every activation for later training.
  * Controller polls its **health signal** (activation variance) to spot bottlenecks.

* **Next** → **GERMINATED** when selected by the controller upon a performance plateau.

---

### 2. GERMINATED (🌱)

* **Trigger**
  Controller calls for germination on the chosen seed.

* **Behavior**

  * Acts as a **parking lot** so seeds train one at a time.
  * Child network weights are re-initialized and set `requires_grad = True`.

* **Next** → **TRAINING** when no other seed is actively training.

---

### 3. TRAINING (🟢) – Initial Reconstruction

* **Trigger**
  SeedManager promotes from **GERMINATED**.

* **Behavior**

  * Child network learns to **reconstruct** its buffered activations (MSE loss).
  * Parent network still uses identity pass—zero impact on existing features.

* **Soft-Test Gate**
  At epoch’s end, check reconstruction error:

  * **Pass** → proceed to **GRAFTING**
  * **Fail** → immediately **CULLED** (enter embargo)

---

### 4. GRAFTING (🔗)

* **Trigger**
  Successful soft-test after **TRAINING**.

* **Behavior**

  * Blend factor `α` ramps from 0 → 1 over *N* steps.
  * Forward pass:

    ```python
    output = (1 − α) * x + α * child(x)
    ```

* **Next** → **STABILIZATION** once `α ≈ 1.0`.

---

### 5. STABILIZATION (🛠)

* **Trigger**
  Completion of **GRAFTING** (`α` at 1.0).

* **Behavior**

  * Hold `α = 1.0` for a handful of epochs with **no** parameter updates.
  * Allows any residual blending artifacts to settle before fine-tuning.

* **Next** → **FINE-TUNING**

---

### 6. FINE-TUNING (🧑‍⚖️)

* **Trigger**
  After **STABILIZATION** hold.

* **Behavior**

  * **Live Test:** child’s residual is fully active: `output = x + child(x)`.
  * **Task Aligned:** switch child training to the model’s primary loss (e.g. cross-entropy).
  * Controller monitors global metrics (`val_acc`, `val_loss`) during this period.

* **Next** → Terminal **FOSSILIZED** if performance improves, or **CULLED** if not.

---

### 7. FOSSILIZED (🦴)

* **Trigger**
  Child raises overall performance during **FINE-TUNING**.

* **Behavior**

  * Freeze the child and **replace** the seed in the graph with a lightweight grafted module.
  * Persist the updated network to disk.

* **Terminal** → lifecycle complete.

---

### 8. CULLED (🥀)

* **Trigger**
  Failure at the soft-test (end of **TRAINING**) or final **FINE-TUNING** assessment.

* **Behavior**

  * Seed is marked culled, parameters frozen, slot embarged.
  * No new germinations in this slot for a set number of epochs.

* **Next** → After embargo, seed resets to **DORMANT**.

---

## Why This Matters

1. **Early Cull:** Removes unstable subnets before they touch the backbone.
2. **Clear Grafting + Stabilization:** Smoothly introduces new capacity without sudden jolts.
3. **Task-Aligned Fine-Tuning:** Ensures subnets learn features that actually improve end-task performance.

This refined lifecycle balances **safety** (no abrupt shifts) with **effectiveness** (every gradient step contributes to your true objective).
