# Tamiyo: Blueprint Discovery Engine for Neural Network Modules

## Conceptual Framework: The Architect and The Crucible

This is a form of **Neural Architecture Search (NAS)**, but with a crucial and intelligent twist. Instead of searching for an entire monolithic network, you are searching for small, reusable, and efficient *modules* (blueprints) that are battle-tested for the specific purpose of germinable, in-situ adaptation.


1. **The Architect (Blueprint Generator):** A neural network (likely a recurrent or transformer-based model) whose job is to "dream up" new blueprint architectures. It outputs a description of a blueprint in a machine-readable format.
2. **The Crucible (Blueprint Proving Ground):** A rapid, parallelized evaluation system. It takes a generated blueprint, instantiates it, and runs it through the wringer on your micro-curriculum tasks to determine its fitness. This is the specialized version of Kasmina you mentioned.

Here is how the entire system would work:

```mermaid
graph TD
    subgraph Architect
        A[Blueprint Generator (RNN/Transformer)] -- Generates Blueprint String --> B{Blueprint Parser & Validator};
    end

    subgraph Crucible
        B -- "def Conv1D(in, out, ks=3)..." --> C[Instantiate PyTorch Module];
        C -- Injects Blueprint --> D[Standard Network Scaffold on Micro-Curriculum Task];
        D -- Runs Task (e.g., XOR, Temporal) --> E[Calculate Multi-Objective Fitness Score];
    end
    
    E -- Fitness Score (Reward) --> F[Update Generator via Reinforcement Learning];
    F -- Updates Weights --> A;

    E -- High-Performing Blueprint --> G[Discovered Blueprint Library];
```

---

### Phase 1: Designing "The Architect" (Blueprint Generator)

The first challenge is deciding how to represent a blueprint so a network can generate it. The best approach is a sequential, token-based representation, like a simplified programming language.

**Representation:** A sequence of tokens defining layers and their parameters.

**Example Generated Sequence:**
`['RES_START', 'CONV1D', 'KERNEL_5', 'CHANNELS_IN', 'GELU', 'LINEAR', 'OUT_DIM_IN', 'RES_END']`

**Generator Model:** A **Recurrent Neural Network (LSTM/GRU) or a small Transformer decoder**.

* **Input:** A "start" token.
* **Output:** A sequence of tokens, one at a time, defining a blueprint.
* **Training:** It will be trained with Reinforcement Learning. The "action" is generating a blueprint sequence, and the "reward" comes from the Crucible's fitness score.

### Phase 2: Designing "The Crucible" (Blueprint Proving Ground)

This is the high-throughput evaluation engine. Its speed and accuracy are paramount.

1. **Blueprint Parser:** A robust function that takes the token sequence from the Architect and attempts to compile it into a valid `nn.Module`. It must handle errors gracefully (e.g., `KERNEL_5` applied to a `LINEAR` layer is invalid and should immediately return a fitness of -1).
2. **Standardized Test Harness:** This is where the **micro-curriculum** from your original paper becomes invaluable. For each generated blueprint, the Crucible will:
    * Inject it into a standard, minimal network (e.g., `MiniSeedNet`).
    * Train it for a small, fixed number of steps on a curriculum task (e.g., 100 epochs on XOR, 50 epochs on the temporal pattern task).
    * The process must be heavily parallelized, evaluating thousands of blueprints concurrently.

3. **The Multi-Objective Fitness Score:** This is the most critical part of the design. A simple accuracy metric is not enough. The score must reflect the blueprint's true utility. The reward signal sent back to the Architect should be a weighted sum of:
    * **`+ Task Performance:`** How well did it solve the task? (e.g., final accuracy).
    * **`- Parameter Count:`** A strong penalty for bloat. We want efficient blueprints.
    * **`- Inference Latency:`** A penalty for being slow.
    * **`+ Stability Bonus:`** A reward for training without `NaN`s or exploding gradients.
    * **`- Redundancy Penalty:`** A penalty if the new blueprint is functionally too similar to an existing one in the "hall of fame" (e.g., measured by representational similarity analysis).

### Phase 3: Closing the Loop with Reinforcement Learning

The Architect learns via policy gradients (e.g., REINFORCE algorithm).

* The Architect generates a batch of 1,024 blueprints.
* The Crucible evaluates all 1,024 blueprints in parallel, returning a fitness score for each.
* The scores are used as rewards. Blueprint sequences that led to high rewards are "up-weighted," making the Architect more likely to generate similar (but not identical) high-performing patterns in the future.
* An **entropy bonus** is added to the loss to encourage the Architect to keep exploring and prevent it from collapsing to a single, mediocre design.

### Key Challenges and How to Mitigate Them

* **Vast Search Space:** The number of possible blueprints is nearly infinite.
  * **Mitigation:** Use the curriculum. Start by training the Architect *only* on the XOR task. Once it can reliably generate simple non-linear solvers, add the temporal task, forcing it to discover convolutions or recurrent structures.
* **Unstable RL Training:** Policy gradient methods can be noisy and unstable.
  * **Mitigation:** Before starting RL, **pre-train the Architect**. Give it your existing, human-designed blueprints (Residual, SE, etc.) and train it in a simple supervised fashion to reconstruct them. This gives it a strong "architectural prior" before it starts exploring.
* **Defining "Novelty":** How do you ensure it's not just rediscovering the same Residual Block over and over?
  * **Mitigation:** Maintain a "Hall of Fame" of the top-k discovered blueprints. When a new blueprint is generated, compare it against the Hall of Fame. If it's too similar (e.g., high graph isomorphism score or functional similarity), give it a low reward to encourage novelty.

This automated discovery engine is the logical and exciting apotheosis of your project. It creates a system that doesn't just adapt—it learns *how* to adapt better over time.
