### **Phase 1: Standardize Data Payloads & Enhance the Logger**

[cite_start]**Objective:** To refactor the `ExperimentLogger` to act as a standardized data bus with strict, verifiable data contracts for all UI events, as specified in the analysis document[cite: 10].

---

* **Task 1.1: Define and Document Standardized Event Payloads**
  * [cite_start]**Goal:** Formally specify the exact data structure for every event the UI will need to render, using the field names confirmed in the analysis[cite: 4, 16].
  * **Action:** In a new file `morphogenetic_engine/events.py`, define `TypedDict` classes for each payload. This makes the data contracts verifiable.
  * **File to Create:** `morphogenetic_engine/events.py`
  * **Implementation Detail:**

        ```python
        # morphogenetic_engine/events.py
        from typing import TypedDict, NotRequired, List, Tuple

        class MetricsPayload(TypedDict):
            [cite_start]"""Payload for metrics table updates[cite: 4, 16]."""
            Epoch: int
            Stage: str
            "Train Loss": float
            "Val Loss": float
            "Val Accuracy": float
            "Best Accuracy": float
            "Seeds Active": str # e.g., "3/16"

        class SeedStatesPayload(TypedDict):
            [cite_start]"""Payload for the seed states panel[cite: 5, 18]."""
            # List of tuples: (name, state, alpha)
            seed_info: List[Tuple[str, str, float]]

        class StageTransitionPayload(TypedDict):
            [cite_start]"""Payload for stage transition banners[cite: 5, 19]."""
            new_stage_name: str

        class GerminationPayload(TypedDict):
            [cite_start]"""Payload for germination banners[cite: 5, 20]."""
            seed_name: str
        ```

  * [cite_start]**Verification:** A code review of `morphogenetic_engine/events.py` confirms that the `TypedDict` definitions exactly match the data requirements outlined in the "Relevant Components" document[cite: 24].

* **Task 1.2: Refactor Logger Methods to Create Standard Payloads**
  * **Goal:** Modify `ExperimentLogger` methods to construct and record `LogEvent` objects that strictly adhere to the new `TypedDict` contracts.
  * **Action:** Refactor `log_metrics`, create `log_seed_states`, etc., in `ExperimentLogger` to use the new payload types.
  * **File to Modify:** `morphogenetic_engine/logger.py`
  * **Implementation Detail:**

        ```python
        # In ExperimentLogger class
        from .events import MetricsPayload, SeedStatesPayload # etc.

        def log_metrics(self, epoch: int, metrics_payload: MetricsPayload) -> None:
            event = LogEvent(
                # ...,
                event_type=EventType.METRICS_UPDATE,
                data=metrics_payload
            )
            self._record_event(event)

        def log_seed_states(self, epoch: int, seed_payload: SeedStatesPayload) -> None:
            event = LogEvent(
                # ...,
                event_type=EventType.SEED_STATE_UPDATE,
                data=seed_payload
            )
            self._record_event(event)
        ```

  * **Verification:** Update tests in `tests/test_logger.py` to call these methods with valid payloads and assert that the generated `LogEvent` has the correct `event_type` and `data`.

---

### **Phase 2: Connect Backend Components to the Logger**

**Objective:** To refactor all backend data sources to correctly compute and send data to the `ExperimentLogger`.

---

* **Task 2.1: Create Data Transformation Helpers**
  * [cite_start]**Goal:** Implement the helper functions identified in the analysis for preparing UI-ready data[cite: 31, 32].
  * **Action:** Create helper functions to count active seeds and format metrics. These can live in `morphogenetic_engine/utils.py` or directly within the experiment runner script.
  * **File to Modify:** `scripts/run_morphogenetic_experiment.py` or `morphogenetic_engine/utils.py`
  * **Implementation Detail:**

        ```python
        def count_active_seeds(seeds: list) -> int:
            [cite_start]"""Counts seeds in the 'active' state[cite: 32]."""
            return sum(1 for s in seeds if s.state == "active")

        def format_metrics(metrics: dict) -> dict:
            [cite_start]"""Formats float values to 4 decimal places for display[cite: 21, 32]."""
            formatted = metrics.copy()
            for key, value in formatted.items():
                if isinstance(value, float):
                    formatted[key] = f"{value:.4f}"
            return formatted
        ```

  * **Verification:** Unit test these helper functions to ensure they produce the correct output.

* **Task 2.2: Integrate the Main Experiment Loop**
  * [cite_start]**Goal:** Modify the main experiment loop to gather all necessary data, use the helpers, and call the logger with the correct payloads after each epoch[cite: 11, 28].
  * **Action:** In `scripts/run_morphogenetic_experiment.py`, gather all metrics and seed data, then call the logger.
  * **File to Modify:** `scripts/run_morphogenetic_experiment.py`
  * **Implementation Detail:**

        ```python
        # In the main experiment loop (for epoch in range(total_epochs):)
        # ... training and validation steps ...

        # [cite_start]1. Prepare Metrics Payload [cite: 16, 28]
        active_seeds_count = count_active_seeds(all_seeds)
        metrics_payload = {
            "Epoch": epoch + 1,
            "Stage": current_stage,
            "Train Loss": train_loss,
            "Val Loss": val_loss,
            "Val Accuracy": val_acc,
            "Best Accuracy": best_acc,
            "Seeds Active": f"{active_seeds_count}/{len(all_seeds)}"
        }
        logger.log_metrics(epoch, metrics_payload)

        # [cite_start]2. Prepare Seed States Payload [cite: 17, 18]
        seed_info = []
        for seed in all_seeds:
            if seed.name not in {"Tamiyo", "Karn"}:
                seed_info.append((seed.name, seed.state, seed.alpha))
        logger.log_seed_states(epoch, {"seed_info": seed_info})

        # [cite_start]3. Handle Stage Transitions & Germinations [cite: 19, 20]
        if stage_changed:
            logger.log_stage_transition(epoch, {"new_stage_name": new_stage})
        newly_active_seeds = get_newly_active_seeds() # From event trigger
        for seed in newly_active_seeds:
            logger.log_germination(epoch, {"seed_name": seed.name})
        ```

  * **Verification:** Run an experiment and inspect the `events.log` file to confirm that `METRICS_UPDATE`, `SEED_STATE_UPDATE`, and other events are being logged correctly on each epoch.

---

### **Phase 3: Implement the UI Data Consumer & Dispatcher**

[cite_start]**Objective:** To implement the data-receiving side of the UI, using the exact method names and structure proposed in the analysis document[cite: 26].

---

* **Task 3.1: Implement `RichDashboard` API and State Storage**
  * **Goal:** Create the `RichDashboard` class with internal state variables and the public API methods that the logger will call.
  * [cite_start]**Action:** In `morphogenetic_engine/cli_dashboard.py`, define the `RichDashboard` class with methods like `update_metrics`, `update_seed_states`, etc.[cite: 26].
  * **File to Modify:** `morphogenetic_engine/cli_dashboard.py`
  * **Implementation Detail:**

        ```python
        # In RichDashboard class
        def __init__(self):
            # ...
            self.metrics = {}
            self.seed_info = []

        def update_metrics(self, payload: MetricsPayload):
            [cite_start]"""Consumes metrics payload and stores it[cite: 26]."""
            self.metrics = payload

        def update_seed_states(self, payload: SeedStatesPayload):
            [cite_start]"""Consumes seed states payload and stores it[cite: 26]."""
            self.seed_info = payload["seed_info"]
        
        def show_stage_transition(self, payload: StageTransitionPayload):
             # ... logic to handle banner display ...

        def notify_germination(self, payload: GerminationPayload):
             # ... logic to handle banner display ...
        ```

  * **Verification:** Unit test the `RichDashboard` methods to ensure they correctly update the internal state variables (`self.metrics`, `self.seed_info`).

* **Task 3.2: Implement the Event Dispatcher in `ExperimentLogger`**
  * **Goal:** Make the logger forward events to the dashboard by calling the newly defined API.
  * **Action:** Implement the logic in `_send_to_dashboard` to map `EventType` to the correct `RichDashboard` method.
  * **File to Modify:** `morphogenetic_engine/logger.py`
  * **Implementation Detail:**

        ```python
        # In ExperimentLogger._send_to_dashboard
        if event.event_type == EventType.METRICS_UPDATE:
            self.dashboard.update_metrics(event.data)
        elif event.event_type == EventType.SEED_STATE_UPDATE:
            self.dashboard.update_seed_states(event.data)
        # ... etc. for other events
        ```

  * **Verification:** Run a test with a mock dashboard and assert that the logger calls the correct methods (`update_metrics`, etc.) in response to events.

---

### **Phase 4: Render Data in the UI Panels**

[cite_start]**Objective:** To connect the dashboard's internal state to the `rich` renderables, bringing the UI to life exactly as described in the analysis[cite: 34, 35].

---

* **Task 4.1: Wire the Metrics and Seed States Panels**
  * **Goal:** Render the `self.metrics` dictionary and `self.seed_info` list into their respective UI panels.
  * **Action:** Modify the `_create_*_panel` methods to generate `rich` objects from the dashboard's state.
  * **File to Modify:** `morphogenetic_engine/cli_dashboard.py`
  * **Implementation Detail:**

        ```python
        # In RichDashboard class
        def _create_metrics_table_panel(self) -> Panel:
            [cite_start]"""Renders the metrics dictionary into a rich Table[cite: 30]."""
            table = Table(title="Metrics")
            # ... add columns ...
            for key, value in self.metrics.items():
                table.add_row(key, str(value))
            return Panel(table)

        def _create_seed_states_panel(self) -> Panel:
            [cite_start]"""Renders the seed_info list[cite: 30]."""
            # ... create a rich.Panel or rich.Table ...
            content = ""
            for name, state, alpha in self.seed_info:
                content += f"{name}: {state} (alpha: {alpha:.2f})\n"
            return Panel(content, title="Seed States")
        ```

  * [cite_start]**Verification:** Run the full experiment. The UI panels for "Metrics" and "Seed States" must populate with live data that updates every epoch, matching the format specified in the analysis[cite: 2, 4, 5].

# Phase 5 - Rename cli_dashboard.py to ui_dashboard.py

Do this renaming please.
