# Phase 1 Detailed Plan: Standardize Data Payloads & Enhance the Logger

**Objective:** Refactor the `ExperimentLogger` to function as a standardized data bus. All UI-related information will be channeled through this logger using strict, verifiable data contracts (`TypedDicts`), decoupling the backend logic from the UI's presentation layer.

---

### **Task 1.1: Define Standardized Event Payloads (`events.py`)**

**Goal:** Create a new file, `morphogenetic_engine/events.py`, to house all data contracts for UI-bound events. This centralizes the "API" between the backend and the UI, ensuring data consistency.

**Steps:**

1. **Create `morphogenetic_engine/events.py`**:
    * This file will contain all event-related definitions, making it the single source of truth for data contracts.
    * Add standard file header and imports: `TypedDict`, `Literal`, `dataclass`, `Any`.

2. **Define a new `EventType` Enum**:
    * Move the `EventType` enum from `logger.py` to the new `events.py`.
    * Refine the event types to be more comprehensive and less presentation-focused:
        * `EXPERIMENT_START` -> `SYSTEM_INIT`
        * `EXPERIMENT_END` -> `SYSTEM_SHUTDOWN`
        * `EPOCH_PROGRESS` -> `METRICS_UPDATE`
        * `PHASE_TRANSITION` -> `PHASE_UPDATE`
        * Consolidate `SEED_STATE_CHANGE`, `GERMINATION`, `BLENDING_PROGRESS` into a single, comprehensive `SEED_STATE_UPDATE`.

3. **Define `TypedDict` Payloads for Each Event**:
    * For each `EventType`, define a corresponding `TypedDict` to enforce a strict data structure.

    ```python
    # morphogenetic_engine/events.py

    from typing import TypedDict, Literal, Any

    class SystemInitPayload(TypedDict):
        """Payload for when the system starts."""
        config: dict[str, Any]
        timestamp: float

    class MetricsUpdatePayload(TypedDict):
        """Payload for sending all metrics after an epoch."""
        epoch: int
        metrics: dict[str, float | int]
        timestamp: float

    class PhaseUpdatePayload(TypedDict):
        """Payload for announcing a change in the experiment phase."""
        epoch: int
        from_phase: str
        to_phase: str
        total_epochs_in_phase: int | None
        timestamp: float

    SeedState = Literal["active", "dormant", "blending", "germinated", "fossilized"]

    class SeedInfo(TypedDict):
        """A snapshot of a single seed's state and metrics."""
        id: str
        state: SeedState
        layer: int
        index_in_layer: int
        metrics: dict[str, float | int | None] # e.g., train_loss, val_acc, alpha

    class SeedStateUpdatePayload(TypedDict):
        """Payload for providing a complete update of all tracked seeds."""
        epoch: int
        seeds: list[SeedInfo]
        timestamp: float

    # ... other payload definitions
    ```

4. **Redefine `LogEvent` Dataclass**:
    * Update the `LogEvent` dataclass in `events.py` to use a union of the new `TypedDict` payloads for its `data` field, creating a fully type-safe structure.

---

### **Task 1.2: Refactor `ExperimentLogger` (`logger.py`)**

**Goal:** Modify the `ExperimentLogger` to use the new event structures. The logger's sole responsibility will be to construct and log these standardized events, completely removing any direct knowledge of the UI/dashboard.

**Steps:**

1. **Update Imports**:
    * In `logger.py`, remove the local `EventType` and `LogEvent` definitions.
    * Import `EventType`, `LogEvent`, and all payload `TypedDict`s from `morphogenetic_engine.events`.

2. **Decouple from Dashboard**:
    * Remove the `dashboard` parameter from the `__init__` method.
    * Delete the `_send_to_dashboard` method entirely. The UI will be responsible for consuming log files, not for receiving direct calls.

3. **Refactor Logging Methods**:
    * Go through each `log_*` method and refactor it to build and record an event using the new `TypedDict` payloads.
    * **Consolidate Seed Logging:** Replace `log_seed_event`, `log_germination`, and `log_blending_progress` with a single method: `log_seed_state_update(self, epoch: int, seeds_data: list[SeedInfo])`. This method will create one `SeedStateUpdatePayload` event containing the state of all seeds.
    * **Rename other methods for clarity:**
        * `log_epoch_progress` -> `log_metrics_update`
        * `log_phase_transition` -> `log_phase_update`
        * `log_experiment_start` -> `log_system_init`
        * `log_experiment_end` -> `log_system_shutdown`

    * **Example Refactoring (`log_metrics_update`)**:

        ```python
        # morphogenetic_engine/logger.py

        from .events import (
            LogEvent,
            EventType,
            MetricsUpdatePayload,
            # ... other imports
        )

        # ...

        def log_metrics_update(self, epoch: int, metrics: dict[str, float | int]) -> None:
            """Log a comprehensive metrics update."""
            payload = MetricsUpdatePayload(
                epoch=epoch,
                metrics=metrics,
                timestamp=time.time()
            )
            event = LogEvent(
                timestamp=payload["timestamp"],
                epoch=epoch,
                event_type=EventType.METRICS_UPDATE,
                message=f"Metrics update for epoch {epoch}",
                data=payload,
            )
            self._record_event(event)
        ```

---

### **Task 1.3: Update Logger Tests (`test_logger.py`)**

**Goal:** Update the unit tests in `tests/test_logger.py` to verify that the refactored `ExperimentLogger` correctly creates and logs events conforming to the new `TypedDict` contracts.

**Steps:**

1. **Update Test Imports**:
    * Import the new `EventType`, `LogEvent`, and payload types from `morphogenetic_engine.events`.

2. **Remove Dashboard Mocking**:
    * Remove any mocks or patches related to the `dashboard` object, as it is no longer a dependency of the logger.

3. **Refactor Test Cases**:
    * Rename test functions to match the new logger method names (e.g., `test_log_metrics_update`).
    * For each test, call the refactored logger method with appropriate test data.
    * Assert that the generated `LogEvent` has the correct `event_type`.
    * Perform detailed assertions on the `event.data` payload to ensure it matches the `TypedDict` structure:
        * Check for the presence of all required keys.
        * Check that the types of the values are correct (e.g., `isinstance(data["epoch"], int)`).

    * **Example Test Snippet**:

        ```python
        # tests/test_logger.py

        def test_log_metrics_update(self, mock_logger, fixed_timestamp):
            """Test that METRICS_UPDATE events are logged with the correct payload."""
            mock_time.return_value = fixed_timestamp
            metrics = {"train_loss": 0.123, "val_acc": 0.95}

            mock_logger.log_metrics_update(epoch=10, metrics=metrics)

            assert len(mock_logger.events) == 1
            event = mock_logger.events[0]
            assert event.event_type == EventType.METRICS_UPDATE

            # Verify payload structure
            payload = event.data
            assert payload["epoch"] == 10
            assert payload["timestamp"] == fixed_timestamp
            assert payload["metrics"]["val_acc"] == 0.95
        ```
