"""
Comprehensive tests for the KasminaMicro component.

This module provides thorough testing coverage for the KasminaMicro germination
controller, focusing on plateau detection, germination triggering, and seed
selection logic. Tests follow modern Python 3.12+ practices with proper
isolation and edge case coverage.

Test Categories:
- Unit tests: Isolated KasminaMicro behavior with mocking
- Integration: Monitoring system integration
- Property-based: Hypothesis-driven boundary testing
- Edge cases: Boundary conditions and accuracy thresholds
"""

# pylint: disable=protected-access,redefined-outer-name

from unittest.mock import Mock, patch

import pytest
from hypothesis import given
from hypothesis import strategies as st

from morphogenetic_engine.core import KasminaMicro, SeedManager


class TestConstants:
    """Constants for test values to avoid magic numbers."""

    LOW_HEALTH_SIGNAL = 0.1
    MEDIUM_HEALTH_SIGNAL = 0.5
    HIGH_HEALTH_SIGNAL = 0.9
    HIGH_ACCURACY = 0.95
    LOW_ACCURACY = 0.5
    PLATEAU_THRESHOLD = 1e-3
    SMALL_LOSS_DELTA = 1e-4
    PATIENCE_SHORT = 2
    PATIENCE_MEDIUM = 5
    PATIENCE_LONG = 15


@pytest.fixture
def clean_seed_manager():
    """Provide a clean SeedManager instance for testing."""
    # Reset singleton first
    SeedManager.reset_singleton()
    manager = SeedManager()
    manager.seeds.clear()
    manager.germination_log.clear()
    return manager


@pytest.fixture
def mock_seed_factory():
    """Factory for creating mock seeds with customizable health signals."""

    def _create_mock_seed(health_signal: float = TestConstants.MEDIUM_HEALTH_SIGNAL) -> Mock:
        mock_seed = Mock()
        mock_seed.get_health_signal = Mock(return_value=health_signal)
        mock_seed.initialize_child = Mock(return_value=None)
        return mock_seed

    return _create_mock_seed


class TestKasminaMicro:
    """Test suite for KasminaMicro germination controller."""

    def test_initialization(self, clean_seed_manager) -> None:
        """Test KasminaMicro initialization."""
        manager = clean_seed_manager
        km = KasminaMicro(
            manager,
            patience=TestConstants.PATIENCE_LONG,
            delta=TestConstants.PLATEAU_THRESHOLD,
            acc_threshold=TestConstants.HIGH_ACCURACY,
        )

        assert km.seed_manager is manager
        assert km.patience == TestConstants.PATIENCE_LONG
        assert km.delta == pytest.approx(TestConstants.PLATEAU_THRESHOLD)
        assert km.acc_threshold == pytest.approx(TestConstants.HIGH_ACCURACY)
        assert km.plateau == 0
        assert km.prev_loss == float("inf")

    def test_step_loss_improvement_resets_plateau(self, clean_seed_manager) -> None:
        """Test step with improving loss resets plateau counter."""
        manager = clean_seed_manager
        km = KasminaMicro(
            manager, patience=TestConstants.PATIENCE_SHORT, delta=TestConstants.PLATEAU_THRESHOLD
        )

        # First step should not trigger germination
        result = km.step(1.0, TestConstants.LOW_ACCURACY)
        assert result is False
        assert km.plateau == 0
        assert km.prev_loss == pytest.approx(1.0)

        # Small improvement should reset plateau
        result = km.step(1.0 + TestConstants.SMALL_LOSS_DELTA, TestConstants.LOW_ACCURACY)
        assert result is False
        assert km.plateau == 1  # No improvement, plateau increases

        # Significant improvement should reset plateau
        result = km.step(0.8, TestConstants.LOW_ACCURACY)
        assert result is False
        assert km.plateau == 0  # Reset due to improvement
        assert km.prev_loss == pytest.approx(0.8)

    def test_plateau_detection_high_accuracy_blocks_germination(self, clean_seed_manager) -> None:
        """Test plateau detection with high accuracy blocks germination."""
        manager = clean_seed_manager
        km = KasminaMicro(
            manager,
            patience=TestConstants.PATIENCE_SHORT,
            delta=TestConstants.PLATEAU_THRESHOLD,
            acc_threshold=TestConstants.HIGH_ACCURACY,
        )

        # Set initial loss
        km.step(1.0, TestConstants.HIGH_ACCURACY)

        # Create plateau with high accuracy (shouldn't germinate)
        for _ in range(TestConstants.PATIENCE_SHORT + 2):
            result = km.step(1.0 + TestConstants.SMALL_LOSS_DELTA, TestConstants.HIGH_ACCURACY)
            assert result is False  # No germination due to high accuracy

    def test_germination_triggered_by_plateau_and_low_accuracy(
        self, clean_seed_manager, mock_seed_factory
    ) -> None:
        """Test germination triggering under correct conditions."""
        manager = clean_seed_manager
        mock_seed = mock_seed_factory(health_signal=TestConstants.LOW_HEALTH_SIGNAL)
        manager.register_seed(mock_seed, "test_seed")

        km = KasminaMicro(
            manager,
            patience=TestConstants.PATIENCE_SHORT,
            delta=TestConstants.PLATEAU_THRESHOLD,
            acc_threshold=TestConstants.HIGH_ACCURACY,
        )

        # Set initial loss with low accuracy
        result1 = km.step(1.0, TestConstants.LOW_ACCURACY)
        assert result1 is False
        assert km.plateau == 0

        # Create plateau with low accuracy
        result2 = km.step(1.0 + TestConstants.SMALL_LOSS_DELTA, TestConstants.LOW_ACCURACY)
        assert result2 is False
        assert km.plateau == 1

        with patch.object(manager, "request_germination", return_value=True) as mock_germ:
            result3 = km.step(1.0 + 2 * TestConstants.SMALL_LOSS_DELTA, TestConstants.LOW_ACCURACY)
            assert result3 is True  # Should trigger germination
            mock_germ.assert_called_once_with("test_seed")

    def test_germination_resets_plateau_counter(
        self, clean_seed_manager, mock_seed_factory
    ) -> None:
        """Test that successful germination resets plateau counter."""
        manager = clean_seed_manager
        mock_seed = mock_seed_factory(health_signal=TestConstants.LOW_HEALTH_SIGNAL)
        manager.register_seed(mock_seed, "test_seed")

        km = KasminaMicro(
            manager, patience=TestConstants.PATIENCE_SHORT, delta=TestConstants.PLATEAU_THRESHOLD
        )

        # Build up plateau
        km.step(1.0, TestConstants.LOW_ACCURACY)
        km.step(1.0 + TestConstants.SMALL_LOSS_DELTA, TestConstants.LOW_ACCURACY)
        assert km.plateau == 1

        # Trigger germination
        with patch.object(manager, "request_germination", return_value=True):
            result = km.step(1.0 + 2 * TestConstants.SMALL_LOSS_DELTA, TestConstants.LOW_ACCURACY)
            assert result is True
            assert km.plateau == 0  # Reset after germination

    def test_select_seed_no_dormant_seeds(self, clean_seed_manager) -> None:
        """Test seed selection when no dormant seeds available."""
        manager = clean_seed_manager
        km = KasminaMicro(manager)

        result = km._select_seed()
        assert result is None

    def test_select_seed_chooses_worst_health(self, clean_seed_manager, mock_seed_factory) -> None:
        """Test seed selection chooses seed with worst health signal."""
        manager = clean_seed_manager

        # Create mock seeds with different health signals
        mock_seed1 = mock_seed_factory(health_signal=TestConstants.MEDIUM_HEALTH_SIGNAL)
        mock_seed2 = mock_seed_factory(health_signal=TestConstants.LOW_HEALTH_SIGNAL)  # Worst
        mock_seed3 = mock_seed_factory(health_signal=0.3)

        manager.register_seed(mock_seed1, "seed1")
        manager.register_seed(mock_seed2, "seed2")
        manager.register_seed(mock_seed3, "seed3")

        km = KasminaMicro(manager)
        result = km._select_seed()

        assert result == "seed2"  # Should select seed with lowest health signal

    def test_select_seed_ignores_non_dormant(self, clean_seed_manager, mock_seed_factory) -> None:
        """Test seed selection ignores non-dormant seeds."""
        manager = clean_seed_manager

        mock_seed1 = mock_seed_factory(health_signal=TestConstants.LOW_HEALTH_SIGNAL)
        mock_seed2 = mock_seed_factory(health_signal=0.2)

        manager.register_seed(mock_seed1, "seed1")
        manager.register_seed(mock_seed2, "seed2")

        # Make seed1 non-dormant (should be ignored despite better health signal)
        manager.seeds["seed1"]["status"] = "active"

        km = KasminaMicro(manager)
        result = km._select_seed()

        assert result == "seed2"  # Should select the dormant seed

    @given(accuracy=st.floats(min_value=0.85, max_value=1.0))
    @pytest.mark.property
    def test_accuracy_threshold_boundary_conditions(self, accuracy: float) -> None:
        """Test accuracy threshold boundary conditions with property-based testing."""
        # Create manager directly to avoid fixture issues with Hypothesis
        SeedManager.reset_singleton()
        manager = SeedManager()
        manager.seeds.clear()
        km = KasminaMicro(manager, patience=1, acc_threshold=0.8)

        # High accuracy should prevent germination regardless of plateau
        km.step(1.0, accuracy)
        result = km.step(1.0 + TestConstants.SMALL_LOSS_DELTA, accuracy)

        if accuracy >= 0.8:
            assert result is False  # No germination due to high accuracy
        # Note: For accuracy < 0.8, we'd need seeds registered to test germination

        # Cleanup
        SeedManager.reset_singleton()

    def test_monitoring_integration(self, clean_seed_manager) -> None:
        """Test integration with monitoring system."""
        manager = clean_seed_manager
        km = KasminaMicro(manager, patience=TestConstants.PATIENCE_SHORT)

        # Mock the monitoring module's get_monitor function
        with patch("morphogenetic_engine.monitoring.get_monitor") as mock_get_monitor:
            mock_monitor = Mock()
            mock_get_monitor.return_value = mock_monitor

            # Test that monitoring methods are called
            km.step(1.0, TestConstants.LOW_ACCURACY)

            mock_monitor.update_kasmina_metrics.assert_called_once_with(
                0, TestConstants.PATIENCE_SHORT
            )

            # Test germination recording
            mock_seed = Mock()
            mock_seed.get_health_signal = Mock(return_value=TestConstants.LOW_HEALTH_SIGNAL)
            manager.register_seed(mock_seed, "test_seed")

            # Build up plateau and trigger germination
            km.step(1.0 + TestConstants.SMALL_LOSS_DELTA, TestConstants.LOW_ACCURACY)
            km.step(1.0 + 2 * TestConstants.SMALL_LOSS_DELTA, TestConstants.LOW_ACCURACY)

            mock_monitor.record_germination.assert_called_once()
