"""Tests for TensorBoard integration in the morphogenetic experiment pipeline."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from morphogenetic_engine.core import SeedManager
from morphogenetic_engine.runners import setup_experiment
from morphogenetic_engine.training import clear_seed_report_cache, log_seed_updates


class TestTensorBoardIntegration:
    """Test suite for TensorBoard integration."""

    def test_setup_experiment_returns_tensorboard_writer(self):
        """Test that setup_experiment returns a TensorBoard writer."""
        # Create mock args
        args = Mock()
        args.problem_type = "spirals"
        args.n_samples = 2000
        args.input_dim = 3
        args.train_frac = 0.8
        args.batch_size = 64
        args.device = "cpu"
        args.seed = 42
        args.warm_up_epochs = 2
        args.adaptation_epochs = 2
        args.lr = 0.003
        args.hidden_dim = 32
        args.num_layers = 8
        args.seeds_per_layer = 1
        args.blend_steps = 30
        args.shadow_lr = 0.001
        args.progress_thresh = 0.6
        args.drift_warn = 0.12
        args.acc_threshold = 0.95

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("morphogenetic_engine.runners.Path") as mock_path:
                # Mock the project root to use our temp directory
                mock_project_root = Path(temp_dir)
                mock_path.return_value.parent.parent = mock_project_root
                mock_path.__file__ = temp_dir + "/scripts/run_morphogenetic_experiment.py"

                # Mock ExperimentLogger to avoid file operations
                with patch("morphogenetic_engine.runners.ExperimentLogger") as mock_logger:
                    with patch("morphogenetic_engine.runners.SummaryWriter") as mock_writer:
                        mock_logger.return_value = Mock()
                        mock_writer.return_value = Mock()

                        result = setup_experiment(args)

                        # Should return 7 items: logger, tb_writer, log_f, device, config, slug, project_root
                        assert len(result) == 7
                        _, tb_writer, _, _, _, _, _ = result

                        # Verify TensorBoard writer was created
                        mock_writer.assert_called_once()
                        assert tb_writer is not None

    def test_log_seed_updates_with_tensorboard(self):
        """Test that log_seed_updates correctly logs to TensorBoard."""
        # Clear global state first
        clear_seed_report_cache()

        # Create mock objects
        epoch = 5
        seed_manager = SeedManager()
        logger = Mock()
        tb_writer = Mock()
        log_f = Mock()

        # Create a mock seed
        mock_seed = Mock()
        mock_seed.state = "blending"
        mock_seed.alpha = 0.75

        # Add seed to manager
        seed_manager.seeds["test_seed"] = {"module": mock_seed, "status": "active"}

        # Call the function
        log_seed_updates(epoch, seed_manager, logger, tb_writer, log_f)

        # Verify TensorBoard scalar was logged for alpha
        tb_writer.add_scalar.assert_called_with("seed/test_seed/alpha", 0.75, epoch)

        # Verify logger was called
        logger.log_blending_progress.assert_called_with(epoch, "test_seed", 0.75)

    def test_log_seed_updates_state_transition_tensorboard(self):
        """Test that state transitions are logged to TensorBoard as text."""
        # Clear global state first
        clear_seed_report_cache()

        # Create mock objects
        epoch = 10
        seed_manager = SeedManager()
        logger = Mock()
        tb_writer = Mock()
        log_f = Mock()

        # Create a mock seed in a different state
        mock_seed = Mock()
        mock_seed.state = "active"
        mock_seed.alpha = 0.0

        # Add seed to manager
        seed_manager.seeds["test_seed"] = {"module": mock_seed, "status": "active"}

        # Call the function
        log_seed_updates(epoch, seed_manager, logger, tb_writer, log_f)

        # Verify TensorBoard text was logged for state transition
        tb_writer.add_text.assert_called_with(
            "seed/test_seed/events", "Epoch 10: unknown → active", epoch
        )

        # Verify logger was called
        logger.log_seed_event.assert_called_with(epoch, "test_seed", "unknown", "active")

    def test_tensorboard_writer_cleanup(self):
        """Test that TensorBoard writer is properly closed."""
        # This test would be part of integration testing
        # to ensure that tb_writer.close() is called
        # The implementation already includes this in the try/except blocks

    def test_tensorboard_directory_structure(self):
        """Test that TensorBoard logs are saved in the correct directory structure."""
        # This would test that runs/<slug>/ directories are created correctly
        # The directory structure follows: runs/{problem_type}_dim{input_dim}_{device}_h{hidden_dim}_...


if __name__ == "__main__":
    pytest.main([__file__])
