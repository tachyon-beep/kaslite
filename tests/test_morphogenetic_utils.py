"""Tests for morphogenetic_engine.utils module.

This module provides comprehensive tests for utility functions including
experiment configuration, logging, metrics export, and testing mode detection.
"""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from morphogenetic_engine import utils
from tests.utils import (
    assert_log_file_format,
    assert_valid_experiment_slug,
    assert_valid_metrics_json,
    create_configured_seed_manager,
    create_mock_args,
    create_test_experiment_config,
    create_test_final_stats,
    temporary_directory,
    temporary_file,
)


class TestTestingModeDetection:
    """Test suite for testing mode detection functionality."""

    def test_is_testing_mode_with_pytest(self):
        """Test that testing mode is detected when pytest is in modules."""
        # pytest should already be in sys.modules when running tests
        assert utils.is_testing_mode() is True

    def test_is_testing_mode_with_unittest(self, mocker):
        """Test that testing mode is detected when unittest is in modules."""
        # Mock sys.modules to include unittest but not pytest
        mock_modules = {"unittest": MagicMock()}
        mocker.patch.dict("sys.modules", mock_modules, clear=True)

        assert utils.is_testing_mode() is True

    def test_is_testing_mode_without_test_modules(self, mocker):
        """Test that testing mode is not detected when no test modules are present."""
        # Mock sys.modules without test modules
        mock_modules = {"os": MagicMock(), "sys": sys}
        mocker.patch.dict("sys.modules", mock_modules, clear=True)

        assert utils.is_testing_mode() is False

    def test_is_testing_mode_handles_exceptions(self, mocker):
        """Test that testing mode detection handles exceptions gracefully."""
        # Mock sys.modules to raise an exception
        mocker.patch("sys.modules", side_effect=ImportError("Mock error"))

        assert utils.is_testing_mode() is False


class TestExperimentSlugGeneration:
    """Test suite for experiment slug generation."""

    def test_generate_experiment_slug_basic(self):
        """Test basic experiment slug generation."""
        args = create_mock_args()
        slug = utils.generate_experiment_slug(args)

        expected = "spirals_dim2_cpu_h64_bs30_lr0.001_pt0.6_dw0.12"
        assert slug == expected
        assert_valid_experiment_slug(slug)

    def test_generate_experiment_slug_different_params(self):
        """Test experiment slug generation with different parameters."""
        args = create_mock_args(
            problem_type="moons",
            input_dim=3,
            device="cuda",
            hidden_dim=128,
            blend_steps=50,
            shadow_lr=0.01,
            progress_thresh=0.8,
            drift_warn=0.05,
        )
        slug = utils.generate_experiment_slug(args)

        expected = "moons_dim3_cuda_h128_bs50_lr0.01_pt0.8_dw0.05"
        assert slug == expected
        assert_valid_experiment_slug(slug)

    @pytest.mark.parametrize(
        "problem_type,expected_prefix",
        [
            ("spirals", "spirals"),
            ("moons", "moons"),
            ("complex_moons", "complex_moons"),
            ("clusters", "clusters"),
            ("spheres", "spheres"),
        ],
    )
    def test_generate_experiment_slug_all_problem_types(self, problem_type, expected_prefix):
        """Test experiment slug generation for all problem types."""
        args = create_mock_args(problem_type=problem_type)
        slug = utils.generate_experiment_slug(args)

        assert slug.startswith(expected_prefix)
        assert_valid_experiment_slug(slug)


class TestExperimentConfiguration:
    """Test suite for experiment configuration creation."""

    def test_create_experiment_config_basic(self):
        """Test basic experiment configuration creation."""
        args = create_mock_args()
        device = "cpu"

        config = utils.create_experiment_config(args, device)

        # Check all required fields are present
        required_fields = [
            "problem_type",
            "n_samples",
            "input_dim",
            "train_frac",
            "batch_size",
            "device",
            "seed",
            "warm_up_epochs",
            "adaptation_epochs",
            "lr",
            "hidden_dim",
            "num_layers",
            "seeds_per_layer",
            "blend_steps",
            "shadow_lr",
            "progress_thresh",
            "drift_warn",
            "acc_threshold",
        ]

        for field in required_fields:
            assert field in config, f"Missing field: {field}"

        # Check specific values
        assert config["problem_type"] == "spirals"
        assert config["device"] == "cpu"
        assert config["input_dim"] == 2
        assert config["hidden_dim"] == 64

    def test_create_experiment_config_device_conversion(self):
        """Test that device is properly converted to string."""
        args = create_mock_args()

        # Test with string device
        config = utils.create_experiment_config(args, "cuda")
        assert config["device"] == "cuda"

        # Test with mock device object
        mock_device = MagicMock()
        mock_device.__str__ = MagicMock(return_value="cuda:0")
        config = utils.create_experiment_config(args, mock_device)
        assert config["device"] == "cuda:0"

    def test_create_experiment_config_preserves_args_values(self):
        """Test that configuration preserves all argument values correctly."""
        args = create_mock_args(
            problem_type="clusters", n_samples=2000, lr=0.01, hidden_dim=256, acc_threshold=0.99
        )
        device = "cuda"

        config = utils.create_experiment_config(args, device)

        assert config["problem_type"] == "clusters"
        assert config["n_samples"] == 2000
        assert abs(config["lr"] - 0.01) < 1e-9
        assert config["hidden_dim"] == 256
        assert abs(config["acc_threshold"] - 0.99) < 1e-9


class TestExperimentLogging:
    """Test suite for experiment logging functionality."""

    def test_write_experiment_log_header_basic(self):
        """Test basic experiment log header writing."""
        with temporary_file(suffix=".log") as log_path:
            config = create_test_experiment_config()
            args = create_mock_args()

            with open(log_path, "w", encoding="utf-8") as log_f:
                utils.write_experiment_log_header(log_f, config, args)

            with open(log_path, "r", encoding="utf-8") as log_f:
                content = log_f.read()

            assert_log_file_format(content)
            assert "# Morphogenetic Architecture Experiment Log" in content
            assert "# problem_type: spirals" in content
            assert "# input_dim: 2" in content
            assert "epoch,seed,state,alpha" in content

    def test_write_experiment_log_header_with_spirals_params(self):
        """Test log header includes spirals-specific parameters."""
        with temporary_file(suffix=".log") as log_path:
            config = create_test_experiment_config(problem_type="spirals")
            args = create_mock_args(problem_type="spirals", noise=0.2, rotations=2.0)

            with open(log_path, "w", encoding="utf-8") as log_f:
                utils.write_experiment_log_header(log_f, config, args)

            with open(log_path, "r", encoding="utf-8") as log_f:
                content = log_f.read()

            assert "# noise: 0.2" in content
            assert "# rotations: 2.0" in content

    def test_write_experiment_log_header_with_moons_params(self):
        """Test log header includes moons-specific parameters."""
        with temporary_file(suffix=".log") as log_path:
            config = create_test_experiment_config(problem_type="moons")
            args = create_mock_args(problem_type="moons", moon_noise=0.15, moon_sep=1.5)

            with open(log_path, "w", encoding="utf-8") as log_f:
                utils.write_experiment_log_header(log_f, config, args)

            with open(log_path, "r", encoding="utf-8") as log_f:
                content = log_f.read()

            assert "# moon_noise: 0.15" in content
            assert "# moon_sep: 1.5" in content

    def test_write_experiment_log_header_with_clusters_params(self):
        """Test log header includes clusters-specific parameters."""
        with temporary_file(suffix=".log") as log_path:
            config = create_test_experiment_config(problem_type="clusters")
            args = create_mock_args(
                problem_type="clusters",
                cluster_count=5,
                cluster_size=200,
                cluster_std=0.8,
                cluster_sep=3.0,
            )

            with open(log_path, "w", encoding="utf-8") as log_f:
                utils.write_experiment_log_header(log_f, config, args)

            with open(log_path, "r", encoding="utf-8") as log_f:
                content = log_f.read()

            assert "# cluster_count: 5" in content
            assert "# cluster_size: 200" in content
            assert "# cluster_std: 0.8" in content
            assert "# cluster_sep: 3.0" in content

    def test_write_experiment_log_footer_basic(self):
        """Test basic experiment log footer writing."""
        with temporary_file(suffix=".log") as log_path:
            final_stats = create_test_final_stats()
            seed_manager = create_configured_seed_manager(num_seeds=3)

            with open(log_path, "w", encoding="utf-8") as log_f:
                utils.write_experiment_log_footer(log_f, final_stats, seed_manager)

            with open(log_path, "r", encoding="utf-8") as log_f:
                content = log_f.read()

            assert "# Experiment completed successfully" in content
            assert "# Final best accuracy: 0.9500" in content
            assert "# Seeds activated:" in content
            assert "# ===== LOG COMPLETE =====" in content

    def test_write_experiment_log_footer_with_active_seeds(self):
        """Test log footer with active seeds."""
        with temporary_file(suffix=".log") as log_path:
            final_stats = create_test_final_stats(seeds_activated=True)
            seed_manager = create_configured_seed_manager(num_seeds=5)

            # Set some seeds to active
            for i, (_, info) in enumerate(seed_manager.seeds.items()):
                if i < 2:  # Activate first 2 seeds
                    info["module"].state = "active"

            with open(log_path, "w", encoding="utf-8") as log_f:
                utils.write_experiment_log_footer(log_f, final_stats, seed_manager)

            with open(log_path, "r", encoding="utf-8") as log_f:
                content = log_f.read()

            assert "# Seeds activated: 2/5" in content

    def test_write_experiment_log_footer_no_seeds_activated(self):
        """Test log footer when no seeds are activated."""
        with temporary_file(suffix=".log") as log_path:
            final_stats = create_test_final_stats(seeds_activated=False)
            seed_manager = create_configured_seed_manager(num_seeds=3)

            with open(log_path, "w", encoding="utf-8") as log_f:
                utils.write_experiment_log_footer(log_f, final_stats, seed_manager)

            with open(log_path, "r", encoding="utf-8") as log_f:
                content = log_f.read()

            assert "# Seeds activated: 0/3" in content


class TestMetricsExport:
    """Test suite for metrics export functionality."""

    def test_export_metrics_for_dvc_basic(self):
        """Test basic metrics export for DVC."""
        with temporary_directory() as temp_dir:
            results_dir = temp_dir / "results"
            results_dir.mkdir()

            final_stats = create_test_final_stats()
            slug = "test_dim2_cpu_h64_bs30_lr0.001_pt0.6_dw0.12"

            utils.export_metrics_for_dvc(final_stats, slug, temp_dir)

            # Check file was created
            metrics_file = results_dir / f"metrics_{slug}.json"
            assert metrics_file.exists()

            # Check file contents
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics = json.load(f)

            assert_valid_metrics_json(metrics)
            assert abs(metrics["best_acc"] - 0.95) < 1e-9
            assert abs(metrics["accuracy_dip"] - 0.05) < 1e-9
            assert metrics["recovery_time"] == 10
            assert metrics["seeds_activated"] is True

    def test_export_metrics_for_dvc_filters_none_values(self):
        """Test that metrics export filters out None values."""
        with temporary_directory() as temp_dir:
            results_dir = temp_dir / "results"
            results_dir.mkdir()

            final_stats = create_test_final_stats(accuracy_dip=None, recovery_time=None)
            slug = "test_slug"

            utils.export_metrics_for_dvc(final_stats, slug, temp_dir)

            # Check file contents
            metrics_file = results_dir / f"metrics_{slug}.json"
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics = json.load(f)

            # None values should be filtered out
            assert "accuracy_dip" not in metrics
            assert "recovery_time" not in metrics
            assert "best_acc" in metrics
            assert "seeds_activated" in metrics

    def test_export_metrics_for_dvc_creates_results_directory(self):
        """Test that metrics export creates results directory if it doesn't exist."""
        with temporary_directory() as temp_dir:
            # Don't create results directory - let the function create it

            final_stats = create_test_final_stats()
            slug = "test_slug"

            utils.export_metrics_for_dvc(final_stats, slug, temp_dir)

            # Check directory and file were created
            results_dir = temp_dir / "results"
            assert results_dir.exists()

            metrics_file = results_dir / f"metrics_{slug}.json"
            assert metrics_file.exists()

    def test_export_metrics_for_dvc_json_format(self):
        """Test that exported JSON is properly formatted."""
        with temporary_directory() as temp_dir:
            results_dir = temp_dir / "results"
            results_dir.mkdir()

            final_stats = create_test_final_stats()
            slug = "test_slug"

            utils.export_metrics_for_dvc(final_stats, slug, temp_dir)

            # Check JSON formatting
            metrics_file = results_dir / f"metrics_{slug}.json"
            with open(metrics_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Should be indented (pretty-printed)
            assert "  " in content or "\t" in content

            # Should be valid JSON
            metrics = json.loads(content)
            assert isinstance(metrics, dict)


class TestIntegration:
    """Integration tests for utils module functionality."""

    def test_full_experiment_workflow(self):
        """Test the full workflow of experiment utilities."""
        with temporary_directory() as temp_dir:
            results_dir = temp_dir / "results"
            results_dir.mkdir()

            # Create experiment setup
            args = create_mock_args(problem_type="spirals")
            device = "cpu"

            # Generate slug and config
            slug = utils.generate_experiment_slug(args)
            config = utils.create_experiment_config(args, device)

            # Create log file
            log_file = temp_dir / f"{slug}.log"
            with open(log_file, "w", encoding="utf-8") as log_f:
                utils.write_experiment_log_header(log_f, config, args)

                # Simulate some experiment data
                log_f.write("0,seed_0,dormant,0.0\n")
                log_f.write("1,seed_0,active,0.3\n")
                log_f.write("2,seed_1,active,0.7\n")

                # Write footer
                final_stats = create_test_final_stats()
                seed_manager = create_configured_seed_manager()
                utils.write_experiment_log_footer(log_f, final_stats, seed_manager)

            # Export metrics
            utils.export_metrics_for_dvc(final_stats, slug, temp_dir)

            # Verify everything was created correctly
            assert log_file.exists()

            metrics_file = results_dir / f"metrics_{slug}.json"
            assert metrics_file.exists()

            # Verify log content
            with open(log_file, "r", encoding="utf-8") as f:
                log_content = f.read()
            assert_log_file_format(log_content)

            # Verify metrics content
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            assert_valid_metrics_json(metrics)

    @patch("morphogenetic_engine.utils.clear_seed_report_cache")
    def test_log_header_clears_seed_cache(self, mock_clear_cache):
        """Test that log header writing clears the seed report cache."""
        with temporary_file(suffix=".log") as log_path:
            config = create_test_experiment_config()
            args = create_mock_args()

            with open(log_path, "w", encoding="utf-8") as log_f:
                utils.write_experiment_log_header(log_f, config, args)

            mock_clear_cache.assert_called_once()
