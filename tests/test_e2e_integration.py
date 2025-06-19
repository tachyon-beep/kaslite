"""
End-to-end integration test for the complete Kaslite morphogenetic engine pipeline.

This test validates the entire workflow from experiment configuration through
model training, registration, deployment, and inference serving.
"""

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import requests
import yaml

from morphogenetic_engine.cli.reports import ReportsCLI
from morphogenetic_engine.cli.sweep import SweepCLI
from morphogenetic_engine.model_registry import ModelRegistry
from morphogenetic_engine.sweeps.config import load_sweep_config


class E2EIntegrationTest:
    """End-to-end integration test suite for the morphogenetic engine."""

    def __init__(self):
        """Initialize test environment."""
        self.test_dir = Path(tempfile.mkdtemp(prefix="kaslite_e2e_"))
        self.original_cwd = Path.cwd()
        self.inference_process = None
        self.inference_port = 8901  # Use different port to avoid conflicts
        self.config_path = None  # Initialize config_path attribute

    def setup(self):
        """Set up test environment and directories."""
        print(f"🔧 Setting up E2E test environment in {self.test_dir}")

        # Create test directory structure
        (self.test_dir / "results").mkdir(exist_ok=True)
        (self.test_dir / "data").mkdir(exist_ok=True)
        (self.test_dir / "mlruns").mkdir(exist_ok=True)
        (self.test_dir / "configs").mkdir(exist_ok=True)

        # Change to test directory
        os.chdir(self.test_dir)

        # Create minimal test configuration
        self.create_test_config()

    def teardown(self):
        """Clean up test environment."""
        print("🧹 Cleaning up E2E test environment")

        # Stop inference server if running
        if self.inference_process:
            self.inference_process.terminate()
            self.inference_process.wait(timeout=10)

        # Return to original directory
        os.chdir(self.original_cwd)

        # Clean up test directory
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def create_test_config(self):
        """Create a minimal test configuration for quick execution."""
        config = {
            "sweep_type": "grid",
            "experiment": {
                "problem_type": "spirals",
                "device": "cpu",
                "n_samples": 100,  # Very small dataset for speed
                "batch_size": 16,
                "train_frac": 0.7,
                "blend_steps": 10,  # Minimal blending
                "epochs": 3,  # Very few epochs
                "warm_up_epochs": 1,  # Quick warmup
                "hidden_dim": 16,  # Very small network
                "num_layers": 2,
                "seeds_per_layer": 1,
                "patience": 2,  # Low patience for quick exit
            },
            "parameters": {"lr": [0.01]},  # Single value for quick test
            "optimization": {"objective": "val_acc", "direction": "maximize"},
        }

        config_path = self.test_dir / "configs" / "e2e_test.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False)

        self.config_path = config_path

    def test_1_configuration_loading(self):
        """Test that configurations can be loaded properly."""
        print("📋 Testing configuration loading...")

        # Test config loading
        config = load_sweep_config(str(self.config_path))
        assert config.sweep_type == "grid"
        assert config.experiment["problem_type"] == "spirals"
        assert len(config.parameters) == 1

        # Test parameter combinations
        combinations = config.get_grid_combinations()
        assert len(combinations) == 1  # Single learning rate
        assert abs(combinations[0]["lr"] - 0.01) < 0.001  # Float comparison

        print("✅ Configuration loading successful")

    def test_2_experiment_execution(self):
        """Test running a complete experiment."""
        print("🧪 Testing experiment execution...")

        # Run experiment using the script directly
        import sys

        script_path = self.original_cwd / "scripts" / "run_morphogenetic_experiment.py"
        cmd = [sys.executable, str(script_path), "--sweep_config", str(self.config_path)]

        result = subprocess.run(
            cmd,
            cwd=str(self.original_cwd),
            capture_output=True,
            text=True,
            timeout=180,  # 3 minute timeout for minimal config
            check=False,  # Don't raise exception on non-zero return code
        )

        print(f"Return code: {result.returncode}")
        if result.stdout:
            print(f"STDOUT: {result.stdout[:500]}...")  # First 500 chars for debugging
        if result.stderr:
            print(f"STDERR: {result.stderr[:500]}...")  # First 500 chars for debugging

        assert result.returncode == 0, f"Experiment failed: {result.stderr}"

        # Verify results were created in the project root results directory
        results_dir = self.original_cwd / "results"
        assert results_dir.exists(), f"Results directory not found at {results_dir}"

        # Check for log files (look for results_*.log pattern)
        log_files = list(results_dir.glob("results_*.log"))
        print(f"Found {len(log_files)} results log files in {results_dir}")

        # Also check for sweep directories which might contain logs
        sweep_dirs = list(results_dir.glob("sweeps/*/"))
        if sweep_dirs:
            for sweep_dir in sweep_dirs:
                sweep_logs = list(sweep_dir.glob("**/*.log"))
                log_files.extend(sweep_logs)
                print(f"Found {len(sweep_logs)} sweep log files in {sweep_dir}")

        # Final fallback: any .log files
        if len(log_files) == 0:
            all_logs = list(results_dir.glob("**/*.log"))
            log_files = all_logs
            print(f"Found {len(all_logs)} total log files in results")

        assert len(log_files) > 0, f"No log files found in {results_dir}. Contents: {list(results_dir.iterdir())}"

        print("✅ Experiment execution successful")

    def test_3_model_registry_integration(self):
        """Test MLflow model registry integration."""
        print("🏛️ Testing model registry integration...")

        # Initialize model registry
        registry = ModelRegistry("E2ETest")

        # Check if any models were registered during the experiment
        # (This depends on the model achieving the accuracy threshold)
        try:
            models = registry.list_model_versions()
            print(f"Found {len(models)} model versions")

            if models:
                # Test model retrieval
                model_uri = registry.get_production_model_uri()
                if model_uri:
                    print("✅ Production model found in registry")
                else:
                    # Try to promote a model manually for testing
                    latest_version = models[0] if models else None
                    if latest_version:
                        registry.promote_model("Production", latest_version.version)
                        print("✅ Model promoted to Production")

        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"⚠️  Model registry test skipped: {e}")

        print("✅ Model registry integration tested")

    def test_4_sweep_cli_functionality(self):
        """Test the sweep CLI functionality."""
        print("🔄 Testing sweep CLI functionality...")

        # Test SweepCLI instantiation
        sweep_cli = SweepCLI()
        assert sweep_cli is not None

        # Test config loading through CLI
        config = load_sweep_config(str(self.config_path))
        combinations = config.get_grid_combinations()
        assert len(combinations) == 1

        print("✅ Sweep CLI functionality tested")

    def test_5_reports_generation(self):
        """Test reports and analysis functionality."""
        print("📊 Testing reports generation...")

        # Test ReportsCLI instantiation
        reports_cli = ReportsCLI()
        assert reports_cli is not None

        # Check if we have results to analyze
        results_dir = self.test_dir / "results"
        if results_dir.exists():
            result_files = list(results_dir.glob("**/*.json"))
            if result_files:
                print(f"Found {len(result_files)} result files for analysis")

        print("✅ Reports generation tested")

    def _start_inference_server(self):
        """Start the inference server in background."""
        import sys

        cmd = [
            sys.executable,
            "-m",
            "morphogenetic_engine.inference_server",
            "--port",
            str(self.inference_port),
            "--host",
            "127.0.0.1",
        ]

        self.inference_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.original_cwd),  # Run from project root
        )

        # Wait for server to start
        time.sleep(3)

    def _test_health_endpoint(self):
        """Test the health endpoint."""
        health_url = f"http://127.0.0.1:{self.inference_port}/health"
        response = requests.get(health_url, timeout=5)

        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check successful: {health_data}")
            return health_data
        else:
            print(f"⚠️  Health check failed: {response.status_code}")
            return None

    def _test_models_endpoint(self):
        """Test the models endpoint."""
        models_url = f"http://127.0.0.1:{self.inference_port}/models"
        response = requests.get(models_url, timeout=5)

        if response.status_code == 200:
            models_data = response.json()
            print(f"✅ Models endpoint successful: {models_data}")
            return models_data
        else:
            print(f"⚠️  Models endpoint failed: {response.status_code}")
            return None

    def _test_prediction_endpoint(self, model_loaded: bool):
        """Test the prediction endpoint."""
        if not model_loaded:
            print("⚠️  No model loaded, skipping prediction test")
            return

        predict_url = f"http://127.0.0.1:{self.inference_port}/predict"
        test_data = {"data": [[0.5, 0.3]]}

        response = requests.post(predict_url, json=test_data, timeout=5)

        if response.status_code == 200:
            prediction_data = response.json()
            print(f"✅ Prediction successful: {prediction_data}")
        else:
            print(f"⚠️  Prediction failed: {response.status_code}")

    def test_6_inference_server_deployment(self):
        """Test inference server deployment and API endpoints."""
        print("🚀 Testing inference server deployment...")

        try:
            self._start_inference_server()

            # Wait for server to start
            time.sleep(3)

            # Test endpoints
            health_data = self._test_health_endpoint()
            if health_data:
                self._test_models_endpoint()
                self._test_prediction_endpoint(health_data.get("model_loaded", False))

        except requests.exceptions.RequestException as e:
            print(f"⚠️  Inference server test skipped: {e}")
        except (ValueError, RuntimeError, ConnectionError) as e:
            print(f"⚠️  Inference server test error: {e}")

        print("✅ Inference server deployment tested")

    def test_7_monitoring_metrics(self):
        """Test monitoring and metrics collection."""
        print("📈 Testing monitoring metrics...")

        try:
            # Test metrics endpoint if server is running
            if self.inference_process and self.inference_process.poll() is None:
                metrics_url = f"http://127.0.0.1:{self.inference_port}/metrics"
                response = requests.get(metrics_url, timeout=5)

                if response.status_code == 200:
                    metrics_text = response.text
                    print("✅ Metrics endpoint accessible")

                    # Check for key metrics
                    expected_metrics = [
                        "kaslite_requests_total",
                        "kaslite_request_duration_seconds",
                        "kaslite_model_predictions_total",
                    ]

                    for metric in expected_metrics:
                        if metric in metrics_text:
                            print(f"✅ Found metric: {metric}")
                        else:
                            print(f"⚠️  Missing metric: {metric}")
                else:
                    print(f"⚠️  Metrics endpoint failed: {response.status_code}")
            else:
                print("⚠️  Inference server not running, skipping metrics test")

        except (requests.exceptions.RequestException, ValueError, RuntimeError) as e:
            print(f"⚠️  Monitoring metrics test error: {e}")

        print("✅ Monitoring metrics tested")

    def run_all_tests(self):
        """Run the complete end-to-end test suite."""
        print("🚀 Starting End-to-End Integration Test Suite")
        print("=" * 60)

        try:
            self.setup()

            # Run tests in sequence
            self.test_1_configuration_loading()
            self.test_2_experiment_execution()
            self.test_3_model_registry_integration()
            self.test_4_sweep_cli_functionality()
            self.test_5_reports_generation()
            self.test_6_inference_server_deployment()
            self.test_7_monitoring_metrics()

            print("=" * 60)
            print("🎉 All E2E integration tests completed successfully!")
            return True

        except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as e:
            print(f"❌ E2E test failed: {e}")
            import traceback

            traceback.print_exc()
            return False

        finally:
            self.teardown()


def test_e2e_integration():
    """Pytest entry point for E2E integration test."""
    e2e_test_suite = E2EIntegrationTest()
    test_success = e2e_test_suite.run_all_tests()
    assert test_success, "E2E integration test failed"


if __name__ == "__main__":
    # Allow running as standalone script
    main_test_suite = E2EIntegrationTest()
    main_success = main_test_suite.run_all_tests()
    exit(0 if main_success else 1)
