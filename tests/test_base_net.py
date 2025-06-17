"""Tests for BaseNet and multi-seed functionality."""

# pylint: disable=protected-access,redefined-outer-name

from typing import cast

import pytest
import torch
import torch.testing  # Added for torch.testing.assert_close

from morphogenetic_engine.components import BaseNet, SentinelSeed
from morphogenetic_engine.core import SeedManager


@pytest.fixture
def seed_manager() -> SeedManager:
    """Fixture for creating a SeedManager instance."""
    return SeedManager()


@pytest.fixture
def default_base_net(seed_manager: SeedManager) -> BaseNet:
    """Fixture for a BaseNet instance with default parameters."""
    return BaseNet(seed_manager=seed_manager, input_dim=2)


@pytest.fixture
def base_net_custom_params(seed_manager: SeedManager) -> BaseNet:
    """Fixture for BaseNet with specific custom parameters for testing."""
    return BaseNet(
        hidden_dim=128,
        seed_manager=seed_manager,
        input_dim=2,
        blend_steps=20,
        shadow_lr=5e-4,
        progress_thresh=0.8,
        drift_warn=0.05,
    )


class TestBaseNet:
    """Test suite for BaseNet class with single seed per layer."""

    def test_initialization_default_params(self, mocker):  # Changed: uses mocker
        """Test BaseNet initialization with default parameters."""
        # Arrange
        mock_seed_manager = mocker.MagicMock(spec=SeedManager)

        # Act
        net = BaseNet(seed_manager=mock_seed_manager, input_dim=2)

        # Assert
        assert net.hidden_dim == 64
        assert net.num_layers == 8
        assert net.seeds_per_layer == 1
        assert net.input_layer.in_features == 2
        assert net.input_layer.out_features == 64
        assert len(net.layers) == 8
        assert net.get_total_seeds() == 8

        for i, seed_module in enumerate(net.all_seeds):
            seed = cast(SentinelSeed, seed_module)
            assert isinstance(seed, SentinelSeed)
            assert seed.seed_id == f"seed{i+1}_1"
            assert seed.dim == net.hidden_dim
            assert seed.seed_manager is mock_seed_manager  # Check mock is used

    def test_initialization_custom_params(self, mocker):  # Changed: uses mocker
        """Test BaseNet with custom parameters using a fixture and mocked SeedManager."""
        # Arrange
        mock_seed_manager = mocker.MagicMock(spec=SeedManager)
        custom_params = {
            "hidden_dim": 128,
            "input_dim": 2,
            "blend_steps": 20,
            "shadow_lr": 5e-4,
            "progress_thresh": 0.8,
            "drift_warn": 0.05,
        }

        # Act
        net = BaseNet(seed_manager=mock_seed_manager, **custom_params)

        # Assert
        assert net.hidden_dim == 128
        assert len(net.all_seeds) > 0, "Net should have seeds"
        seed = cast(SentinelSeed, net.all_seeds[0])  # Assuming at least one seed
        assert seed.dim == 128
        assert seed.blend_steps == 20
        assert seed.shadow_lr == pytest.approx(5e-4)
        assert seed.progress_thresh == pytest.approx(0.8)
        assert seed.drift_warn == pytest.approx(0.05)
        assert seed.seed_manager is mock_seed_manager

    def test_freeze_backbone(self, default_base_net: BaseNet):
        """Test backbone freezing functionality."""
        # Arrange
        net = default_base_net

        for seed_module in net.all_seeds:
            seed = cast(SentinelSeed, seed_module)
            seed.initialize_child()

        # Act
        net.freeze_backbone()

        # Assert
        for name, param in net.named_parameters():
            if "seed" not in name:  # Backbone
                assert not param.requires_grad, f"Backbone parameter {name} should be frozen"
            elif "child" in name:  # Seed's internal network
                assert param.requires_grad, f"Seed parameter {name} should remain trainable"
            # Other seed parameters (like alpha, etc.) might not have requires_grad or are not trained directly

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_forward_pass_and_shapes(self, default_base_net: BaseNet, batch_size: int):
        """Test forward pass with various batch sizes and deterministic behavior."""
        # Arrange
        net = default_base_net
        x = torch.randn(batch_size, net.input_layer.in_features)

        # Act
        output = net.forward(x)

        # Assert
        assert output.shape == (batch_size, net.out.out_features)

        # Test deterministic behavior in eval mode
        net.eval()
        x_eval = torch.randn(4, net.input_layer.in_features)
        torch.manual_seed(42)  # Set seed for deterministic comparison

        # Act & Assert for determinism
        with torch.no_grad():
            output1 = net.forward(x_eval)
            output2 = net.forward(x_eval)
        assert torch.allclose(output1, output2), "Forward pass should be deterministic in eval mode"

    def test_gradient_flow_dormant_seeds(self, default_base_net: BaseNet):
        """Test gradient flow when seeds are in dormant state (identity function)."""
        # Arrange
        net = default_base_net
        # Keep seeds in dormant state (default)
        x = torch.randn(4, net.input_layer.in_features, requires_grad=True)

        # Act
        output = net.forward(x)
        loss = output.sum()
        loss.backward()

        # Assert
        assert x.grad is not None, "Input tensor should have gradients"
        assert not torch.allclose(
            x.grad, torch.zeros_like(x.grad)
        ), "Input gradients should not be all zero"

        # Check that backbone parameters (input_layer, layers, out) receive gradients
        backbone_params_with_grads = 0
        seed_params_with_grads = 0

        for name, param in net.named_parameters():
            if param.requires_grad:
                if "seed" not in name:  # Backbone parameters
                    assert (
                        param.grad is not None
                    ), f"Backbone parameter {name} should have gradients"
                    assert not torch.allclose(
                        param.grad, torch.zeros_like(param.grad)
                    ), f"Backbone gradient for {name} should not be all zero"
                    backbone_params_with_grads += 1
                else:  # Seed parameters
                    # Dormant seeds should not receive gradients since they don't participate in forward pass
                    if param.grad is not None and not torch.allclose(
                        param.grad, torch.zeros_like(param.grad)
                    ):
                        seed_params_with_grads += 1

        assert backbone_params_with_grads > 0, "Backbone parameters should receive gradients"
        # In dormant state, seed parameters should not receive meaningful gradients
        assert seed_params_with_grads == 0, "Dormant seed parameters should not receive gradients"

    def test_gradient_flow_active_seeds(self, default_base_net: BaseNet):
        """Test gradient flow when seeds are in active/training state."""
        # Arrange
        net = default_base_net
        # Initialize and activate some seeds
        for seed_module in net.all_seeds:
            seed = cast(SentinelSeed, seed_module)
            seed.initialize_child()  # Sets to training state
            # Manually activate seed to ensure it participates in forward pass
            seed._set_state("active")

        x = torch.randn(4, net.input_layer.in_features, requires_grad=True)

        # Act
        output = net.forward(x)
        loss = output.sum()
        loss.backward()

        # Assert
        assert x.grad is not None, "Input tensor should have gradients"
        assert not torch.allclose(
            x.grad, torch.zeros_like(x.grad)
        ), "Input gradients should not be all zero"

        # Check that both backbone and seed parameters receive gradients
        backbone_params_with_grads = 0
        seed_child_params_with_grads = 0

        for name, param in net.named_parameters():
            if param.requires_grad:
                if "seed" not in name:  # Backbone parameters
                    assert (
                        param.grad is not None
                    ), f"Backbone parameter {name} should have gradients"
                    backbone_params_with_grads += 1
                elif "child" in name:  # Seed child network parameters
                    # Active seeds should receive gradients since they participate in forward pass
                    assert (
                        param.grad is not None
                    ), f"Active seed parameter {name} should have gradients"
                    assert not torch.allclose(
                        param.grad, torch.zeros_like(param.grad)
                    ), f"Active seed gradient for {name} should not be all zero"
                    seed_child_params_with_grads += 1

        assert backbone_params_with_grads > 0, "Backbone parameters should receive gradients"
        assert (
            seed_child_params_with_grads > 0
        ), "Active seed child parameters should receive gradients"

    def test_seed_integration_and_independence(self, default_base_net: BaseNet):
        """Test seed integration and that seed instances are distinct."""
        # Arrange
        net = default_base_net
        assert len(net.all_seeds) > 0, "Net should have seeds"

        first_seed = cast(SentinelSeed, net.all_seeds[0])
        first_seed.initialize_child()

        x = torch.randn(4, net.input_layer.in_features)

        # Act
        output = net.forward(x)

        # Assert
        assert output.shape == (4, net.out.out_features)
        assert first_seed.seed_id == "seed1_1"  # Based on default naming

        # Test seed independence (instances are distinct)
        all_seeds_list = net.get_all_seeds()
        for i, seed1_module in enumerate(all_seeds_list):
            for j, seed2_module in enumerate(all_seeds_list):
                if i != j:
                    seed1 = cast(SentinelSeed, seed1_module)
                    seed2 = cast(SentinelSeed, seed2_module)
                    assert (
                        seed1 is not seed2
                    ), f"Seeds at index {i} and {j} should be different instances"
                    assert (
                        seed1.seed_id != seed2.seed_id
                    ), f"Seed IDs {seed1.seed_id} and {seed2.seed_id} should be unique"

    def test_architecture_consistency(self, default_base_net: BaseNet):
        """Test network architecture consistency."""
        # Arrange
        net = default_base_net
        hidden_dim = net.hidden_dim
        input_dim = net.input_layer.in_features
        output_dim = net.out.out_features

        # Act & Assert
        assert net.input_layer.in_features == input_dim
        assert net.input_layer.out_features == hidden_dim
        assert net.out.in_features == hidden_dim
        assert net.out.out_features == output_dim

        for layer in net.layers:
            assert layer.in_features == hidden_dim
            assert layer.out_features == hidden_dim

        for seed_module in net.get_all_seeds():
            seed = cast(SentinelSeed, seed_module)
            assert seed.dim == hidden_dim
        assert net.get_total_seeds() == net.num_layers * net.seeds_per_layer


class TestBaseNetInitializationEdgeCases:
    """Tests for BaseNet initialization with edge case or invalid parameters."""

    def test_invalid_dimensions_or_layers(self, seed_manager: SeedManager):
        """Test BaseNet instantiation with invalid dimension or layer counts."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="num_layers must be positive"):
            BaseNet(seed_manager=seed_manager, input_dim=2, num_layers=0)
        with pytest.raises(ValueError, match="num_layers must be positive"):
            BaseNet(seed_manager=seed_manager, input_dim=2, num_layers=-1)
        with pytest.raises(ValueError, match="input_dim must be positive"):
            BaseNet(seed_manager=seed_manager, input_dim=0)
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            BaseNet(seed_manager=seed_manager, input_dim=2, hidden_dim=0)

    def test_seeds_per_layer_handling(self, seed_manager: SeedManager):
        """Test how BaseNet handles non-positive seeds_per_layer."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="seeds_per_layer must be positive"):
            BaseNet(seed_manager=seed_manager, input_dim=2, seeds_per_layer=0)

        with pytest.raises(ValueError, match="seeds_per_layer must be positive"):
            BaseNet(seed_manager=seed_manager, input_dim=2, seeds_per_layer=-5)

    def test_missing_seed_manager(self):
        """Test BaseNet instantiation without a SeedManager (should be TypeError)."""
        # Arrange & Act & Assert
        with pytest.raises(TypeError):  # Due to missing required keyword argument
            BaseNet(input_dim=2)  # pylint: disable=missing-kwoa # type: ignore[call-arg]


# Parametrization for comprehensive tests (including large configs for architectural testing)
@pytest.mark.parametrize(
    "num_layers, seeds_per_layer, expected_total_seeds, expected_ids_pattern",
    [
        # Original test cases
        (2, 3, 6, ["seed1_1", "seed1_2", "seed1_3", "seed2_1", "seed2_2", "seed2_3"]),
        (3, 1, 3, ["seed1_1", "seed2_1", "seed3_1"]),  # Single seed compatibility
        (1, 5, 5, ["seed1_1", "seed1_2", "seed1_3", "seed1_4", "seed1_5"]),  # Extreme config
        # Additional edge cases consolidated from TestBaseNetExtended
        (1, 1, 1, ["seed1_1"]),  # Minimal configuration
        (
            1,
            10,
            10,
            [
                "seed1_1",
                "seed1_2",
                "seed1_3",
                "seed1_4",
                "seed1_5",
                "seed1_6",
                "seed1_7",
                "seed1_8",
                "seed1_9",
                "seed1_10",
            ],
        ),  # Single layer, many seeds
        (20, 5, 100, None),  # Large configuration for architectural testing
    ],
)
class TestMultiSeedBaseNet:
    """Test BaseNet with multiple seeds per layer using comprehensive parameterization."""

    @pytest.fixture
    def multi_seed_net(
        self, seed_manager: SeedManager, num_layers: int, seeds_per_layer: int
    ) -> BaseNet:
        """Fixture for BaseNet with varying multi-seed configurations."""
        # Use larger hidden_dim for large configurations to test scalability
        hidden_dim = 128 if num_layers >= 20 else 32
        return BaseNet(
            hidden_dim=hidden_dim,
            seed_manager=seed_manager,
            input_dim=3,  # Using 3 to differentiate from default tests
            num_layers=num_layers,
            seeds_per_layer=seeds_per_layer,
        )

    def test_comprehensive_multi_seed_architecture(
        self,
        multi_seed_net: BaseNet,
        num_layers: int,
        seeds_per_layer: int,
        expected_total_seeds: int,
        expected_ids_pattern: list[str],
    ):
        """Comprehensive test covering initialization, structure, forward pass, and behavior."""
        # Arrange
        net = multi_seed_net

        # Test initialization and structure
        assert net.num_layers == num_layers
        assert net.seeds_per_layer == seeds_per_layer
        assert net.get_total_seeds() == expected_total_seeds

        # Test seed ID patterns (skip for very large configs)
        if expected_ids_pattern is not None:
            actual_ids = [cast(SentinelSeed, seed).seed_id for seed in net.all_seeds]
            assert actual_ids == expected_ids_pattern

        # Test get_seeds_for_layer functionality
        for i in range(num_layers):
            layer_seeds = net.get_seeds_for_layer(i)
            assert len(layer_seeds) == seeds_per_layer
            for j, seed_module in enumerate(layer_seeds):
                seed = cast(SentinelSeed, seed_module)
                assert seed.seed_id == f"seed{i+1}_{j+1}"

        # Test forward pass shape correctness
        batch_size = 4
        x = torch.randn(batch_size, net.input_layer.in_features)
        output = net.forward(x)
        assert output.shape == (batch_size, net.out.out_features)

        # Test seed independence for reasonable sized configs
        if expected_total_seeds <= 50:  # Only test independence for smaller configs
            all_seeds = [cast(SentinelSeed, s) for s in net.all_seeds]
            for i, seed1 in enumerate(all_seeds):
                for j, seed2 in enumerate(all_seeds):
                    if i != j:
                        assert (
                            seed1 is not seed2
                        ), f"Seeds at index {i} and {j} should be different instances"
                        assert seed1.seed_id != seed2.seed_id, "Seed IDs should be unique"

    def test_get_seeds_for_layer_out_of_bounds(
        self,
        multi_seed_net: BaseNet,
        num_layers: int,
        seeds_per_layer: int,
        expected_total_seeds: int,
        expected_ids_pattern: list[str],
    ):
        """Test get_seeds_for_layer with out-of-bounds layer indices."""
        # Arrange
        net = multi_seed_net
        # Consume parameters for pytest compatibility
        _ = seeds_per_layer, expected_total_seeds, expected_ids_pattern

        if num_layers == 0:  # Can't test out of bounds if there are no bounds
            pytest.skip("Test not applicable for zero layers")

        # Act & Assert
        with pytest.raises(IndexError):
            net.get_seeds_for_layer(-1)
        with pytest.raises(IndexError):
            net.get_seeds_for_layer(num_layers)  # Max valid index is num_layers - 1
        with pytest.raises(IndexError):
            net.get_seeds_for_layer(num_layers + 10)


class TestSeedAveraging:
    """Test suite for seed averaging behavior (not parameterized)."""

    def test_seed_averaging_behavior_forward_pass(self, seed_manager: SeedManager):
        """
        Test forward pass when multiple seeds are in a layer.
        This test focuses on the pass succeeding, not the exact averaging math.
        """
        net = BaseNet(
            hidden_dim=32,
            seed_manager=seed_manager,
            input_dim=2,
            num_layers=1,  # Single layer for simplicity
            seeds_per_layer=3,
        )

        seeds = [cast(SentinelSeed, s) for s in net.get_seeds_for_layer(0)]
        assert len(seeds) == 3

        # Set seeds to different states to ensure forward handles them
        seeds[0].initialize_child()  # -> training
        seeds[1].initialize_child()
        seeds[1]._set_state("blending")  # Manually set for testing
        seeds[1].alpha = 0.5
        seeds[2]._set_state("active")  # Manually set for testing

        x = torch.randn(4, net.input_layer.in_features)
        # The forward pass should average the outputs of these seeds
        output = net.forward(x)
        assert output.shape == (4, net.out.out_features)

        # Verify each seed was accessed (e.g. by checking their state or buffer if applicable)
        # This is implicitly tested by the forward pass not crashing and by seed state setup.
        # A more direct check could involve mocking seed forward methods if detailed interaction is needed.
        for seed in seeds:
            assert seed.state in [
                "training",
                "blending",
                "active",
            ], f"Seed {seed.seed_id} has unexpected state {seed.state}"


class TestBaseNetArchitecturalProperties:
    """Test BaseNet architectural properties, scaling, and edge cases."""

    def test_parameter_scaling(self, seed_manager: SeedManager):
        """Test parameter count scaling with architecture size."""

        # Arrange
        def count_parameters(net):
            return sum(p.numel() for p in net.parameters())

        net1 = BaseNet(
            seed_manager=seed_manager, input_dim=2, num_layers=2, seeds_per_layer=1, hidden_dim=4
        )

        net2 = BaseNet(
            seed_manager=seed_manager, input_dim=2, num_layers=2, seeds_per_layer=3, hidden_dim=4
        )

        # Act
        params1 = count_parameters(net1)
        params2 = count_parameters(net2)

        # Assert
        assert params2 > params1, "More seeds should result in more parameters"

    @pytest.mark.parametrize("input_dim", [1, 2, 3, 5, 10])
    def test_various_input_dimensions(self, seed_manager: SeedManager, input_dim: int):
        """Test networks with different input dimensions."""
        # Arrange
        net = BaseNet(
            seed_manager=seed_manager, input_dim=input_dim, num_layers=2, seeds_per_layer=2
        )

        # Act
        x = torch.randn(4, input_dim)
        output = net(x)

        # Assert
        assert output.shape == (
            4,
            2,
        ), f"Output should always be 2D regardless of input_dim={input_dim}"

    def test_seed_output_averaging_correctness(self, seed_manager: SeedManager, mocker):
        """Test that BaseNet correctly averages outputs from multiple seeds in a layer."""
        # Arrange
        input_dim = 2
        hidden_dim = 4
        num_layers = 1
        seeds_per_layer = 3
        batch_size = 2

        net = BaseNet(
            seed_manager=seed_manager,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            seeds_per_layer=seeds_per_layer,
        )
        net.eval()

        layer_idx_to_test = 0
        seeds_in_layer = net.get_seeds_for_layer(layer_idx_to_test)
        assert len(seeds_in_layer) == seeds_per_layer

        # Set seeds to active state and mock their forward methods
        mock_outputs = []
        for i, seed_module in enumerate(seeds_in_layer):
            seed = cast(SentinelSeed, seed_module)
            # Set to active state so the seed actually processes input
            seed._set_state("active")
            mock_output = torch.ones(batch_size, hidden_dim) * (i + 1.0)
            mock_outputs.append(mock_output)
            mocker.patch.object(seed, "forward", return_value=mock_output)

        # Expected average of the mock outputs
        expected_average = torch.stack(mock_outputs).mean(dim=0)

        # Capture the averaged output going to the next layer
        captured_averaged_output = None

        def capture_hook(_module, input_args, _output_args):
            nonlocal captured_averaged_output
            # The input to the next layer should be the averaged seed output
            captured_averaged_output = input_args[0].clone()

        # Hook the next layer (or output layer if this is the last)
        if layer_idx_to_test < len(net.layers) - 1:
            hook_target = net.layers[layer_idx_to_test + 1]
        else:
            hook_target = net.out

        handle = hook_target.register_forward_hook(capture_hook)

        # Act
        test_input = torch.randn(batch_size, input_dim)
        _ = net.forward(test_input)
        handle.remove()

        # Assert - the hook captured the averaged output
        torch.testing.assert_close(
            captured_averaged_output,
            expected_average,
            msg="Averaged seed output is not mathematically correct",
        )


# Integration tests with different approach - moved from TestBaseNetExtended
class TestBaseNetIntegration:
    """Integration tests for BaseNet - moved outside individual test classes."""

    def test_seed_averaging_behavior_forward_pass(self, seed_manager: SeedManager):
        """
        Test forward pass when multiple seeds are in a layer.
        This test focuses on the pass succeeding, not the exact averaging math.
        """
        net = BaseNet(
            hidden_dim=32,
            seed_manager=seed_manager,
            input_dim=2,
            num_layers=1,  # Single layer for simplicity
            seeds_per_layer=3,
        )

        seeds = [cast(SentinelSeed, s) for s in net.get_seeds_for_layer(0)]
        assert len(seeds) == 3

        # Set seeds to different states to ensure forward handles them
        seeds[0].initialize_child()  # -> training
        seeds[1].initialize_child()
        seeds[1]._set_state("blending")  # Manually set for testing
        seeds[1].alpha = 0.5
        seeds[2]._set_state("active")  # Manually set for testing

        x = torch.randn(4, net.input_layer.in_features)
        # The forward pass should average the outputs of these seeds
        output = net.forward(x)
        assert output.shape == (4, net.out.out_features)

        # Verify each seed was accessed (e.g. by checking their state or buffer if applicable)
        # This is implicitly tested by the forward pass not crashing and by seed state setup.
        # A more direct check could involve mocking seed forward methods if detailed interaction is needed.
        for seed in seeds:
            assert seed.state in [
                "training",
                "blending",
                "active",
            ], f"Seed {seed.seed_id} has unexpected state {seed.state}"


# Separate parametrization for gradient flow tests (excluding large configs that suffer from vanishing gradients)
@pytest.mark.parametrize(
    "num_layers, seeds_per_layer, expected_total_seeds, expected_ids_pattern",
    [
        # Only smaller configurations suitable for gradient flow testing
        # Large networks (100+ seeds, 20+ layers) are excluded because they suffer from
        # vanishing gradient problems that make gradient flow tests unreliable
        (2, 3, 6, ["seed1_1", "seed1_2", "seed1_3", "seed2_1", "seed2_2", "seed2_3"]),
        (3, 1, 3, ["seed1_1", "seed2_1", "seed3_1"]),  # Single seed compatibility
        (1, 5, 5, ["seed1_1", "seed1_2", "seed1_3", "seed1_4", "seed1_5"]),  # Extreme config
        (1, 1, 1, ["seed1_1"]),  # Minimal configuration
        (
            1,
            10,
            10,
            [
                "seed1_1",
                "seed1_2",
                "seed1_3",
                "seed1_4",
                "seed1_5",
                "seed1_6",
                "seed1_7",
                "seed1_8",
                "seed1_9",
                "seed1_10",
            ],
        ),  # Single layer, many seeds - still manageable for gradient flow
    ],
)
class TestMultiSeedBaseNetGradientFlow:
    """Test gradient flow behavior for BaseNet with multiple seeds per layer.

    Uses smaller configurations to avoid vanishing gradient issues that make
    gradient flow tests unreliable in very large networks.
    """

    @pytest.fixture
    def multi_seed_net(
        self, seed_manager: SeedManager, num_layers: int, seeds_per_layer: int
    ) -> BaseNet:
        """Fixture for BaseNet with varying multi-seed configurations for gradient flow testing."""
        return BaseNet(
            hidden_dim=32,  # Smaller hidden_dim for gradient flow testing
            seed_manager=seed_manager,
            input_dim=3,
            num_layers=num_layers,
            seeds_per_layer=seeds_per_layer,
        )

    def test_gradient_flow_dormant_seeds(
        self,
        multi_seed_net: BaseNet,
        num_layers: int,
        seeds_per_layer: int,
        expected_total_seeds: int,
        expected_ids_pattern: list[str],
    ):
        """Test gradient flow with seeds in dormant state."""
        # Arrange
        net = multi_seed_net
        # Consume parameters for pytest compatibility
        _ = num_layers, expected_total_seeds, expected_ids_pattern

        if seeds_per_layer == 0:
            pytest.skip("Test not applicable for zero seeds per layer")

        # Keep seeds in dormant state (default)
        x = torch.randn(4, net.input_layer.in_features, requires_grad=True)

        # Act
        output = net.forward(x)
        loss = output.sum()
        loss.backward()

        # Assert
        assert x.grad is not None, "Input tensor should have gradients"
        # Use a more reasonable tolerance for large networks where gradients can be very small
        assert not torch.allclose(
            x.grad, torch.zeros_like(x.grad), atol=1e-8
        ), "Input gradients should not be all zero"

        # Check backbone parameters receive gradients, seed parameters should not
        backbone_params_with_grads = 0
        seed_params_with_grads = 0

        for name, param in net.named_parameters():
            if param.requires_grad:
                if "seed" not in name:  # Backbone parameters
                    assert (
                        param.grad is not None
                    ), f"Backbone parameter {name} should have gradients"
                    assert not torch.allclose(
                        param.grad, torch.zeros_like(param.grad), atol=1e-8
                    ), f"Backbone gradient for {name} should not be all zero"
                    backbone_params_with_grads += 1
                else:  # Seed parameters
                    # Dormant seeds should not receive meaningful gradients
                    if param.grad is not None and not torch.allclose(
                        param.grad, torch.zeros_like(param.grad), atol=1e-8
                    ):
                        seed_params_with_grads += 1

        assert backbone_params_with_grads > 0, "Backbone parameters should receive gradients"
        assert (
            seed_params_with_grads == 0
        ), "Dormant seed parameters should not receive meaningful gradients"

    def test_gradient_flow_active_seeds(
        self,
        multi_seed_net: BaseNet,
        num_layers: int,
        seeds_per_layer: int,
        expected_total_seeds: int,
        expected_ids_pattern: list[str],
    ):
        """Test gradient flow with seeds in active state."""
        # Arrange
        net = multi_seed_net
        # Consume parameters for pytest compatibility
        _ = num_layers, expected_total_seeds, expected_ids_pattern
        if seeds_per_layer == 0:
            pytest.skip("Test not applicable for zero seeds per layer")

        # Initialize and activate some seeds (first seed of each layer)
        for i in range(net.num_layers):
            layer_seeds = net.get_seeds_for_layer(i)
            if layer_seeds:  # If there are seeds in this layer
                first_seed = cast(SentinelSeed, layer_seeds[0])
                first_seed.initialize_child()
                # Set to active state so the seed actually processes input
                first_seed.state = "active"

        x = torch.randn(4, net.input_layer.in_features, requires_grad=True)

        # Act
        output = net.forward(x)
        loss = output.sum()
        loss.backward()

        # Assert
        assert x.grad is not None, "Input tensor should have gradients"
        assert not torch.allclose(
            x.grad, torch.zeros_like(x.grad), atol=1e-8
        ), "Input gradients should not be all zero"

        # Check both backbone and activated seed parameters receive gradients
        backbone_params_with_grads = 0
        seed_child_params_with_grads = 0

        for name, param in net.named_parameters():
            if param.requires_grad:
                if "seed" not in name:  # Backbone parameters
                    assert (
                        param.grad is not None
                    ), f"Backbone parameter {name} should have gradients"
                    backbone_params_with_grads += 1
                elif "child" in name:  # Seed child network parameters
                    # Active seeds should receive gradients
                    assert (
                        param.grad is not None
                    ), f"Active seed parameter {name} should have gradients"
                    assert not torch.allclose(
                        param.grad, torch.zeros_like(param.grad), atol=1e-8
                    ), f"Active seed gradient for {name} should not be all zero"
                    seed_child_params_with_grads += 1

        assert backbone_params_with_grads > 0, "Backbone parameters should receive gradients"
        assert (
            seed_child_params_with_grads > 0
        ), "Active seed child parameters should receive gradients"
