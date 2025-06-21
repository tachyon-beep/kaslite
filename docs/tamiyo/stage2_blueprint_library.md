# Stage 2: Blueprint Library Implementation Guide

## Overview

Stage 2 builds upon the foundation infrastructure to implement a comprehensive blueprint library supporting curriculum stages 0-2. This includes vision, language, and adversarial processing blueprints.

## Blueprint Categories

### 1. Enhanced Core Blueprints

#### 1.1 SE-Module (Squeeze-and-Excitation)

```python
# morphogenetic_engine/blueprints/attention.py
import torch
from torch import nn
from .base import Blueprint
from .registry import BlueprintRegistry

class SEModuleBlueprint(Blueprint):
    """Squeeze-and-Excitation module for channel recalibration."""
    
    def __init__(self, blueprint_id: str, input_dim: int, reduction_ratio: int = 16, **kwargs):
        super().__init__(blueprint_id, input_dim, **kwargs)
        self.reduction_ratio = reduction_ratio
        self.reduced_dim = max(input_dim // reduction_ratio, 1)
        self.use_case = "channel recalibration"
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Squeeze and excitation layers
        self.squeeze = nn.Linear(input_dim, self.reduced_dim)
        self.excitation = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.reduced_dim, input_dim),
            nn.Sigmoid()
        )
        
        self.parameter_count = self.get_parameter_count()
        self.initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim) or (batch_size, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
            squeeze_dim = True
        else:
            squeeze_dim = False
        
        batch_size, seq_len, channels = x.shape
        
        # Global average pooling over sequence dimension
        pooled = self.global_pool(x.transpose(1, 2)).transpose(1, 2)  # (batch, 1, channels)
        pooled = pooled.squeeze(1)  # (batch, channels)
        
        # Squeeze and excitation
        squeezed = self.squeeze(pooled)  # (batch, reduced_dim)
        excited = self.excitation(squeezed)  # (batch, channels)
        
        # Scale original input
        scale = excited.unsqueeze(1)  # (batch, 1, channels)
        output = x * scale
        
        if squeeze_dim:
            output = output.squeeze(1)
        
        return output
    
    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def get_memory_usage(self) -> dict:
        param_mb = self.parameter_count * 4 / (1024 * 1024)
        activation_mb = self.input_dim * 4 / (1024 * 1024)
        return {"parameters": param_mb, "activations": activation_mb}
    
    def get_latency_estimate(self, hardware_context) -> float:
        # FLOPs: pooling + 2 linear layers + activations
        flops = self.input_dim + 2 * self.input_dim * self.reduced_dim + self.input_dim
        return flops / hardware_context.flops_per_ms

class DepthwiseConvBlueprint(Blueprint):
    """Depthwise separable convolution for spatial processing."""
    
    def __init__(self, blueprint_id: str, input_dim: int, kernel_size: int = 3, **kwargs):
        super().__init__(blueprint_id, input_dim, **kwargs)
        self.kernel_size = kernel_size
        self.use_case = "local spatial processing"
        
        # Depthwise convolution (assuming input is channel dimension)
        self.depthwise = nn.Conv1d(
            input_dim, input_dim, 
            kernel_size=kernel_size, 
            padding=kernel_size//2, 
            groups=input_dim
        )
        
        # Pointwise convolution
        self.pointwise = nn.Conv1d(input_dim, input_dim, kernel_size=1)
        
        self.parameter_count = self.get_parameter_count()
        self.initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, input_dim) or (batch_size, seq_len, input_dim)
        if x.dim() == 2:
            # Add sequence dimension and transpose for conv1d
            x = x.unsqueeze(-1).transpose(1, 2)  # (batch, input_dim, 1)
            squeeze_output = True
        else:
            # Transpose for conv1d: (batch, input_dim, seq_len)
            x = x.transpose(1, 2)
            squeeze_output = False
        
        # Apply depthwise then pointwise convolution
        x = self.depthwise(x)
        x = self.pointwise(x)
        
        # Transpose back and optionally squeeze
        x = x.transpose(1, 2)
        if squeeze_output:
            x = x.squeeze(-1)
        
        return x
    
    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def get_memory_usage(self) -> dict:
        param_mb = self.parameter_count * 4 / (1024 * 1024)
        activation_mb = self.input_dim * 4 / (1024 * 1024)
        return {"parameters": param_mb, "activations": activation_mb}
    
    def get_latency_estimate(self, hardware_context) -> float:
        # FLOPs for depthwise + pointwise convolutions
        depthwise_flops = self.input_dim * self.kernel_size
        pointwise_flops = self.input_dim * self.input_dim
        total_flops = depthwise_flops + pointwise_flops
        return total_flops / hardware_context.flops_per_ms

# Register attention blueprints
BlueprintRegistry.register("se_module", SEModuleBlueprint, {"reduction_ratio": 16})
BlueprintRegistry.register("depthwise_conv", DepthwiseConvBlueprint, {"kernel_size": 3})
```

### 2. Language Processing Blueprints

#### 2.1 Mini Self-Attention

```python
# morphogenetic_engine/blueprints/language.py
import torch
from torch import nn
import torch.nn.functional as F
import math
from .base import Blueprint
from .registry import BlueprintRegistry

class MiniSelfAttentionBlueprint(Blueprint):
    """Lightweight self-attention for sequence processing."""
    
    def __init__(self, blueprint_id: str, input_dim: int, num_heads: int = 4, 
                 dropout: float = 0.1, **kwargs):
        super().__init__(blueprint_id, input_dim, **kwargs)
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_case = "context dependencies"
        
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.qkv_proj = nn.Linear(input_dim, 3 * input_dim, bias=False)
        self.output_proj = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.parameter_count = self.get_parameter_count()
        self.initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim) or (batch_size, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3 * embed_dim)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, embed_dim
        )
        output = self.output_proj(attn_output)
        
        if squeeze_output:
            output = output.squeeze(1)
        
        return output
    
    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def get_memory_usage(self) -> dict:
        param_mb = self.parameter_count * 4 / (1024 * 1024)
        # Attention requires O(seq_len^2) memory
        activation_mb = (self.input_dim + self.num_heads * 64 * 64) * 4 / (1024 * 1024)
        return {"parameters": param_mb, "activations": activation_mb}
    
    def get_latency_estimate(self, hardware_context) -> float:
        seq_len = 64  # Assume typical sequence length
        # QKV projection + attention computation + output projection
        qkv_flops = 3 * self.input_dim * self.input_dim
        attn_flops = 2 * self.num_heads * seq_len * seq_len * self.head_dim
        out_flops = self.input_dim * self.input_dim
        total_flops = qkv_flops + attn_flops + out_flops
        return total_flops / hardware_context.flops_per_ms

class AdapterBlueprint(Blueprint):
    """Parameter-efficient adapter module for language models."""
    
    def __init__(self, blueprint_id: str, input_dim: int, adapter_dim: int = None, 
                 dropout: float = 0.1, **kwargs):
        super().__init__(blueprint_id, input_dim, **kwargs)
        self.adapter_dim = adapter_dim or max(input_dim // 8, 8)
        self.use_case = "parameter-efficient fine-tuning"
        
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, self.adapter_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.adapter_dim, input_dim)
        )
        
        # Initialize to near-zero output
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        
        self.parameter_count = self.get_parameter_count()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.adapter(x)  # Residual connection
    
    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def get_memory_usage(self) -> dict:
        param_mb = self.parameter_count * 4 / (1024 * 1024)
        activation_mb = (self.input_dim + self.adapter_dim) * 4 / (1024 * 1024)
        return {"parameters": param_mb, "activations": activation_mb}
    
    def get_latency_estimate(self, hardware_context) -> float:
        flops = 2 * self.input_dim * self.adapter_dim + 2 * self.adapter_dim * self.input_dim
        return flops / hardware_context.flops_per_ms

# Register language blueprints
BlueprintRegistry.register("mini_self_attention", MiniSelfAttentionBlueprint, 
                          {"num_heads": 4, "dropout": 0.1})
BlueprintRegistry.register("adapter", AdapterBlueprint, 
                          {"adapter_dim": None, "dropout": 0.1})
```

### 3. Vision and Adversarial Blueprints

#### 3.1 Denoising AutoEncoder

```python
# morphogenetic_engine/blueprints/vision.py
import torch
from torch import nn
from .base import Blueprint
from .registry import BlueprintRegistry

class DenoisingAutoEncoderBlueprint(Blueprint):
    """Denoising autoencoder for adversarial filtration."""
    
    def __init__(self, blueprint_id: str, input_dim: int, hidden_dim: int = None,
                 noise_std: float = 0.1, **kwargs):
        super().__init__(blueprint_id, input_dim, **kwargs)
        self.hidden_dim = hidden_dim or max(input_dim // 2, 16)
        self.noise_std = noise_std
        self.use_case = "adversarial filtration"
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim // 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, input_dim),
            nn.Tanh()  # Bounded output
        )
        
        self.parameter_count = self.get_parameter_count()
        self.initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.noise_std > 0:
            # Add noise during training
            noise = torch.randn_like(x) * self.noise_std
            x_noisy = x + noise
        else:
            x_noisy = x
        
        # Encode and decode
        encoded = self.encoder(x_noisy)
        decoded = self.decoder(encoded)
        
        # Residual connection for stability
        return x + 0.1 * (decoded - x)
    
    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def get_memory_usage(self) -> dict:
        param_mb = self.parameter_count * 4 / (1024 * 1024)
        activation_mb = (self.input_dim + self.hidden_dim) * 4 / (1024 * 1024)
        return {"parameters": param_mb, "activations": activation_mb}
    
    def get_latency_estimate(self, hardware_context) -> float:
        # Encoder + decoder FLOPs
        encoder_flops = (
            2 * self.input_dim * self.hidden_dim + 
            2 * self.hidden_dim * (self.hidden_dim // 2)
        )
        decoder_flops = (
            2 * (self.hidden_dim // 2) * self.hidden_dim + 
            2 * self.hidden_dim * self.input_dim
        )
        total_flops = encoder_flops + decoder_flops
        return total_flops / hardware_context.flops_per_ms

class AttentionFilterBlueprint(Blueprint):
    """Attention-based filter for adversarial detection."""
    
    def __init__(self, blueprint_id: str, input_dim: int, filter_heads: int = 2, **kwargs):
        super().__init__(blueprint_id, input_dim, **kwargs)
        self.filter_heads = filter_heads
        self.head_dim = input_dim // filter_heads
        self.use_case = "adversarial detection"
        
        assert input_dim % filter_heads == 0
        
        # Attention mechanism for filtering
        self.query_proj = nn.Linear(input_dim, input_dim)
        self.key_proj = nn.Linear(input_dim, input_dim)
        self.value_proj = nn.Linear(input_dim, input_dim)
        self.output_proj = nn.Linear(input_dim, input_dim)
        
        # Adversarial detection head
        self.detector = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.parameter_count = self.get_parameter_count()
        self.initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Multi-head attention for feature filtering
        q = self.query_proj(x).view(batch_size, self.filter_heads, self.head_dim)
        k = self.key_proj(x).view(batch_size, self.filter_heads, self.head_dim)
        v = self.value_proj(x).view(batch_size, self.filter_heads, self.head_dim)
        
        # Compute attention weights
        scores = torch.sum(q * k, dim=-1) / math.sqrt(self.head_dim)
        weights = torch.softmax(scores, dim=-1)
        
        # Apply attention
        filtered = torch.sum(weights.unsqueeze(-1) * v, dim=1)
        filtered = filtered.view(batch_size, -1)
        
        # Output projection
        output = self.output_proj(filtered)
        
        # Adversarial detection (for monitoring)
        adv_score = self.detector(x)
        
        # Store detection score in metadata (if needed)
        if hasattr(x, 'metadata'):
            x.metadata['adversarial_score'] = adv_score
        
        return output
    
    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def get_memory_usage(self) -> dict:
        param_mb = self.parameter_count * 4 / (1024 * 1024)
        activation_mb = self.input_dim * 4 / (1024 * 1024)
        return {"parameters": param_mb, "activations": activation_mb}
    
    def get_latency_estimate(self, hardware_context) -> float:
        # QKV projections + attention + output projection + detector
        proj_flops = 4 * self.input_dim * self.input_dim
        attn_flops = 2 * self.filter_heads * self.head_dim
        detector_flops = 2 * self.input_dim * (self.input_dim // 2) + self.input_dim // 2
        total_flops = proj_flops + attn_flops + detector_flops
        return total_flops / hardware_context.flops_per_ms

# Register vision blueprints
BlueprintRegistry.register("denoising_autoencoder", DenoisingAutoEncoderBlueprint,
                          {"hidden_dim": None, "noise_std": 0.1})
BlueprintRegistry.register("attention_filter", AttentionFilterBlueprint,
                          {"filter_heads": 2})
```

### 4. Advanced Utility Blueprints

#### 4.1 GLU (Gated Linear Unit)

```python
# morphogenetic_engine/blueprints/utility.py
import torch
from torch import nn
from .base import Blueprint
from .registry import BlueprintRegistry

class GLUBlueprint(Blueprint):
    """Gated Linear Unit for controlled information flow."""
    
    def __init__(self, blueprint_id: str, input_dim: int, **kwargs):
        super().__init__(blueprint_id, input_dim, **kwargs)
        self.use_case = "gated flow"
        
        # GLU: Linear(d->2d) -> [x,a] -> x*σ(a)
        self.projection = nn.Linear(input_dim, 2 * input_dim)
        
        self.parameter_count = self.get_parameter_count()
        self.initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projection(x)
        
        # Split into value and gate
        value, gate = projected.chunk(2, dim=-1)
        
        # Apply sigmoid gating
        gated_value = value * torch.sigmoid(gate)
        
        return gated_value
    
    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def get_memory_usage(self) -> dict:
        param_mb = self.parameter_count * 4 / (1024 * 1024)
        activation_mb = 2 * self.input_dim * 4 / (1024 * 1024)  # 2x for projection
        return {"parameters": param_mb, "activations": activation_mb}
    
    def get_latency_estimate(self, hardware_context) -> float:
        flops = 2 * self.input_dim * (2 * self.input_dim) + self.input_dim  # projection + sigmoid
        return flops / hardware_context.flops_per_ms

class ResidualMLPBlueprint(Blueprint):
    """Standard residual MLP with configurable depth."""
    
    def __init__(self, blueprint_id: str, input_dim: int, hidden_dim: int = None,
                 num_layers: int = 2, dropout: float = 0.1, **kwargs):
        super().__init__(blueprint_id, input_dim, **kwargs)
        self.hidden_dim = hidden_dim or input_dim * 2
        self.num_layers = num_layers
        self.use_case = "general capacity"
        
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.extend([
                    nn.Linear(input_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
            elif i == num_layers - 1:
                layers.append(nn.Linear(self.hidden_dim, input_dim))
            else:
                layers.extend([
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
        
        self.mlp = nn.Sequential(*layers)
        
        self.parameter_count = self.get_parameter_count()
        self.initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)  # Residual connection
    
    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def get_memory_usage(self) -> dict:
        param_mb = self.parameter_count * 4 / (1024 * 1024)
        activation_mb = (self.input_dim + self.hidden_dim) * 4 / (1024 * 1024)
        return {"parameters": param_mb, "activations": activation_mb}
    
    def get_latency_estimate(self, hardware_context) -> float:
        # Approximate FLOPs for multi-layer MLP
        flops = (
            2 * self.input_dim * self.hidden_dim +  # First layer
            2 * (self.num_layers - 2) * self.hidden_dim * self.hidden_dim +  # Hidden layers
            2 * self.hidden_dim * self.input_dim  # Final layer
        )
        return flops / hardware_context.flops_per_ms

# Register utility blueprints
BlueprintRegistry.register("glu", GLUBlueprint)
BlueprintRegistry.register("residual_mlp", ResidualMLPBlueprint,
                          {"hidden_dim": None, "num_layers": 2, "dropout": 0.1})
```

## Blueprint Testing Framework

### 1. Performance Testing Suite

```python
# tests/test_blueprint_performance.py
import pytest
import torch
import time
from morphogenetic_engine.blueprints.registry import BlueprintRegistry
from morphogenetic_engine.telemetry.types import HardwareContext

class TestBlueprintPerformance:
    """Performance testing for all blueprints."""
    
    @pytest.fixture
    def hardware_contexts(self):
        return {
            "cpu": HardwareContext(
                device_type="cpu", memory_gb=8.0, flops_per_ms=1e6,
                bandwidth_gbps=25.6, power_budget_watts=65.0, latency_target_ms=100.0
            ),
            "gpu": HardwareContext(
                device_type="gpu", memory_gb=24.0, flops_per_ms=1e9,
                bandwidth_gbps=1008.0, power_budget_watts=450.0, latency_target_ms=10.0
            )
        }
    
    @pytest.mark.parametrize("blueprint_name", [
        "no_op", "bottleneck_adapter", "low_rank_residual", 
        "se_module", "depthwise_conv", "mini_self_attention",
        "adapter", "denoising_autoencoder", "attention_filter",
        "glu", "residual_mlp"
    ])
    def test_blueprint_latency(self, blueprint_name, hardware_contexts):
        """Test that blueprint latency estimates are reasonable."""
        blueprint = BlueprintRegistry.create_blueprint(blueprint_name, input_dim=64)
        
        for hw_name, hw_context in hardware_contexts.items():
            latency = blueprint.get_latency_estimate(hw_context)
            
            # Latency should be positive and reasonable
            assert latency > 0
            assert latency < hw_context.latency_target_ms
    
    @pytest.mark.parametrize("input_dim", [32, 64, 128, 256])
    def test_blueprint_scaling(self, input_dim):
        """Test blueprint behavior with different input dimensions."""
        blueprint = BlueprintRegistry.create_blueprint("bottleneck_adapter", input_dim=input_dim)
        
        x = torch.randn(10, input_dim)
        output = blueprint(x)
        
        assert output.shape == x.shape
        assert blueprint.get_parameter_count() > 0
    
    def test_memory_usage_estimation(self):
        """Test memory usage estimation accuracy."""
        blueprint = BlueprintRegistry.create_blueprint("mini_self_attention", input_dim=128)
        
        memory_stats = blueprint.get_memory_usage()
        
        assert "parameters" in memory_stats
        assert "activations" in memory_stats
        assert memory_stats["parameters"] > 0
        assert memory_stats["activations"] > 0
    
    def test_blueprint_forward_speed(self):
        """Benchmark actual forward pass speed."""
        blueprint = BlueprintRegistry.create_blueprint("se_module", input_dim=256)
        blueprint.eval()
        
        x = torch.randn(32, 256)
        
        # Warmup
        for _ in range(10):
            _ = blueprint(x)
        
        # Benchmark
        start_time = time.time()
        for _ in range(100):
            _ = blueprint(x)
        end_time = time.time()
        
        avg_time_ms = (end_time - start_time) * 1000 / 100
        assert avg_time_ms < 50  # Should be fast enough
```

### 2. Integration Testing

```python
# tests/test_blueprint_integration.py
import pytest
import torch
from morphogenetic_engine.blueprints.registry import BlueprintRegistry
from morphogenetic_engine.core import EnhancedSeedManager
from morphogenetic_engine.telemetry.collectors import TelemetryCollector

class TestBlueprintIntegration:
    """Test blueprint integration with core systems."""
    
    def test_blueprint_with_seed_manager(self):
        """Test blueprint integration with SeedManager."""
        manager = EnhancedSeedManager(hardware_profile="cpu_basic")
        
        # Register a blueprint-based seed
        manager.register_blueprint_seed(
            "se_module", "test_seed", input_dim=64, reduction_ratio=8
        )
        
        assert "test_seed" in manager.seeds
        assert "test_seed" in manager.blueprint_instances
        
        # Test telemetry collection
        blueprint = manager.blueprint_instances["test_seed"]
        x = torch.randn(10, 64)
        output = blueprint(x)
        
        # Record telemetry
        manager.telemetry_collector.record_activation("test_seed", output)
        telemetry = manager.get_comprehensive_telemetry("test_seed")
        
        assert telemetry.seed_id == "test_seed"
        assert telemetry.activation_variance >= 0
    
    def test_multiple_blueprints(self):
        """Test system with multiple different blueprints."""
        manager = EnhancedSeedManager(hardware_profile="gpu_rtx4090")
        
        # Register different blueprint types
        blueprint_configs = [
            ("no_op", "seed1", {}),
            ("bottleneck_adapter", "seed2", {"bottleneck_dim": 16}),
            ("mini_self_attention", "seed3", {"num_heads": 2}),
            ("glu", "seed4", {})
        ]
        
        for blueprint_name, seed_id, config in blueprint_configs:
            manager.register_blueprint_seed(
                blueprint_name, seed_id, input_dim=128, **config
            )
        
        # Test that all seeds work together
        x = torch.randn(5, 128)
        
        for seed_id in ["seed1", "seed2", "seed3", "seed4"]:
            blueprint = manager.blueprint_instances[seed_id]
            output = blueprint(x)
            assert output.shape == x.shape
    
    def test_blueprint_error_handling(self):
        """Test error handling for invalid blueprint configurations."""
        with pytest.raises(ValueError, match="Unknown blueprint"):
            BlueprintRegistry.create_blueprint("nonexistent_blueprint", input_dim=64)
        
        # Test invalid configuration
        with pytest.raises(AssertionError):
            BlueprintRegistry.create_blueprint(
                "mini_self_attention", input_dim=63, num_heads=4  # Not divisible
            )
```

## Blueprint Documentation and Examples

### Usage Examples

```python
# examples/blueprint_showcase.py
"""
Showcase different blueprint capabilities and use cases.
"""
import torch
from morphogenetic_engine.blueprints.registry import BlueprintRegistry
from morphogenetic_engine.telemetry.types import HardwareContext

def showcase_blueprints():
    """Demonstrate various blueprint capabilities."""
    
    # Create hardware context
    gpu_context = HardwareContext(
        device_type="gpu", memory_gb=24.0, flops_per_ms=1e9,
        bandwidth_gbps=1008.0, power_budget_watts=450.0, latency_target_ms=10.0
    )
    
    # Sample input
    batch_size, seq_len, input_dim = 16, 32, 128
    x = torch.randn(batch_size, seq_len, input_dim)
    
    print("Blueprint Showcase")
    print("=" * 50)
    
    for blueprint_name in BlueprintRegistry.list_blueprints():
        print(f"\nTesting {blueprint_name}:")
        
        # Create blueprint
        blueprint = BlueprintRegistry.create_blueprint(blueprint_name, input_dim=input_dim)
        
        # Test forward pass
        if blueprint_name in ["mini_self_attention", "se_module"]:
            # These can handle sequence inputs
            output = blueprint(x)
        else:
            # Use single timestep
            output = blueprint(x[:, 0, :])
        
        # Print statistics
        params = blueprint.get_parameter_count()
        memory = blueprint.get_memory_usage()
        latency = blueprint.get_latency_estimate(gpu_context)
        
        print(f"  Parameters: {params:,}")
        print(f"  Memory: {memory['parameters']:.2f} MB")
        print(f"  Estimated latency: {latency:.2f} ms")
        print(f"  Use case: {blueprint.use_case}")

if __name__ == "__main__":
    showcase_blueprints()
```

## Blueprint Configuration System

### 1. Configuration Management

```python
# morphogenetic_engine/blueprints/config.py
from typing import Dict, Any, Optional
import yaml
import json
from pathlib import Path

class BlueprintConfigManager:
    """Manage blueprint configurations and presets."""
    
    def __init__(self, config_dir: str = "configs/blueprints"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.presets = {}
        self.load_presets()
    
    def load_presets(self):
        """Load blueprint presets from configuration files."""
        for config_file in self.config_dir.glob("*.yaml"):
            with open(config_file, 'r') as f:
                preset_data = yaml.safe_load(f)
                self.presets.update(preset_data)
    
    def save_preset(self, name: str, blueprint_type: str, config: Dict[str, Any]):
        """Save a blueprint configuration preset."""
        preset_file = self.config_dir / f"{name}.yaml"
        preset_data = {
            name: {
                "blueprint_type": blueprint_type,
                "config": config,
                "description": f"Preset configuration for {blueprint_type}"
            }
        }
        
        with open(preset_file, 'w') as f:
            yaml.dump(preset_data, f, default_flow_style=False)
        
        self.presets[name] = preset_data[name]
    
    def get_preset(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a blueprint preset by name."""
        return self.presets.get(name)
    
    def list_presets(self) -> Dict[str, str]:
        """List available presets with descriptions."""
        return {
            name: data.get("description", "No description")
            for name, data in self.presets.items()
        }
    
    def create_from_preset(self, preset_name: str, input_dim: int, **override_kwargs):
        """Create a blueprint from a preset configuration."""
        preset = self.get_preset(preset_name)
        if not preset:
            raise ValueError(f"Preset '{preset_name}' not found")
        
        config = preset["config"].copy()
        config.update(override_kwargs)
        
        return BlueprintRegistry.create_blueprint(
            preset["blueprint_type"], input_dim, **config
        )

# Example preset configurations
STAGE2_PRESETS = {
    "vision_lightweight": {
        "blueprint_type": "se_module",
        "config": {"reduction_ratio": 8},
        "description": "Lightweight SE module for vision tasks"
    },
    "language_efficient": {
        "blueprint_type": "adapter",
        "config": {"adapter_dim": 32, "dropout": 0.05},
        "description": "Efficient adapter for language models"
    },
    "adversarial_defense": {
        "blueprint_type": "denoising_autoencoder",
        "config": {"hidden_dim": 64, "noise_std": 0.15},
        "description": "Robust denoising for adversarial inputs"
    },
    "attention_mini": {
        "blueprint_type": "mini_self_attention",
        "config": {"num_heads": 2, "dropout": 0.05},
        "description": "Minimal self-attention for sequences"
    }
}
```

## Performance Benchmarks

### Benchmark Suite

```python
# benchmarks/blueprint_benchmarks.py
import torch
import time
import memory_profiler
from morphogenetic_engine.blueprints.registry import BlueprintRegistry

class BlueprintBenchmark:
    """Comprehensive benchmark suite for blueprints."""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_blueprint(self, blueprint_name: str, input_dims: list, 
                          batch_sizes: list, num_iterations: int = 100):
        """Benchmark a blueprint across different configurations."""
        results = {}
        
        for input_dim in input_dims:
            for batch_size in batch_sizes:
                print(f"Benchmarking {blueprint_name} - dim:{input_dim}, batch:{batch_size}")
                
                # Create blueprint
                blueprint = BlueprintRegistry.create_blueprint(blueprint_name, input_dim=input_dim)
                blueprint.eval()
                
                # Prepare input
                x = torch.randn(batch_size, input_dim)
                
                # Warmup
                for _ in range(10):
                    _ = blueprint(x)
                
                # Timing benchmark
                start_time = time.time()
                for _ in range(num_iterations):
                    _ = blueprint(x)
                end_time = time.time()
                
                avg_time_ms = (end_time - start_time) * 1000 / num_iterations
                
                # Memory benchmark
                @memory_profiler.profile
                def memory_test():
                    return blueprint(x)
                
                # Store results
                key = f"dim_{input_dim}_batch_{batch_size}"
                results[key] = {
                    "avg_time_ms": avg_time_ms,
                    "parameters": blueprint.get_parameter_count(),
                    "memory_usage": blueprint.get_memory_usage()
                }
        
        self.results[blueprint_name] = results
        return results
    
    def run_all_benchmarks(self):
        """Run benchmarks for all registered blueprints."""
        input_dims = [32, 64, 128, 256]
        batch_sizes = [1, 8, 32, 64]
        
        for blueprint_name in BlueprintRegistry.list_blueprints():
            try:
                self.benchmark_blueprint(blueprint_name, input_dims, batch_sizes)
            except Exception as e:
                print(f"Error benchmarking {blueprint_name}: {e}")
    
    def generate_report(self):
        """Generate a performance report."""
        print("\nBlueprint Performance Report")
        print("=" * 60)
        
        for blueprint_name, results in self.results.items():
            print(f"\n{blueprint_name}:")
            for config, metrics in results.items():
                print(f"  {config}: {metrics['avg_time_ms']:.2f}ms, "
                      f"{metrics['parameters']:,} params")

if __name__ == "__main__":
    benchmark = BlueprintBenchmark()
    benchmark.run_all_benchmarks()
    benchmark.generate_report()
```

## Deliverables Checklist

- [ ] SE-Module and Depthwise Conv blueprints
- [ ] Mini Self-Attention and Adapter blueprints  
- [ ] Denoising AutoEncoder and Attention Filter
- [ ] GLU and Residual MLP blueprints
- [ ] Comprehensive performance testing suite
- [ ] Integration tests with SeedManager
- [ ] Blueprint configuration management system
- [ ] Performance benchmarking framework
- [ ] Documentation and usage examples
- [ ] Memory and latency profiling
- [ ] Error handling and validation
- [ ] Preset configuration system

## Stage 2 Success Criteria

1. **Functional Requirements**
   - All 11 blueprint types implemented and tested
   - Forward pass works correctly for all input shapes
   - Parameter counting and memory estimation accurate
   - Integration with existing SeedManager seamless

2. **Performance Requirements**
   - No blueprint exceeds 50ms latency on target hardware
   - Memory overhead <10% of base model size
   - Parameter count matches theoretical estimates
   - All tests pass with 85%+ coverage

3. **Quality Requirements**
   - Comprehensive documentation for each blueprint
   - Usage examples and configuration presets
   - Performance benchmarks and profiling data
   - Error handling for edge cases

This completes Stage 2, providing a rich blueprint library ready for Stage 3's policy network integration.
