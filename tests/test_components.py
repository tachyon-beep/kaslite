"""
Comprehensive test suite for the morphogenetic_engine.components module.

This test file has been refactored and split into focused modules:

- test_sentinel_seed.py: Tests for SentinelSeed component
- test_base_net.py: Tests for BaseNet and multi-seed functionality  
- test_components_integration.py: Tests for CLI flags, compatibility, and edge cases
- test_utils.py: Common test utilities and helpers

The original comprehensive tests have been reorganized for better maintainability
while preserving all test coverage. Import the specific test modules directly
or run the entire test suite via pytest.

Migration Guide:
- SentinelSeed tests: see test_sentinel_seed.py
- BaseNet tests: see test_base_net.py  
- CLI flags tests: see test_components_integration.py
- Edge cases: see test_components_integration.py

This refactoring reduces file size from 1124 lines to focused, manageable modules
while maintaining complete test coverage of the morphogenetic architecture system.
"""

# Re-export key test classes for backward compatibility
from .test_sentinel_seed import TestSentinelSeed
from .test_base_net import TestBaseNet, TestMultiSeedBaseNet
from .test_components_integration import (
    TestCLIFlags,
    TestMultiSeedArchitecture, 
    TestBackwardCompatibility,
    TestEdgeCases
)

__all__ = [
    "TestSentinelSeed",
    "TestBaseNet", 
    "TestMultiSeedBaseNet",
    "TestCLIFlags",
    "TestMultiSeedArchitecture",
    "TestBackwardCompatibility", 
    "TestEdgeCases"
]
