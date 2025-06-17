#!/usr/bin/env python3
"""Script to fix model registry tests."""

import re

# Read the test file
with open('tests/test_model_registry.py', 'r') as f:
    content = f.read()

# Apply systematic fixes
fixes = [
    # Fix tests that use MlflowClient patches
    (
        r'(def test_[^(]+\([^)]*mock_client_class[^)]*\):\s*"""[^"]*"""\s*# Setup mocks\s*mock_client = Mock\(\)\s*mock_client_class\.return_value = mock_client)',
        r'\1\n        self.registry.client = mock_client'
    ),
    (
        r'(def test_[^(]+\([^)]*mock_client_class[^)]*\):\s*"""[^"]*"""\s*# Setup mocks\s*mock_client = Mock\(\)\s*mock_client_class\.return_value = mock_client\s*\n)',
        r'\1        self.registry.client = mock_client\n'
    )
]

# Apply fixes
for pattern, replacement in fixes:
    content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)

# Manual fix for specific patterns I know exist
patterns_to_fix = [
    # Fix list_model_versions tests
    (
        r'(@patch\(\'morphogenetic_engine\.model_registry\.MlflowClient\'\)\s+def test_list_model_versions[^(]*\([^)]*mock_client_class[^)]*\):\s*"""[^"]*"""\s*# Setup mocks\s*mock_client = Mock\(\)\s*mock_client_class\.return_value = mock_client)',
        r'\1\n        self.registry.client = mock_client'
    ),
    # Fix promote_model tests
    (
        r'(@patch\(\'morphogenetic_engine\.model_registry\.MlflowClient\'\)\s+def test_promote_model[^(]*\([^)]*mock_client_class[^)]*\):\s*"""[^"]*"""\s*# Setup mocks\s*mock_client = Mock\(\)\s*mock_client_class\.return_value = mock_client)',
        r'\1\n        self.registry.client = mock_client'
    )
]

for pattern, replacement in patterns_to_fix:
    content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)

# Write back
with open('tests/test_model_registry.py', 'w') as f:
    f.write(content)

print('Applied systematic fixes to model registry tests')
