#!/usr/bin/env python3
"""
E2E Integration Test Runner

This script runs the comprehensive end-to-end integration test for the
Kaslite morphogenetic engine.

Usage:
    python run_e2e_test.py
    
    # Or with pytest
    pytest tests/test_e2e_integration.py -v -s
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tests.test_e2e_integration import E2EIntegrationTest


def main():
    """Run the E2E integration test suite."""
    print("🚀 Starting Kaslite End-to-End Integration Test")
    print("=" * 60)
    
    test_suite = E2EIntegrationTest()
    success = test_suite.run_all_tests()
    
    if success:
        print("\n🎉 E2E Integration Test PASSED!")
        print("The Kaslite morphogenetic engine is fully functional!")
        return 0
    else:
        print("\n❌ E2E Integration Test FAILED!")
        print("Please check the output above for details.")
        return 1


if __name__ == "__main__":
    exit(main())
