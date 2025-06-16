#!/usr/bin/env python3
"""
Simple test of Phase 3 sweep functionality.
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_basic_functionality():
    """Test basic sweep functionality without running actual experiments."""
    print("🚀 Testing Phase 3 Sweep Functionality")
    
    # Test 1: Import modules
    try:
        from morphogenetic_engine.sweeps.config import load_sweep_config, SweepConfig
        from morphogenetic_engine.cli.sweep import SweepCLI
        print("✅ All modules imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test 2: Load configuration
    try:
        config_path = Path("examples/quick_sweep.yaml")
        if not config_path.exists():
            print(f"❌ Config file not found: {config_path}")
            return False
        
        config = load_sweep_config(config_path)
        print(f"✅ Configuration loaded: {config.sweep_type}")
        print(f"   Parameters: {list(config.parameters.keys())}")
    except Exception as e:
        print(f"❌ Config loading error: {e}")
        return False
    
    # Test 3: Generate combinations
    try:
        combinations = config.get_grid_combinations()
        print(f"✅ Generated {len(combinations)} parameter combinations")
        if combinations:
            print(f"   Example combination: {combinations[0]}")
    except Exception as e:
        print(f"❌ Combination generation error: {e}")
        return False
    
    # Test 4: CLI functionality
    try:
        cli = SweepCLI()
        parser = cli.create_parser()
        
        # Test argument parsing
        args = parser.parse_args(['grid', '--config', str(config_path), '--parallel', '2'])
        print(f"✅ CLI argument parsing successful")
        print(f"   Command: {args.command}")
        print(f"   Config: {args.config}")
        print(f"   Parallel: {args.parallel}")
    except Exception as e:
        print(f"❌ CLI error: {e}")
        return False
    
    # Test 5: Bayesian config (if available)
    try:
        bayesian_path = Path("examples/bayesian_sweep.yaml")
        if bayesian_path.exists():
            bayesian_config = load_sweep_config(bayesian_path)
            search_space = bayesian_config.get_bayesian_search_space()
            print(f"✅ Bayesian configuration loaded")
            print(f"   Search space parameters: {list(search_space.keys())}")
        else:
            print("⚠️  Bayesian config not found, but that's OK")
    except Exception as e:
        print(f"⚠️  Bayesian config error (optional): {e}")
    
    # Test 6: Check for Optuna (now required)
    import optuna
    print("✅ Optuna available for Bayesian optimization")
    
    print("\n🎉 Phase 3 basic functionality test completed successfully!")
    print("\nNext steps:")
    print("• Run: python -m morphogenetic_engine.cli.sweep --help")
    print("• Run: python -m morphogenetic_engine.cli.reports --help") 
    print("• Try: python -m morphogenetic_engine.cli.sweep quick --problem spirals --trials 2")
    
    return True

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
