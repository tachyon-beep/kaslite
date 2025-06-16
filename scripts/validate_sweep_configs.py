#!/usr/bin/env python3
"""
Quick validation script for all sweep configurations.
"""

from pathlib import Path
from morphogenetic_engine.sweeps.config import load_sweep_config

def test_configurations():
    """Test all sweep configurations in the examples directory."""
    examples_dir = Path("examples")
    sweep_files = list(examples_dir.glob("*sweep*.yaml"))
    
    print("🔍 Testing Sweep Configurations")
    print("=" * 40)
    
    results = []
    for config_file in sorted(sweep_files):
        try:
            config = load_sweep_config(config_file)
            
            if config.sweep_type == 'grid':
                combinations = config.get_grid_combinations()
                count_str = f"{len(combinations)} combinations"
            elif config.sweep_type == 'bayesian':
                search_space = config.get_bayesian_search_space()
                count_str = f"{len(search_space)} parameters"
            else:
                count_str = "unknown type"
            
            print(f"✅ {config_file.name}: {config.sweep_type} sweep, {count_str}")
            results.append((config_file.name, True, count_str))
            
        except Exception as e:
            print(f"❌ {config_file.name}: Error - {e}")
            results.append((config_file.name, False, str(e)))
    
    print("=" * 40)
    
    # Summary
    successful = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print(f"📊 Summary: {successful}/{total} configurations valid")
    
    if successful == total:
        print("🎉 All sweep configurations are now working!")
    else:
        print("⚠️  Some configurations still need fixes")
    
    return results

if __name__ == "__main__":
    test_configurations()
