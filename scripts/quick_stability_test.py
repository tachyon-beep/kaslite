#!/usr/bin/env python3
"""
Quick test to verify layout stability fixes are working.
"""

import time
from rich.console import Console
from morphogenetic_engine.cli_dashboard import RichDashboard


def quick_test():
    console = Console()
    console.print("🧪 Quick Layout Stability Test")
    
    # Test the context manager approach (like in real experiment)
    with RichDashboard() as dashboard:
        original_layout_id = id(dashboard.layout)
        console.print(f"Initial layout ID: {original_layout_id}")
        
        # Initialize some data
        dashboard.update_seed("seed1", "dormant")
        dashboard.update_seed("seed2", "dormant")
        
        # Start a phase (this was causing the issue)
        dashboard.start_phase("test_phase", 3, "Testing stability")
        after_phase_layout_id = id(dashboard.layout)
        
        if original_layout_id == after_phase_layout_id:
            console.print("✅ Layout remained stable after start_phase")
        else:
            console.print("❌ Layout was recreated after start_phase")
            return False
        
        # Update progress a few times
        for i in range(3):
            dashboard.update_progress(i+1, {
                "train_loss": 1.0 - i*0.1,
                "val_loss": 0.9 - i*0.1,
                "val_acc": 0.5 + i*0.1,
                "best_acc": 0.5 + i*0.1
            })
            time.sleep(0.5)
        
        final_layout_id = id(dashboard.layout)
        if original_layout_id == final_layout_id:
            console.print("✅ Layout remained stable throughout updates")
            return True
        else:
            console.print("❌ Layout was recreated during updates")
            return False


if __name__ == "__main__":
    success = quick_test()
    if success:
        print("🎉 Layout stability fixes are working!")
    else:
        print("❌ Layout stability issues remain")
        exit(1)
