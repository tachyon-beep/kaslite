#!/usr/bin/env python3
"""
Simple test script to verify Prometheus monitoring integration.
"""

import time
import logging
from morphogenetic_engine.monitoring import initialize_monitoring, cleanup_monitoring

def test_monitoring():
    """Test basic monitoring functionality."""
    logging.basicConfig(level=logging.INFO)
    
    print("🔧 Testing Prometheus monitoring integration...")
    
    # Initialize monitoring
    monitor = initialize_monitoring(experiment_id="test_experiment", port=8001)
    
    print("✅ Monitoring initialized")
    print("📊 Metrics server started on http://localhost:8001/metrics")
    
    # Simulate some metrics
    print("📈 Simulating training metrics...")
    
    for epoch in range(1, 6):
        # Simulate epoch completion
        monitor.record_epoch_completion("phase_1", epoch_duration=1.5)
        
        # Simulate training metrics
        train_loss = 1.0 / epoch  # Decreasing loss
        val_loss = 0.8 / epoch
        val_acc = min(0.95, 0.5 + epoch * 0.1)  # Increasing accuracy
        best_acc = val_acc
        
        monitor.update_training_metrics("phase_1", train_loss, val_loss, val_acc, best_acc)
        
        # Simulate seed metrics
        monitor.update_seed_metrics(
            seed_id="test_seed_1",
            state="dormant" if epoch < 3 else "blending",
            alpha=max(0.0, (epoch - 3) * 0.3),
            drift=0.05 + epoch * 0.01,
            health_signal=0.1 + epoch * 0.02
        )
        
        print(f"   Epoch {epoch}: loss={train_loss:.3f}, acc={val_acc:.3f}")
        time.sleep(1)
    
    # Simulate germination
    print("🌱 Simulating seed germination...")
    monitor.record_germination()
    
    # Simulate phase transition
    print("🔄 Simulating phase transition...")
    monitor.record_phase_transition("phase_1", "phase_2")
    
    print("✅ Test completed successfully!")
    print("🌐 View metrics at: http://localhost:8001/metrics")
    print("⏱️  Keeping server running for 30 seconds for manual testing...")
    
    time.sleep(30)
    
    # Cleanup
    cleanup_monitoring()
    print("🧹 Monitoring cleaned up")

if __name__ == "__main__":
    test_monitoring()
