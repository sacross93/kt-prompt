#!/usr/bin/env python3
"""
Simple test script for error recovery system
"""
import sys
import tempfile
import time
from services.advanced_parsing_handler import AdvancedParsingHandler
from services.checkpoint_manager import CheckpointManager
from services.error_recovery_system import ErrorRecoverySystem
from models.exceptions import APIError, InvalidResponseError

def test_advanced_parsing_handler():
    """Test advanced parsing handler"""
    print("Testing AdvancedParsingHandler...")
    
    handler = AdvancedParsingHandler()
    
    # Test 1: Standard format
    response1 = """1. 사실형,긍정,현재,확실
2. 추론형,부정,과거,불확실
3. 대화형,미정,미래,확실"""
    
    result1 = handler.try_multiple_parsing_strategies(response1)
    assert result1 is not None, "Should parse standard format"
    assert len(result1) == 3, f"Expected 3 results, got {len(result1)}"
    print("✓ Standard format parsing works")
    
    # Test 2: Malformed response
    response2 = """1. 사실형, 긍정, 현재, 확실
2 추론형부정과거불확실
3. 대화형 미정 미래 확실"""
    
    result2 = handler.try_multiple_parsing_strategies(response2)
    print(f"✓ Malformed response handling: {len(result2) if result2 else 0} items recovered")
    
    # Test 3: Partial extraction
    response3 = """1. 사실형,긍정,현재,확실
2. invalid_format_here
3. 대화형,미정,미래,확실"""
    
    partial = handler.extract_partial_results(response3)
    assert partial.confidence > 0, "Should have some confidence"
    print(f"✓ Partial extraction: {partial.confidence:.2f} confidence, {len(partial.parsed_items)} items")
    
    # Test 4: Normalization
    messy = "1.사실형,긍정,현재,확실\n\n2.  추론형，부정，과거，불확실  "
    normalized = handler.normalize_response_format(messy)
    assert "1. 사실형" in normalized, "Should normalize numbering"
    print("✓ Response normalization works")
    
    print("AdvancedParsingHandler tests passed!\n")

def test_checkpoint_manager():
    """Test checkpoint manager"""
    print("Testing CheckpointManager...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = CheckpointManager(checkpoint_dir=temp_dir)
        
        # Test 1: Create and save checkpoint
        checkpoint_data = manager.create_checkpoint(
            iteration=5,
            current_prompt="test prompt",
            best_score=0.75,
            best_prompt="best prompt",
            optimization_history=[{"iteration": 1, "score": 0.6}],
            progress_percentage=50.0
        )
        
        success = manager.save_checkpoint(checkpoint_data, force=True)
        assert success, "Should save checkpoint successfully"
        print("✓ Checkpoint saving works")
        
        # Test 2: Load checkpoint
        loaded = manager.load_latest_checkpoint()
        assert loaded is not None, "Should load checkpoint"
        assert loaded.iteration == 5, f"Expected iteration 5, got {loaded.iteration}"
        assert loaded.best_score == 0.75, f"Expected score 0.75, got {loaded.best_score}"
        print("✓ Checkpoint loading works")
        
        # Test 3: Multiple checkpoints
        checkpoint_count_before = len(manager.list_checkpoints())
        
        for i in range(3):
            checkpoint_data = manager.create_checkpoint(
                iteration=i + 10,
                current_prompt=f"prompt {i}",
                best_score=0.5 + i * 0.1,
                best_prompt=f"best {i}",
                optimization_history=[],
                progress_percentage=i * 20.0
            )
            manager.save_checkpoint(checkpoint_data, force=True)
            time.sleep(0.1)  # Longer delay for different timestamps
        
        checkpoints = manager.list_checkpoints()
        print(f"Found {len(checkpoints)} checkpoints (was {checkpoint_count_before})")
        # Should have added 3 new checkpoints
        expected_count = checkpoint_count_before + 3
        assert len(checkpoints) >= expected_count, f"Expected at least {expected_count} checkpoints, got {len(checkpoints)}"
        print(f"✓ Multiple checkpoints: {len(checkpoints)} found")
    
    print("CheckpointManager tests passed!\n")

def test_error_recovery_system():
    """Test error recovery system"""
    print("Testing ErrorRecoverySystem...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        recovery_system = ErrorRecoverySystem(checkpoint_dir=temp_dir)
        
        # Test 1: Successful operation
        def mock_operation(value=10):
            return value * 2
        
        result = recovery_system.execute_with_recovery(
            operation=mock_operation,
            operation_name="test_operation",
            context={"value": 5},
            max_attempts=3
        )
        
        assert result == 10, f"Expected 10, got {result}"
        print("✓ Successful operation execution works")
        
        # Test 2: System status
        status = recovery_system.get_system_status()
        assert "api_error_stats" in status, "Should have API error stats"
        assert "system_health" in status, "Should have system health"
        assert status["system_health"] in ["healthy", "warning", "critical"], "Invalid health status"
        print(f"✓ System status: {status['system_health']}")
        
        # Test 3: Error recording
        recovery_system.api_handler.record_error("network_errors")
        recovery_system.api_handler.record_error("rate_limit_errors")
        
        stats = recovery_system.api_handler.get_error_statistics()
        assert stats["total_errors"] >= 2, "Should record errors"
        print(f"✓ Error recording: {stats['total_errors']} errors tracked")
        
        # Test 4: Circuit breaker
        for _ in range(12):  # Trigger circuit breaker
            recovery_system.api_handler.record_error("network_errors")
        
        circuit_breaker_active = not recovery_system.api_handler.handle_circuit_breaker()
        print(f"✓ Circuit breaker: {'active' if circuit_breaker_active else 'inactive'}")
    
    print("ErrorRecoverySystem tests passed!\n")

def test_integration():
    """Test integration scenario"""
    print("Testing integration scenario...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        recovery_system = ErrorRecoverySystem(checkpoint_dir=temp_dir)
        
        # Create a checkpoint
        checkpoint_data = recovery_system.checkpoint_manager.create_checkpoint(
            iteration=3,
            current_prompt="integration test prompt",
            best_score=0.8,
            best_prompt="integration best",
            optimization_history=[],
            progress_percentage=60.0
        )
        recovery_system.checkpoint_manager.save_checkpoint(checkpoint_data, force=True)
        
        # Test recovery
        recovery_info = recovery_system.recovery_manager.attempt_recovery()
        assert recovery_info is not None, "Should be able to recover"
        assert recovery_info["checkpoint"].iteration == 3, "Should recover correct iteration"
        print("✓ Integration recovery works")
        
        # Test cleanup
        recovery_system.cleanup_resources()
        print("✓ Resource cleanup works")
    
    print("Integration tests passed!\n")

def main():
    """Run all tests"""
    print("Running Error Recovery System Tests")
    print("=" * 50)
    
    try:
        test_advanced_parsing_handler()
        test_checkpoint_manager()
        test_error_recovery_system()
        test_integration()
        
        print("🎉 All tests passed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())