#!/usr/bin/env python3
"""
Test enhanced error handling functionality
"""
import sys
sys.path.append('.')

def test_enhanced_parsing():
    """Test enhanced parsing capabilities"""
    print("=== Testing Enhanced Parsing ===")
    
    from services.advanced_parsing_handler import AdvancedParsingHandler
    
    handler = AdvancedParsingHandler()
    
    # Test cases with various malformed responses
    test_cases = [
        "1. 사실형,긍정,현재,확실\n2. 추론형,부정,과거,불확실",  # Good format
        "1. 사실형 긍정 현재 확실\n2 추론형부정과거불확실",  # Missing separators
        "1. fact,positive,present,certain\n2. 추론형,부정,과거,불확실",  # Mixed languages
        "답: 사실형,긍정,현재,확실",  # With prefix
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest {i+1}: {test_case[:50]}...")
        
        # Try standard parsing
        result = handler.try_multiple_parsing_strategies(test_case)
        print(f"  Standard result: {result}")
        
        # Try enhanced recovery
        enhanced = handler.enhance_response_format_recovery(test_case, "parsing_error")
        enhanced_result = handler.try_multiple_parsing_strategies(enhanced)
        print(f"  Enhanced result: {enhanced_result}")
        
        # Show partial extraction
        partial = handler.extract_partial_results(test_case)
        print(f"  Partial: {len(partial.parsed_items)} items, confidence: {partial.confidence:.2f}")

def test_api_error_handling():
    """Test API error handling"""
    print("\n=== Testing API Error Handling ===")
    
    from services.api_error_handler import APIErrorHandler
    from models.exceptions import InvalidResponseError
    
    handler = APIErrorHandler()
    
    # Test response format error handling
    malformed_response = "1. 사실형 긍정 현재 확실\n2 추론형부정과거불확실"
    
    try:
        result = handler.handle_response_format_error(malformed_response, 0, "test_error")
        print(f"Format error handling result: {result}")
    except Exception as e:
        print(f"Format error handling failed: {e}")
    
    # Test adaptive retry strategy
    error_history = ["parsing_error", "rate_limit_error"]
    strategy = handler.adaptive_retry_strategy("parsing", 1, error_history)
    print(f"Adaptive strategy: {strategy}")

def test_checkpoint_enhancements():
    """Test checkpoint enhancements"""
    print("\n=== Testing Checkpoint Enhancements ===")
    
    from services.checkpoint_manager import CheckpointManager
    
    manager = CheckpointManager(checkpoint_dir="test_enhanced_checkpoints")
    
    # Create base checkpoint
    base_checkpoint = manager.create_checkpoint(
        iteration=1,
        current_prompt="base prompt",
        best_score=0.6,
        best_prompt="base best",
        optimization_history=[],
        progress_percentage=20.0
    )
    
    manager.save_checkpoint(base_checkpoint, force=True)
    print("Base checkpoint saved")
    
    # Create incremental checkpoint
    incremental = manager.create_incremental_checkpoint(
        base_checkpoint,
        iteration=2,
        best_score=0.7,
        progress_percentage=40.0
    )
    
    manager.save_checkpoint(incremental, force=True)
    print("Incremental checkpoint saved")
    
    # Test progress checkpoint
    progress_success = manager.save_progress_checkpoint("test_operation", {
        "step": "parsing",
        "items_processed": 50,
        "success_rate": 0.8
    })
    print(f"Progress checkpoint saved: {progress_success}")
    
    # Get statistics
    stats = manager.get_checkpoint_statistics()
    print(f"Checkpoint statistics: {stats}")

def test_error_recovery_integration():
    """Test integrated error recovery"""
    print("\n=== Testing Error Recovery Integration ===")
    
    from services.error_recovery_system import ErrorRecoverySystem
    from models.exceptions import InvalidResponseError
    
    recovery_system = ErrorRecoverySystem(checkpoint_dir="test_enhanced_checkpoints")
    
    # Test operation that succeeds
    def successful_operation(value=5):
        return value * 2
    
    try:
        result = recovery_system.execute_with_recovery(
            operation=successful_operation,
            operation_name="test_success",
            context={"value": 3},
            max_attempts=3
        )
        print(f"Successful operation result: {result}")
    except Exception as e:
        print(f"Operation failed: {e}")
    
    # Get comprehensive report
    report = recovery_system.get_comprehensive_system_report()
    print(f"System health: {report['basic_status']['system_health']}")
    print(f"Performance metrics: {report['performance_metrics']}")

def main():
    """Run all enhanced tests"""
    print("Enhanced Error Handling System Test")
    print("=" * 50)
    
    try:
        test_enhanced_parsing()
        test_api_error_handling()
        test_checkpoint_enhancements()
        test_error_recovery_integration()
        
        print("\n" + "=" * 50)
        print("All enhanced error handling tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)