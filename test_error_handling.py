#!/usr/bin/env python3
"""
Test script for advanced error handling system
"""
import sys
sys.path.append('.')

def test_advanced_parsing_handler():
    """Test AdvancedParsingHandler functionality"""
    print("Testing AdvancedParsingHandler...")
    
    from services.advanced_parsing_handler import AdvancedParsingHandler
    
    handler = AdvancedParsingHandler()
    
    # Test standard format
    response1 = "1. 사실형,긍정,현재,확실\n2. 추론형,부정,과거,불확실"
    result1 = handler.try_multiple_parsing_strategies(response1)
    print(f"Standard format result: {result1}")
    
    # Test malformed response recovery
    response2 = "1. 사실형, 긍정, 현재, 확실\n2 추론형부정과거불확실\n3. 대화형 미정 미래 확실"
    result2 = handler.try_multiple_parsing_strategies(response2)
    print(f"Malformed recovery result: {result2}")
    
    # Test partial result extraction
    partial = handler.extract_partial_results(response2)
    print(f"Partial extraction confidence: {partial.confidence:.2f}")
    
    # Test enhanced response format recovery
    messy_response = "1 사실 positive 현재 certain\n2. 추론형부정과거불확실"
    enhanced = handler.enhance_response_format_recovery(messy_response, "parsing error")
    print(f"Enhanced recovery result: {enhanced}")
    
    # Test intelligent recovery
    complex_response = "답: 첫번째는 사실형이고 긍정적이며 현재시제로 확실합니다. 두번째는 추론형 부정 과거 불확실"
    intelligent_result = handler.try_multiple_parsing_strategies(complex_response)
    print(f"Intelligent recovery result: {intelligent_result}")
    
    return True

def test_api_error_handler():
    """Test APIErrorHandler functionality"""
    print("\nTesting APIErrorHandler...")
    
    from services.api_error_handler import APIErrorHandler
    from models.exceptions import RateLimitError
    
    api_handler = APIErrorHandler(max_retries=3)
    print(f"API handler initialized with max_retries: {api_handler.max_retries}")
    
    # Test error classification
    rate_error = RateLimitError("Rate limit exceeded")
    error_type = api_handler.classify_error(rate_error)
    print(f"Rate limit error classified as: {error_type}")
    
    # Test exponential backoff
    backoff_time = api_handler.exponential_backoff(2)
    print(f"Backoff time for attempt 2: {backoff_time:.2f}s")
    
    return True

def test_checkpoint_manager():
    """Test CheckpointManager functionality"""
    print("\nTesting CheckpointManager...")
    
    from services.checkpoint_manager import CheckpointManager
    
    checkpoint_mgr = CheckpointManager(checkpoint_dir="test_checkpoints")
    
    # Create and save a checkpoint
    checkpoint_data = checkpoint_mgr.create_checkpoint(
        iteration=5,
        current_prompt="test prompt",
        best_score=0.75,
        best_prompt="best prompt",
        optimization_history=[{"iteration": 1, "score": 0.6}],
        progress_percentage=50.0
    )
    
    success = checkpoint_mgr.save_checkpoint(checkpoint_data, force=True)
    print(f"Checkpoint saved: {success}")
    
    # Load checkpoint
    loaded = checkpoint_mgr.load_latest_checkpoint()
    if loaded:
        print(f"Checkpoint loaded: iteration={loaded.iteration}, score={loaded.best_score}")
    
    return True

def test_error_recovery_system():
    """Test ErrorRecoverySystem functionality"""
    print("\nTesting ErrorRecoverySystem...")
    
    from services.error_recovery_system import ErrorRecoverySystem
    
    recovery_system = ErrorRecoverySystem(checkpoint_dir="test_checkpoints")
    
    # Test successful operation
    def test_operation(value=10):
        return value * 2
    
    result = recovery_system.execute_with_recovery(
        operation=test_operation,
        operation_name="test_operation",
        context={"value": 5},
        max_attempts=3
    )
    print(f"Successful operation result: {result}")
    
    # Get system status
    status = recovery_system.get_system_status()
    print(f"System health: {status['system_health']}")
    
    # Test comprehensive system report
    report = recovery_system.get_comprehensive_system_report()
    print(f"System performance metrics: {report.get('performance_metrics', {})}")
    
    return True

def main():
    """Run all tests"""
    print("=== Testing Advanced Error Handling System ===")
    
    try:
        test_advanced_parsing_handler()
        test_api_error_handler()
        test_checkpoint_manager()
        test_error_recovery_system()
        
        print("\n=== All Error Handling Components Working Successfully ===")
        return True
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)