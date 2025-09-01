#!/usr/bin/env python3
"""
Advanced Error Recovery System Demo

This example demonstrates the comprehensive error handling and recovery capabilities
of the advanced prompt optimizer system.
"""
import sys
import time
import random
sys.path.append('.')

from services.error_recovery_system import ErrorRecoverySystem, with_error_recovery
from services.advanced_parsing_handler import AdvancedParsingHandler
from services.api_error_handler import APIErrorHandler
from services.checkpoint_manager import CheckpointManager
from models.exceptions import APIError, InvalidResponseError, RateLimitError

def simulate_api_call_with_errors(attempt_count=0):
    """Simulate an API call that might fail in various ways"""
    
    # Simulate different types of failures based on attempt
    if attempt_count == 0:
        # First attempt: Rate limit error
        raise RateLimitError("API rate limit exceeded, please wait")
    
    elif attempt_count == 1:
        # Second attempt: Invalid response format
        error = InvalidResponseError("Invalid response format")
        error.response = "1. 사실형 긍정 현재 확실\n2 추론형부정과거불확실\n3. incomplete response"
        raise error
    
    elif attempt_count == 2:
        # Third attempt: Network error
        raise APIError("Network connection timeout")
    
    elif attempt_count == 3:
        # Fourth attempt: Success with good response
        return [
            "사실형,긍정,현재,확실",
            "추론형,부정,과거,불확실", 
            "대화형,미정,미래,확실"
        ]
    
    else:
        # Subsequent attempts: Success
        return ["사실형,긍정,현재,확실"]

def demonstrate_parsing_recovery():
    """Demonstrate advanced parsing recovery capabilities"""
    print("=== Parsing Recovery Demonstration ===")
    
    handler = AdvancedParsingHandler()
    
    # Test various malformed responses
    test_cases = [
        {
            "name": "Mixed separators",
            "response": "1. 사실형, 긍정; 현재| 확실\n2. 추론형 부정 과거 불확실"
        },
        {
            "name": "English mixed with Korean",
            "response": "1. fact,positive,present,certain\n2. 추론형,부정,과거,불확실"
        },
        {
            "name": "Incomplete classifications",
            "response": "1. 사실형,긍정\n2. 추론형,부정,과거\n3. 대화형,미정,미래,확실"
        },
        {
            "name": "Verbose response",
            "response": "답변: 첫 번째 문장은 사실형이며 긍정적이고 현재 시제로 확실합니다.\n두 번째는 추론형 부정 과거 불확실한 특성을 가집니다."
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        print(f"Input: {test_case['response'][:100]}...")
        
        # Try standard parsing first
        standard_result = handler.try_multiple_parsing_strategies(test_case['response'])
        print(f"Standard parsing: {standard_result}")
        
        # Try enhanced recovery
        enhanced_response = handler.enhance_response_format_recovery(
            test_case['response'], 
            "parsing_error"
        )
        enhanced_result = handler.try_multiple_parsing_strategies(enhanced_response)
        print(f"Enhanced recovery: {enhanced_result}")
        
        # Show partial extraction
        partial = handler.extract_partial_results(test_case['response'])
        print(f"Partial extraction: {len(partial.parsed_items)} items, confidence: {partial.confidence:.2f}")

def demonstrate_api_error_handling():
    """Demonstrate API error handling with adaptive retry strategies"""
    print("\n=== API Error Handling Demonstration ===")
    
    api_handler = APIErrorHandler(max_retries=5)
    
    # Simulate error history for adaptive strategy testing
    error_history = [
        "rate_limit_error_1",
        "parsing_error_1", 
        "rate_limit_error_2",
        "network_error_1",
        "parsing_error_2"
    ]
    
    # Test adaptive retry strategies for different error types
    error_types = ["rate_limit", "parsing", "network", "unknown"]
    
    for error_type in error_types:
        print(f"\nTesting adaptive strategy for: {error_type}")
        
        for retry_count in range(3):
            strategy = api_handler.adaptive_retry_strategy(error_type, retry_count, error_history)
            print(f"  Attempt {retry_count + 1}: wait={strategy['wait_time']:.2f}s, "
                  f"retry={strategy['should_retry']}, modifications={strategy['modifications']}")

def demonstrate_checkpoint_recovery():
    """Demonstrate checkpoint-based recovery"""
    print("\n=== Checkpoint Recovery Demonstration ===")
    
    checkpoint_mgr = CheckpointManager(checkpoint_dir="demo_checkpoints")
    
    # Create a series of checkpoints simulating optimization progress
    optimization_history = []
    
    for iteration in range(1, 6):
        score = 0.5 + (iteration * 0.05) + random.uniform(-0.02, 0.02)
        
        checkpoint_data = checkpoint_mgr.create_checkpoint(
            iteration=iteration,
            current_prompt=f"prompt_v{iteration}",
            best_score=score,
            best_prompt=f"best_prompt_v{iteration}",
            optimization_history=optimization_history.copy(),
            error_count=random.randint(0, 3),
            progress_percentage=iteration * 20
        )
        
        optimization_history.append({
            "iteration": iteration,
            "score": score,
            "timestamp": time.time()
        })
        
        success = checkpoint_mgr.save_checkpoint(checkpoint_data, force=True)
        print(f"Checkpoint {iteration}: score={score:.3f}, saved={success}")
        
        time.sleep(0.1)  # Small delay for different timestamps
    
    # Demonstrate checkpoint statistics
    stats = checkpoint_mgr.get_checkpoint_statistics()
    print(f"\nCheckpoint Statistics:")
    print(f"  Total checkpoints: {stats['total_checkpoints']}")
    print(f"  Best score: {stats['best_score_checkpoint']['best_score']:.3f}")
    print(f"  Average score: {stats['average_score']:.3f}")
    print(f"  Score trend: {stats['score_trend']}")
    print(f"  Storage usage: {stats['storage_usage']} bytes")

@with_error_recovery("demo_operation", max_attempts=5)
def demonstrate_full_recovery_system():
    """Demonstrate the complete error recovery system"""
    print("\n=== Full Recovery System Demonstration ===")
    
    # This function will be wrapped with error recovery
    attempt_count = getattr(demonstrate_full_recovery_system, '_attempt_count', 0)
    demonstrate_full_recovery_system._attempt_count = attempt_count + 1
    
    print(f"Executing operation (attempt {attempt_count + 1})")
    
    # Simulate the API call that might fail
    result = simulate_api_call_with_errors(attempt_count)
    
    print(f"Operation succeeded with result: {result}")
    return result

def demonstrate_comprehensive_system_report():
    """Demonstrate comprehensive system reporting"""
    print("\n=== Comprehensive System Report ===")
    
    recovery_system = ErrorRecoverySystem(checkpoint_dir="demo_checkpoints")
    
    # Generate some activity to show in the report
    for i in range(3):
        recovery_system.api_handler.record_error("network_errors")
        recovery_system.api_handler.record_success()
    
    # Get comprehensive report
    report = recovery_system.get_comprehensive_system_report()
    
    print("System Report:")
    print(f"  System Health: {report['basic_status']['system_health']}")
    print(f"  Error Rate: {report['performance_metrics']['error_rate']:.2%}")
    print(f"  Success Rate: {report['performance_metrics']['success_rate']:.2%}")
    print(f"  Recovery Effectiveness: {report['performance_metrics']['recovery_effectiveness']:.2%}")
    
    if report['system_recommendations']:
        print("  Recommendations:")
        for rec in report['system_recommendations']:
            print(f"    - {rec}")

def main():
    """Run all demonstrations"""
    print("Advanced Error Recovery System Demo")
    print("=" * 50)
    
    try:
        # Run all demonstrations
        demonstrate_parsing_recovery()
        demonstrate_api_error_handling()
        demonstrate_checkpoint_recovery()
        demonstrate_full_recovery_system()
        demonstrate_comprehensive_system_report()
        
        print("\n" + "=" * 50)
        print("All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)