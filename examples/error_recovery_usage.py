#!/usr/bin/env python3
"""
Example usage of the advanced error recovery system
"""
import time
import random
from services.error_recovery_system import ErrorRecoverySystem, with_error_recovery
from services.advanced_parsing_handler import AdvancedParsingHandler
from models.exceptions import APIError, InvalidResponseError, RateLimitError

def simulate_gemini_api_call(prompt: str, attempt: int = 0) -> str:
    """Simulate a Gemini API call that might fail"""
    
    # Simulate different types of failures
    failure_chance = random.random()
    
    if attempt == 0 and failure_chance < 0.3:
        # 30% chance of rate limit on first attempt
        raise RateLimitError("Rate limit exceeded, please try again later")
    
    elif attempt == 1 and failure_chance < 0.2:
        # 20% chance of network error on second attempt
        raise APIError("Network connection timeout")
    
    elif failure_chance < 0.1:
        # 10% chance of malformed response
        error = InvalidResponseError("Invalid response format")
        error.response = """1. 사실형,긍정,현재,확실
2. invalid format here
3. 대화형,미정,미래,확실"""
        raise error
    
    # Success case - return properly formatted response
    return """1. 사실형,긍정,현재,확실
2. 추론형,부정,과거,불확실
3. 대화형,미정,미래,확실
4. 예측형,긍정,미래,불확실"""

def example_basic_usage():
    """Example of basic error recovery usage"""
    print("=== Basic Error Recovery Usage ===")
    
    recovery_system = ErrorRecoverySystem(checkpoint_dir="example_checkpoints")
    
    def test_operation(prompt="test prompt", iteration=1):
        """Test operation that might fail"""
        print(f"Executing operation with prompt: '{prompt[:30]}...'")
        
        # Simulate API call
        response = simulate_gemini_api_call(prompt, attempt=iteration-1)
        
        # Parse response
        handler = AdvancedParsingHandler()
        parsed = handler.try_multiple_parsing_strategies(response)
        
        if not parsed:
            raise InvalidResponseError("Could not parse response")
        
        return {
            "parsed_classifications": parsed,
            "score": len(parsed) / 4.0,  # Simple scoring
            "iteration": iteration
        }
    
    try:
        result = recovery_system.execute_with_recovery(
            operation=test_operation,
            operation_name="gemini_classification",
            context={
                "prompt": "Classify these Korean sentences...",
                "iteration": 1,
                "best_score": 0.0,
                "optimization_history": []
            },
            max_attempts=3
        )
        
        print(f"✓ Operation succeeded: {len(result['parsed_classifications'])} classifications")
        print(f"  Score: {result['score']:.2f}")
        
    except Exception as e:
        print(f"✗ Operation failed after all retries: {e}")
    
    # Show system status
    status = recovery_system.get_system_status()
    print(f"\nSystem Status: {status['system_health']}")
    print(f"Total errors: {status['api_error_stats']['total_errors']}")

def example_decorator_usage():
    """Example using the decorator for automatic error recovery"""
    print("\n=== Decorator Usage ===")
    
    @with_error_recovery("decorated_operation", max_attempts=3)
    def classify_sentences(sentences, model="gemini-2.5-flash"):
        """Function with automatic error recovery"""
        print(f"Classifying {len(sentences)} sentences with {model}")
        
        # Simulate processing
        response = simulate_gemini_api_call("test prompt")
        
        # Parse response
        handler = AdvancedParsingHandler()
        return handler.try_multiple_parsing_strategies(response)
    
    try:
        sentences = ["문장 1", "문장 2", "문장 3", "문장 4"]
        result = classify_sentences(sentences)
        print(f"✓ Decorated function succeeded: {len(result)} results")
        
    except Exception as e:
        print(f"✗ Decorated function failed: {e}")

def example_checkpoint_recovery():
    """Example of checkpoint-based recovery"""
    print("\n=== Checkpoint Recovery ===")
    
    recovery_system = ErrorRecoverySystem(checkpoint_dir="example_checkpoints")
    
    # Create a checkpoint manually
    checkpoint_data = recovery_system.checkpoint_manager.create_checkpoint(
        iteration=5,
        current_prompt="Advanced prompt with few-shot examples...",
        best_score=0.85,
        best_prompt="Best performing prompt so far...",
        optimization_history=[
            {"iteration": 1, "score": 0.6},
            {"iteration": 2, "score": 0.7},
            {"iteration": 3, "score": 0.75},
            {"iteration": 4, "score": 0.8},
            {"iteration": 5, "score": 0.85}
        ],
        progress_percentage=75.0
    )
    
    # Save checkpoint
    recovery_system.checkpoint_manager.save_checkpoint(checkpoint_data, force=True)
    print("✓ Checkpoint saved")
    
    # Simulate recovery
    recovery_info = recovery_system.recovery_manager.attempt_recovery()
    if recovery_info:
        checkpoint = recovery_info["checkpoint"]
        print(f"✓ Recovery available from iteration {checkpoint.iteration}")
        print(f"  Best score achieved: {checkpoint.best_score:.3f}")
        print(f"  Progress: {checkpoint.progress_percentage:.1f}%")
        
        # Show recommendations
        recommendations = recovery_info["recommendations"]
        if recommendations:
            print("  Recommendations:")
            for rec in recommendations:
                print(f"    - {rec}")
    else:
        print("✗ No recovery available")

def example_parsing_strategies():
    """Example of advanced parsing strategies"""
    print("\n=== Advanced Parsing Strategies ===")
    
    handler = AdvancedParsingHandler()
    
    # Test different malformed responses
    test_responses = [
        # Standard format
        "1. 사실형,긍정,현재,확실\n2. 추론형,부정,과거,불확실",
        
        # Missing commas
        "1. 사실형 긍정 현재 확실\n2. 추론형 부정 과거 불확실",
        
        # Mixed separators
        "1. 사실형，긍정，현재，확실\n2. 추론형|부정|과거|불확실",
        
        # JSON-like format
        '{"1": "사실형,긍정,현재,확실", "2": "추론형,부정,과거,불확실"}',
        
        # Completely malformed
        "1사실형긍정현재확실\n2추론형부정과거불확실\n3. invalid line here",
    ]
    
    for i, response in enumerate(test_responses):
        print(f"\nTest {i+1}: {response[:50]}...")
        
        # Try parsing
        result = handler.try_multiple_parsing_strategies(response)
        if result:
            print(f"  ✓ Parsed {len(result)} items: {result}")
        else:
            # Try partial extraction
            partial = handler.extract_partial_results(response)
            print(f"  ⚠ Partial result: {len(partial.parsed_items)} items "
                  f"({partial.confidence:.2f} confidence)")
            
            if partial.failed_items:
                print(f"    Failed items: {len(partial.failed_items)}")
                
                # Generate feedback
                feedback = handler.generate_parsing_feedback(partial.failed_items)
                print(f"    Feedback generated: {len(feedback)} characters")

def example_error_patterns():
    """Example of handling different error patterns"""
    print("\n=== Error Pattern Handling ===")
    
    recovery_system = ErrorRecoverySystem(checkpoint_dir="example_checkpoints")
    
    # Test different error scenarios
    error_scenarios = [
        ("Rate Limit", lambda: RateLimitError("Rate limit exceeded")),
        ("Network Error", lambda: APIError("Connection timeout")),
        ("Parsing Error", lambda: InvalidResponseError("Invalid format")),
        ("Unknown Error", lambda: Exception("Unexpected error")),
    ]
    
    for scenario_name, error_func in error_scenarios:
        print(f"\nTesting {scenario_name}:")
        
        def failing_operation():
            raise error_func()
        
        try:
            recovery_system.execute_with_recovery(
                operation=failing_operation,
                operation_name=f"test_{scenario_name.lower().replace(' ', '_')}",
                context={},
                max_attempts=1  # Fail quickly for demo
            )
        except Exception as e:
            error_type = recovery_system.api_handler.classify_error(e)
            should_retry = recovery_system.api_handler.should_retry(e, 0)
            
            print(f"  Error type: {error_type}")
            print(f"  Should retry: {should_retry}")
            print(f"  Error: {e}")

def main():
    """Run all examples"""
    print("Advanced Error Recovery System Examples")
    print("=" * 50)
    
    try:
        example_basic_usage()
        example_decorator_usage()
        example_checkpoint_recovery()
        example_parsing_strategies()
        example_error_patterns()
        
        print("\n" + "=" * 50)
        print("🎉 All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()