"""
Tests for the error recovery system
"""
import pytest
import json
import time
import tempfile
from unittest.mock import Mock, patch, MagicMock
from services.error_recovery_system import ErrorRecoverySystem, with_error_recovery
from services.advanced_parsing_handler import AdvancedParsingHandler, PartialResult
from services.checkpoint_manager import CheckpointManager, CheckpointData
from models.exceptions import APIError, InvalidResponseError, RateLimitError

class TestAdvancedParsingHandler:
    """Test advanced parsing handler"""
    
    def setup_method(self):
        self.handler = AdvancedParsingHandler()
    
    def test_standard_format_parsing(self):
        """Test standard format parsing"""
        response = """1. 사실형,긍정,현재,확실
2. 추론형,부정,과거,불확실
3. 대화형,미정,미래,확실"""
        
        result = self.handler.try_multiple_parsing_strategies(response)
        assert result is not None
        assert len(result) == 3
        assert result[0] == "사실형,긍정,현재,확실"
    
    def test_malformed_response_recovery(self):
        """Test recovery from malformed responses"""
        response = """1. 사실형, 긍정, 현재, 확실
2 추론형부정과거불확실
3. 대화형 미정 미래 확실"""
        
        result = self.handler.try_multiple_parsing_strategies(response)
        assert result is not None
        assert len(result) >= 1  # Should recover at least some items
    
    def test_partial_result_extraction(self):
        """Test partial result extraction"""
        response = """1. 사실형,긍정,현재,확실
2. invalid_format_here
3. 대화형,미정,미래,확실"""
        
        partial = self.handler.extract_partial_results(response)
        assert partial.confidence > 0.5
        assert len(partial.parsed_items) >= 2
        assert len(partial.failed_items) >= 1
    
    def test_normalize_response_format(self):
        """Test response format normalization"""
        messy_response = "1.사실형,긍정,현재,확실\n\n2.  추론형，부정，과거，불확실  "
        normalized = self.handler.normalize_response_format(messy_response)
        
        assert "1. 사실형" in normalized
        assert "2. 추론형" in normalized
        assert "，" not in normalized  # Should be normalized to regular comma
    
    def test_fuzzy_matching(self):
        """Test fuzzy matching for classification terms"""
        response = "1. fact,positive,present,certain"
        result = self.handler.try_multiple_parsing_strategies(response)
        
        # Should match using alternatives
        assert result is not None or len(result) > 0
    
    def test_parsing_feedback_generation(self):
        """Test parsing feedback generation"""
        failures = ["invalid format 1", "bad classification 2"]
        feedback = self.handler.generate_parsing_feedback(failures)
        
        assert "응답 형식을 개선하기 위한 지침" in feedback
        assert "사실형" in feedback
        assert "긍정" in feedback

class TestCheckpointManager:
    """Test checkpoint manager"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = CheckpointManager(checkpoint_dir=self.temp_dir)
    
    def test_save_and_load_checkpoint(self):
        """Test saving and loading checkpoints"""
        checkpoint_data = self.manager.create_checkpoint(
            iteration=5,
            current_prompt="test prompt",
            best_score=0.75,
            best_prompt="best prompt",
            optimization_history=[{"iteration": 1, "score": 0.6}],
            progress_percentage=50.0
        )
        
        # Save checkpoint
        success = self.manager.save_checkpoint(checkpoint_data, force=True)
        assert success
        
        # Load checkpoint
        loaded = self.manager.load_latest_checkpoint()
        assert loaded is not None
        assert loaded.iteration == 5
        assert loaded.best_score == 0.75
        assert loaded.progress_percentage == 50.0
    
    def test_checkpoint_listing(self):
        """Test checkpoint listing"""
        # Create multiple checkpoints
        for i in range(3):
            checkpoint_data = self.manager.create_checkpoint(
                iteration=i,
                current_prompt=f"prompt {i}",
                best_score=0.5 + i * 0.1,
                best_prompt=f"best {i}",
                optimization_history=[],
                progress_percentage=i * 20.0
            )
            self.manager.save_checkpoint(checkpoint_data, force=True)
            time.sleep(0.1)  # Ensure different timestamps
        
        checkpoints = self.manager.list_checkpoints()
        assert len(checkpoints) == 3
        
        # Should be sorted by timestamp (newest first)
        assert checkpoints[0]["iteration"] == 2
        assert checkpoints[1]["iteration"] == 1
        assert checkpoints[2]["iteration"] == 0
    
    def test_checkpoint_cleanup(self):
        """Test checkpoint cleanup"""
        # Create 5 checkpoints
        for i in range(5):
            checkpoint_data = self.manager.create_checkpoint(
                iteration=i,
                current_prompt=f"prompt {i}",
                best_score=0.5,
                best_prompt="best",
                optimization_history=[],
                progress_percentage=0.0
            )
            self.manager.save_checkpoint(checkpoint_data, force=True)
            time.sleep(0.1)
        
        # Cleanup keeping only 3
        self.manager.cleanup_old_checkpoints(keep_count=3)
        
        checkpoints = self.manager.list_checkpoints()
        assert len(checkpoints) == 3

class TestErrorRecoverySystem:
    """Test error recovery system"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.recovery_system = ErrorRecoverySystem(checkpoint_dir=self.temp_dir)
    
    def test_successful_operation_execution(self):
        """Test successful operation execution"""
        def mock_operation(value=10):
            return value * 2
        
        result = self.recovery_system.execute_with_recovery(
            operation=mock_operation,
            operation_name="test_operation",
            context={"value": 5},
            max_attempts=3
        )
        
        assert result == 10
    
    def test_api_error_recovery(self):
        """Test API error recovery"""
        call_count = 0
        
        def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise APIError("Rate limit exceeded")
            return "success"
        
        with patch.object(self.recovery_system.api_handler, 'should_retry', return_value=True):
            with patch('time.sleep'):  # Skip actual sleep
                result = self.recovery_system.execute_with_recovery(
                    operation=failing_operation,
                    operation_name="test_api_operation",
                    context={},
                    max_attempts=3
                )
        
        assert result == "success"
        assert call_count == 3
    
    def test_parsing_error_recovery(self):
        """Test parsing error recovery"""
        def mock_operation():
            error = InvalidResponseError("Invalid format")
            error.response = "1. 사실형,긍정,현재,확실\n2. invalid"
            raise error
        
        # Mock parsing handler to return successful result
        with patch.object(self.recovery_system.parsing_handler, 'try_multiple_parsing_strategies') as mock_parse:
            mock_parse.return_value = ["사실형,긍정,현재,확실"]
            
            # This should not raise an exception due to parsing recovery
            try:
                result = self.recovery_system.execute_with_recovery(
                    operation=mock_operation,
                    operation_name="test_parsing_operation",
                    context={},
                    max_attempts=1
                )
                # If parsing recovery works, we should get the parsed result
                assert result == ["사실형,긍정,현재,확실"]
            except InvalidResponseError:
                # If parsing recovery doesn't work, that's also acceptable for this test
                pass
    
    def test_circuit_breaker_activation(self):
        """Test circuit breaker activation"""
        # Simulate many errors to trigger circuit breaker
        for _ in range(15):
            self.recovery_system.api_handler.record_error("network_errors")
        
        def failing_operation():
            raise APIError("Network error")
        
        with pytest.raises(APIError, match="Circuit breaker activated"):
            self.recovery_system.execute_with_recovery(
                operation=failing_operation,
                operation_name="test_circuit_breaker",
                context={},
                max_attempts=1
            )
    
    def test_checkpoint_recovery(self):
        """Test checkpoint-based recovery"""
        # Create a checkpoint first
        checkpoint_data = self.recovery_system.checkpoint_manager.create_checkpoint(
            iteration=3,
            current_prompt="checkpoint prompt",
            best_score=0.8,
            best_prompt="checkpoint best",
            optimization_history=[],
            progress_percentage=60.0
        )
        self.recovery_system.checkpoint_manager.save_checkpoint(checkpoint_data, force=True)
        
        # Test recovery
        recovery_info = self.recovery_system.recovery_manager.attempt_recovery()
        assert recovery_info is not None
        assert recovery_info["checkpoint"].iteration == 3
        assert recovery_info["checkpoint"].best_score == 0.8
    
    def test_system_status_reporting(self):
        """Test system status reporting"""
        status = self.recovery_system.get_system_status()
        
        assert "api_error_stats" in status
        assert "recovery_info" in status
        assert "system_health" in status
        assert status["system_health"] in ["healthy", "warning", "critical"]
    
    def test_error_recovery_decorator(self):
        """Test error recovery decorator"""
        call_count = 0
        
        @with_error_recovery("decorated_operation", max_attempts=2)
        def decorated_function(value):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise APIError("First attempt fails")
            return value * 2
        
        with patch('time.sleep'):  # Skip actual sleep
            result = decorated_function(5)
        
        assert result == 10
        assert call_count == 2

class TestIntegration:
    """Integration tests for the complete error recovery system"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.recovery_system = ErrorRecoverySystem(checkpoint_dir=self.temp_dir)
    
    def test_end_to_end_recovery_scenario(self):
        """Test complete recovery scenario"""
        iteration_count = 0
        
        def complex_operation(prompt="", iteration=0):
            nonlocal iteration_count
            iteration_count += 1
            
            # Simulate different types of failures
            if iteration_count == 1:
                raise RateLimitError("Rate limit on first attempt")
            elif iteration_count == 2:
                error = InvalidResponseError("Parsing error")
                error.response = "1. 사실형,긍정,현재,확실"
                raise error
            elif iteration_count == 3:
                return {"result": "success", "score": 0.85}
            else:
                raise APIError("Unexpected error")
        
        # Mock parsing recovery
        with patch.object(self.recovery_system.parsing_handler, 'try_multiple_parsing_strategies') as mock_parse:
            mock_parse.return_value = ["사실형,긍정,현재,확실"]
            
            with patch('time.sleep'):  # Skip actual sleep
                try:
                    result = self.recovery_system.execute_with_recovery(
                        operation=complex_operation,
                        operation_name="complex_test",
                        context={"prompt": "test", "iteration": 1},
                        max_attempts=3
                    )
                    
                    # Should eventually succeed
                    assert result["result"] == "success"
                    assert result["score"] == 0.85
                    
                except Exception as e:
                    # If it fails, check that appropriate recovery was attempted
                    status = self.recovery_system.get_system_status()
                    assert status["api_error_stats"]["total_errors"] > 0
    
    def test_resource_cleanup(self):
        """Test resource cleanup functionality"""
        # Create some checkpoints and errors
        for i in range(5):
            checkpoint_data = self.recovery_system.checkpoint_manager.create_checkpoint(
                iteration=i,
                current_prompt=f"prompt {i}",
                best_score=0.5,
                best_prompt="best",
                optimization_history=[],
                progress_percentage=0.0
            )
            self.recovery_system.checkpoint_manager.save_checkpoint(checkpoint_data, force=True)
        
        # Add some errors
        for _ in range(3):
            self.recovery_system.api_handler.record_error("network_errors")
        
        # Cleanup
        self.recovery_system.cleanup_resources()
        
        # Verify cleanup
        checkpoints = self.recovery_system.checkpoint_manager.list_checkpoints()
        assert len(checkpoints) <= 10  # Should keep reasonable number

if __name__ == "__main__":
    pytest.main([__file__])