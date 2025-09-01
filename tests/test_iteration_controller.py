"""
Unit tests for iteration controller
"""
import unittest
from unittest.mock import Mock
from services.iteration_controller import IterationController
from models.data_models import IterationState
from config import OptimizationConfig

class TestIterationController(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock config
        self.config = Mock(spec=OptimizationConfig)
        self.config.target_accuracy = 0.95
        self.config.max_iterations = 10
        self.config.convergence_threshold = 0.001
        self.config.patience = 3
        
        self.controller = IterationController(self.config)
    
    def test_initialize_optimization(self):
        """Test optimization initialization"""
        total_samples = 100
        state = self.controller.initialize_optimization(total_samples)
        
        self.assertIsInstance(state, IterationState)
        self.assertEqual(state.total_samples, total_samples)
        self.assertEqual(state.target_accuracy, 0.95)
        self.assertEqual(state.iteration, 0)
        self.assertFalse(state.is_converged)
    
    def test_start_iteration(self):
        """Test starting an iteration"""
        self.controller.initialize_optimization(100)
        self.controller.start_iteration(1)
        
        self.assertEqual(self.controller.current_state.iteration, 1)
    
    def test_update_results_improvement(self):
        """Test updating results with improvement"""
        self.controller.initialize_optimization(100)
        self.controller.start_iteration(1)
        
        # First update
        self.controller.update_results(0.8, 80, 20)
        self.assertEqual(self.controller.current_state.current_accuracy, 0.8)
        self.assertEqual(self.controller.current_state.best_accuracy, 0.8)
        self.assertEqual(self.controller.no_improvement_count, 0)
        
        # Second update with improvement
        self.controller.update_results(0.85, 85, 15)
        self.assertEqual(self.controller.current_state.best_accuracy, 0.85)
        self.assertEqual(self.controller.no_improvement_count, 0)
    
    def test_update_results_no_improvement(self):
        """Test updating results without improvement"""
        self.controller.initialize_optimization(100)
        self.controller.start_iteration(1)
        
        # First update
        self.controller.update_results(0.8, 80, 20)
        
        # Second update without significant improvement
        self.controller.update_results(0.8005, 80, 20)  # Very small improvement
        self.assertEqual(self.controller.no_improvement_count, 1)
    
    def test_check_convergence_target_reached(self):
        """Test convergence when target is reached"""
        self.controller.initialize_optimization(100)
        self.controller.start_iteration(1)
        self.controller.update_results(0.96, 96, 4)  # Above target
        
        converged = self.controller.check_convergence()
        self.assertTrue(converged)
        self.assertTrue(self.controller.current_state.is_converged)
    
    def test_check_convergence_max_iterations(self):
        """Test convergence when max iterations reached"""
        self.controller.initialize_optimization(100)
        
        # Simulate reaching max iterations
        for i in range(1, 11):  # 10 iterations
            self.controller.start_iteration(i)
            self.controller.update_results(0.8, 80, 20)
        
        converged = self.controller.check_convergence()
        self.assertTrue(converged)
    
    def test_check_convergence_early_stopping(self):
        """Test convergence with early stopping"""
        self.controller.initialize_optimization(100)
        
        # Simulate no improvement for patience iterations
        for i in range(1, 5):  # 4 iterations
            self.controller.start_iteration(i)
            self.controller.update_results(0.8, 80, 20)  # Same accuracy
        
        # Should trigger early stopping after patience (3) iterations
        converged = self.controller.check_convergence()
        self.assertTrue(converged)
    
    def test_should_continue(self):
        """Test should continue logic"""
        self.controller.initialize_optimization(100)
        self.controller.start_iteration(1)
        self.controller.update_results(0.8, 80, 20)
        
        # Should continue when not converged
        should_continue = self.controller.should_continue()
        self.assertTrue(should_continue)
        
        # Should not continue when target reached
        self.controller.update_results(0.96, 96, 4)
        should_continue = self.controller.should_continue()
        self.assertFalse(should_continue)
    
    def test_get_progress_info(self):
        """Test progress info generation"""
        self.controller.initialize_optimization(100)
        self.controller.start_iteration(1)
        self.controller.update_results(0.8, 80, 20)
        
        progress_info = self.controller.get_progress_info()
        
        self.assertIn("Iteration: 1", progress_info)
        self.assertIn("Current Accuracy: 0.8000", progress_info)
        self.assertIn("Target Accuracy: 0.9500", progress_info)
    
    def test_get_iteration_summary(self):
        """Test iteration summary"""
        self.controller.initialize_optimization(100)
        self.controller.start_iteration(1)
        self.controller.update_results(0.8, 80, 20)
        
        summary = self.controller.get_iteration_summary()
        
        self.assertEqual(summary["iteration"], 1)
        self.assertEqual(summary["current_accuracy"], 0.8)
        self.assertEqual(summary["target_accuracy"], 0.95)
        self.assertEqual(summary["error_count"], 20)
        self.assertFalse(summary["target_reached"])
    
    def test_finalize_optimization(self):
        """Test optimization finalization"""
        self.controller.initialize_optimization(100)
        self.controller.start_iteration(1)
        self.controller.update_results(0.8, 80, 20)
        
        result = self.controller.finalize_optimization("test_prompt.txt")
        
        self.assertEqual(result.final_accuracy, 0.8)
        self.assertEqual(result.total_iterations, 1)
        self.assertEqual(result.final_prompt_path, "test_prompt.txt")
        self.assertGreater(result.execution_time, 0)
    
    def test_get_optimization_statistics(self):
        """Test optimization statistics"""
        self.controller.initialize_optimization(100)
        
        # Simulate multiple iterations
        accuracies = [0.7, 0.75, 0.8, 0.85]
        for i, acc in enumerate(accuracies, 1):
            self.controller.start_iteration(i)
            correct = int(acc * 100)
            self.controller.update_results(acc, correct, 100 - correct)
        
        stats = self.controller.get_optimization_statistics()
        
        self.assertEqual(stats["total_iterations"], 4)
        self.assertEqual(stats["initial_accuracy"], 0.7)
        self.assertEqual(stats["final_accuracy"], 0.85)
        self.assertEqual(stats["best_accuracy"], 0.85)
        self.assertGreater(stats["total_improvement"], 0)
    
    def test_reset_optimization(self):
        """Test optimization reset"""
        self.controller.initialize_optimization(100)
        self.controller.start_iteration(1)
        self.controller.update_results(0.8, 80, 20)
        
        self.controller.reset_optimization()
        
        self.assertIsNone(self.controller.current_state)
        self.assertEqual(len(self.controller.iteration_history), 0)
        self.assertEqual(self.controller.no_improvement_count, 0)
    
    def test_set_best_prompt_path(self):
        """Test setting best prompt path"""
        self.controller.initialize_optimization(100)
        self.controller.start_iteration(1)
        
        self.controller.set_best_prompt_path("best_prompt.txt")
        
        self.assertEqual(self.controller.best_prompt_path, "best_prompt.txt")
        self.assertEqual(self.controller.current_state.best_prompt_version, 1)
    
    def test_get_convergence_info(self):
        """Test convergence information"""
        self.controller.initialize_optimization(100)
        self.controller.start_iteration(1)
        self.controller.update_results(0.8, 80, 20)
        
        conv_info = self.controller.get_convergence_info()
        
        self.assertFalse(conv_info["is_converged"])
        self.assertFalse(conv_info["target_reached"])
        self.assertEqual(conv_info["no_improvement_count"], 0)
        self.assertEqual(conv_info["patience"], 3)
        self.assertEqual(conv_info["iterations_remaining"], 9)
    
    def test_estimate_remaining_time(self):
        """Test remaining time estimation"""
        self.controller.initialize_optimization(100)
        
        # Simulate some iterations
        import time
        start_time = time.time()
        for i in range(1, 4):
            self.controller.start_iteration(i)
            self.controller.update_results(0.8, 80, 20)
            time.sleep(0.01)  # Small delay to simulate processing time
        
        remaining_time = self.controller.estimate_remaining_time()
        
        # Should be positive (some time remaining)
        self.assertGreaterEqual(remaining_time, 0)
    
    def test_get_performance_trend(self):
        """Test performance trend analysis"""
        self.controller.initialize_optimization(100)
        
        # Simulate improving trend
        accuracies = [0.7, 0.75, 0.8]
        for i, acc in enumerate(accuracies, 1):
            self.controller.start_iteration(i)
            correct = int(acc * 100)
            self.controller.update_results(acc, correct, 100 - correct)
        
        trend = self.controller.get_performance_trend()
        self.assertEqual(trend, "improving")
        
        # Simulate stable trend
        for i in range(4, 7):
            self.controller.start_iteration(i)
            self.controller.update_results(0.8, 80, 20)  # Same accuracy
        
        trend = self.controller.get_performance_trend()
        self.assertEqual(trend, "stable")

if __name__ == '__main__':
    unittest.main()