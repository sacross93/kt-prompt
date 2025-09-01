"""
Unit tests for configuration
"""
import unittest
import tempfile
import os
from unittest.mock import patch
from config import OptimizationConfig

class TestOptimizationConfig(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary CSV file for testing
        self.temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.temp_csv.write("id,sentence,type,polarity,tense,certainty\n")
        self.temp_csv.write("1,test,사실형,긍정,현재,확실\n")
        self.temp_csv.close()
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_csv.name):
            os.unlink(self.temp_csv.name)
    
    def test_config_creation(self):
        """Test basic config creation"""
        config = OptimizationConfig(
            gemini_api_key="test_key",
            samples_csv_path=self.temp_csv.name
        )
        
        self.assertEqual(config.gemini_api_key, "test_key")
        self.assertEqual(config.target_accuracy, 0.95)
        self.assertEqual(config.max_iterations, 10)
        self.assertEqual(config.batch_size, 50)
    
    def test_config_with_custom_values(self):
        """Test config with custom values"""
        config = OptimizationConfig(
            gemini_api_key="test_key",
            target_accuracy=0.9,
            max_iterations=5,
            batch_size=25,
            samples_csv_path=self.temp_csv.name
        )
        
        self.assertEqual(config.target_accuracy, 0.9)
        self.assertEqual(config.max_iterations, 5)
        self.assertEqual(config.batch_size, 25)
    
    @patch.dict(os.environ, {
        'GEMINI_API_KEY': 'env_test_key',
        'TARGET_ACCURACY': '0.98',
        'MAX_ITERATIONS': '15',
        'BATCH_SIZE': '100',
        'SAMPLES_CSV_PATH': 'test.csv'
    })
    def test_from_env(self):
        """Test config creation from environment variables"""
        # Create temporary file for the path
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            temp_file.write("test,data\n")
            temp_path = temp_file.name
        
        try:
            with patch.dict(os.environ, {'SAMPLES_CSV_PATH': temp_path}):
                config = OptimizationConfig.from_env()
            
            self.assertEqual(config.gemini_api_key, 'env_test_key')
            self.assertEqual(config.target_accuracy, 0.98)
            self.assertEqual(config.max_iterations, 15)
            self.assertEqual(config.batch_size, 100)
        finally:
            os.unlink(temp_path)
    
    def test_from_env_missing_api_key(self):
        """Test config creation with missing API key"""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                OptimizationConfig.from_env()
    
    def test_validate_success(self):
        """Test successful validation"""
        config = OptimizationConfig(
            gemini_api_key="test_key",
            samples_csv_path=self.temp_csv.name
        )
        
        # Should not raise any exception
        config.validate()
    
    def test_validate_missing_api_key(self):
        """Test validation with missing API key"""
        config = OptimizationConfig(
            gemini_api_key="",
            samples_csv_path=self.temp_csv.name
        )
        
        with self.assertRaises(ValueError):
            config.validate()
    
    def test_validate_invalid_target_accuracy(self):
        """Test validation with invalid target accuracy"""
        config = OptimizationConfig(
            gemini_api_key="test_key",
            target_accuracy=1.5,  # Invalid: > 1
            samples_csv_path=self.temp_csv.name
        )
        
        with self.assertRaises(ValueError):
            config.validate()
        
        config.target_accuracy = -0.1  # Invalid: < 0
        with self.assertRaises(ValueError):
            config.validate()
    
    def test_validate_invalid_max_iterations(self):
        """Test validation with invalid max iterations"""
        config = OptimizationConfig(
            gemini_api_key="test_key",
            max_iterations=0,  # Invalid: <= 0
            samples_csv_path=self.temp_csv.name
        )
        
        with self.assertRaises(ValueError):
            config.validate()
    
    def test_validate_invalid_batch_size(self):
        """Test validation with invalid batch size"""
        config = OptimizationConfig(
            gemini_api_key="test_key",
            batch_size=-1,  # Invalid: <= 0
            samples_csv_path=self.temp_csv.name
        )
        
        with self.assertRaises(ValueError):
            config.validate()
    
    def test_validate_missing_csv_file(self):
        """Test validation with missing CSV file"""
        config = OptimizationConfig(
            gemini_api_key="test_key",
            samples_csv_path="nonexistent.csv"
        )
        
        with self.assertRaises(FileNotFoundError):
            config.validate()
    
    def test_create_directories(self):
        """Test directory creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = OptimizationConfig(
                gemini_api_key="test_key",
                prompt_dir=os.path.join(temp_dir, "prompts"),
                analysis_dir=os.path.join(temp_dir, "analysis"),
                samples_csv_path=self.temp_csv.name
            )
            
            config.create_directories()
            
            self.assertTrue(os.path.exists(config.prompt_dir))
            self.assertTrue(os.path.exists(config.analysis_dir))
    
    def test_default_values(self):
        """Test default configuration values"""
        config = OptimizationConfig(
            gemini_api_key="test_key",
            samples_csv_path=self.temp_csv.name
        )
        
        self.assertEqual(config.target_accuracy, 0.95)
        self.assertEqual(config.max_iterations, 10)
        self.assertEqual(config.batch_size, 50)
        self.assertEqual(config.api_retry_count, 3)
        self.assertEqual(config.flash_model, "gemini-1.5-flash")
        self.assertEqual(config.pro_model, "gemini-1.5-pro")
        self.assertEqual(config.convergence_threshold, 0.001)
        self.assertEqual(config.patience, 3)
    
    def test_model_configuration(self):
        """Test model configuration"""
        config = OptimizationConfig(
            gemini_api_key="test_key",
            flash_model="custom-flash-model",
            pro_model="custom-pro-model",
            samples_csv_path=self.temp_csv.name
        )
        
        self.assertEqual(config.flash_model, "custom-flash-model")
        self.assertEqual(config.pro_model, "custom-pro-model")
    
    def test_convergence_settings(self):
        """Test convergence settings"""
        config = OptimizationConfig(
            gemini_api_key="test_key",
            convergence_threshold=0.005,
            patience=5,
            samples_csv_path=self.temp_csv.name
        )
        
        self.assertEqual(config.convergence_threshold, 0.005)
        self.assertEqual(config.patience, 5)

if __name__ == '__main__':
    unittest.main()