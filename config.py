"""
Configuration management for Gemini Prompt Optimizer
"""
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class OptimizationConfig:
    """Configuration class for optimization parameters"""
    
    # API Configuration
    gemini_api_key: str
    
    # Optimization Parameters
    target_accuracy: float = 0.95
    max_iterations: int = 10
    batch_size: int = 50
    api_retry_count: int = 3
    
    # File Paths
    samples_csv_path: str = "data/samples.csv"
    prompt_dir: str = "prompt"
    analysis_dir: str = "analysis"
    
    # Model Configuration
    flash_model: str = "gemini-2.5-flash"  # Gemini 2.5 Flash
    pro_model: str = "gemini-2.5-pro"      # Gemini 2.5 Pro
    temperature: float = 0.4
    
    # Convergence Settings
    convergence_threshold: float = 0.001  # Stop if improvement < threshold
    patience: int = 3  # Stop if no improvement for N iterations
    
    @classmethod
    def from_env(cls) -> 'OptimizationConfig':
        """Create configuration from environment variables"""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        return cls(
            gemini_api_key=api_key,
            target_accuracy=float(os.getenv('TARGET_ACCURACY', '0.95')),
            max_iterations=int(os.getenv('MAX_ITERATIONS', '10')),
            batch_size=int(os.getenv('BATCH_SIZE', '50')),
            api_retry_count=int(os.getenv('API_RETRY_COUNT', '3')),
            samples_csv_path=os.getenv('SAMPLES_CSV_PATH', 'data/samples.csv'),
            prompt_dir=os.getenv('PROMPT_DIR', 'prompt'),
            analysis_dir=os.getenv('ANALYSIS_DIR', 'analysis')
        )
    
    def validate(self) -> None:
        """Validate configuration parameters"""
        if not self.gemini_api_key:
            raise ValueError("Gemini API key is required")
        
        if not 0 < self.target_accuracy <= 1:
            raise ValueError("Target accuracy must be between 0 and 1")
        
        if self.max_iterations <= 0:
            raise ValueError("Max iterations must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if not os.path.exists(self.samples_csv_path):
            raise FileNotFoundError(f"Samples CSV file not found: {self.samples_csv_path}")
    
    def create_directories(self) -> None:
        """Create necessary directories if they don't exist"""
        os.makedirs(self.prompt_dir, exist_ok=True)
        os.makedirs(self.analysis_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.samples_csv_path), exist_ok=True)