"""
Logging utilities for Gemini Prompt Optimizer
"""
import logging
import sys
from datetime import datetime
from typing import Optional
from pathlib import Path

class ColoredFormatter(logging.Formatter):
    """Colored log formatter for console output"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        return super().format(record)

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """Setup logging configuration"""
    
    # Create logger
    logger = logging.getLogger("gemini_optimizer")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def log_iteration_start(logger: logging.Logger, iteration: int, target_accuracy: float):
    """Log iteration start"""
    logger.info(f"{'='*50}")
    logger.info(f"Starting Iteration {iteration}")
    logger.info(f"Target Accuracy: {target_accuracy:.4f}")
    logger.info(f"{'='*50}")

def log_iteration_result(logger: logging.Logger, iteration: int, accuracy: float, 
                        error_count: int, total_samples: int):
    """Log iteration result"""
    logger.info(f"Iteration {iteration} Results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Correct: {total_samples - error_count}/{total_samples}")
    logger.info(f"  Errors: {error_count}")

def log_analysis_summary(logger: logging.Logger, analysis_report):
    """Log analysis summary"""
    logger.info("Analysis Summary:")
    logger.info(f"  Total Errors: {analysis_report.total_errors}")
    logger.info(f"  Error Patterns: {analysis_report.error_patterns}")
    logger.info(f"  Confidence Score: {analysis_report.confidence_score:.2f}")
    logger.info(f"  Suggestions: {len(analysis_report.improvement_suggestions)}")

def log_prompt_update(logger: logging.Logger, version: int, changes: list):
    """Log prompt update"""
    logger.info(f"Prompt updated to version {version}")
    for change in changes:
        logger.info(f"  - {change}")

def log_convergence(logger: logging.Logger, final_accuracy: float, iterations: int):
    """Log convergence achievement"""
    logger.info(f"{'='*50}")
    logger.info("CONVERGENCE ACHIEVED!")
    logger.info(f"Final Accuracy: {final_accuracy:.4f}")
    logger.info(f"Total Iterations: {iterations}")
    logger.info(f"{'='*50}")

def log_optimization_complete(logger: logging.Logger, result):
    """Log optimization completion"""
    logger.info(f"{'='*60}")
    logger.info("OPTIMIZATION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Final Accuracy: {result.final_accuracy:.4f}")
    logger.info(f"Best Accuracy: {result.best_accuracy:.4f}")
    logger.info(f"Best Prompt Version: {result.best_prompt_version}")
    logger.info(f"Total Iterations: {result.total_iterations}")
    logger.info(f"Execution Time: {result.execution_time:.2f} seconds")
    logger.info(f"Final Prompt: {result.final_prompt_path}")
    logger.info(f"{'='*60}")

def log_error(logger: logging.Logger, error: Exception, context: str = ""):
    """Log error with context"""
    if context:
        logger.error(f"Error in {context}: {str(error)}")
    else:
        logger.error(f"Error: {str(error)}")
    
    # Log stack trace for debugging
    logger.debug("Stack trace:", exc_info=True)