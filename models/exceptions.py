"""
Custom exceptions for Gemini Prompt Optimizer
"""

class GeminiOptimizerError(Exception):
    """Base exception for Gemini Optimizer"""
    pass

class APIError(GeminiOptimizerError):
    """API related errors"""
    def __init__(self, message: str, status_code: int = None, retry_after: int = None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after

class ValidationError(GeminiOptimizerError):
    """Data validation errors"""
    pass

class ConfigurationError(GeminiOptimizerError):
    """Configuration related errors"""
    pass

class FileProcessingError(GeminiOptimizerError):
    """File processing errors"""
    pass

class PromptOptimizationError(GeminiOptimizerError):
    """Prompt optimization specific errors"""
    pass

class ConvergenceError(GeminiOptimizerError):
    """Convergence related errors"""
    pass

class RateLimitError(APIError):
    """Rate limit exceeded error"""
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = 60):
        super().__init__(message, status_code=429, retry_after=retry_after)

class QuotaExceededError(APIError):
    """API quota exceeded error"""
    def __init__(self, message: str = "API quota exceeded"):
        super().__init__(message, status_code=403)

class InvalidResponseError(APIError):
    """Invalid API response error"""
    def __init__(self, message: str = "Invalid API response format"):
        super().__init__(message, status_code=422)