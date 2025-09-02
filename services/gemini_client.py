"""
Gemini API client for authentication and basic operations
"""
import google.generativeai as genai
from typing import Optional, Dict, Any
import time
import logging
from models.exceptions import APIError, RateLimitError, QuotaExceededError, InvalidResponseError
from config import OptimizationConfig

logger = logging.getLogger("gemini_optimizer.gemini_client")

class GeminiClient:
    """Base Gemini API client with authentication and error handling"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.api_key = config.gemini_api_key
        self._authenticated = False
        self._flash_model = None
        self._pro_model = None
        
        # Configure API
        self._configure_api()
    
    def _configure_api(self) -> None:
        """Configure Gemini API with authentication"""
        try:
            genai.configure(api_key=self.api_key)
            self._authenticated = True
            logger.info("Gemini API configured successfully")
        except Exception as e:
            raise APIError(f"Failed to configure Gemini API: {e}")
    
    def validate_api_key(self) -> bool:
        """Validate API key by making a test request"""
        try:
            # Try to list models to validate API key
            models = list(genai.list_models())
            logger.info(f"API key validation successful. Available models: {len(models)}")
            return True
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return False
    
    def get_flash_model(self) -> genai.GenerativeModel:
        """Get Gemini Flash model instance"""
        if self._flash_model is None:
            try:
                self._flash_model = genai.GenerativeModel(self.config.flash_model)
                logger.debug(f"Initialized Flash model: {self.config.flash_model}")
            except Exception as e:
                raise APIError(f"Failed to initialize Flash model: {e}")
        return self._flash_model
    
    def get_pro_model(self) -> genai.GenerativeModel:
        """Get Gemini Pro model instance"""
        if self._pro_model is None:
            try:
                self._pro_model = genai.GenerativeModel(self.config.pro_model)
                logger.debug(f"Initialized Pro model: {self.config.pro_model}")
            except Exception as e:
                raise APIError(f"Failed to initialize Pro model: {e}")
        return self._pro_model
    
    def generate_content_with_retry(
        self, 
        model: genai.GenerativeModel, 
        prompt: str,
        max_retries: int = None
    ) -> str:
        """Generate content with retry logic and error handling"""
        if max_retries is None:
            max_retries = self.config.api_retry_count
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                logger.debug(f"API call attempt {attempt + 1}/{max_retries + 1}")
                
                # Generate content with temperature setting
                generation_config = genai.types.GenerationConfig(
                    temperature=self.config.temperature
                )
                response = model.generate_content(prompt, generation_config=generation_config)
                
                if not response or not response.text:
                    raise InvalidResponseError("Empty response from API")
                
                logger.debug(f"API call successful on attempt {attempt + 1}")
                return response.text.strip()
                
            except Exception as e:
                last_exception = e
                error_message = str(e).lower()
                
                # Handle specific error types
                if "rate limit" in error_message or "quota" in error_message:
                    if "rate limit" in error_message:
                        wait_time = self._calculate_backoff_time(attempt)
                        logger.warning(f"Rate limit hit, waiting {wait_time} seconds...")
                        if attempt < max_retries:
                            time.sleep(wait_time)
                            continue
                        else:
                            raise RateLimitError(f"Rate limit exceeded after {max_retries} retries")
                    else:
                        raise QuotaExceededError("API quota exceeded")
                
                elif "network" in error_message or "connection" in error_message:
                    wait_time = self._calculate_backoff_time(attempt)
                    logger.warning(f"Network error, retrying in {wait_time} seconds...")
                    if attempt < max_retries:
                        time.sleep(wait_time)
                        continue
                    else:
                        raise APIError(f"Network error after {max_retries} retries: {e}")
                
                else:
                    # For other errors, don't retry
                    logger.error(f"API error on attempt {attempt + 1}: {e}")
                    if attempt < max_retries:
                        wait_time = self._calculate_backoff_time(attempt)
                        time.sleep(wait_time)
                        continue
                    else:
                        raise APIError(f"API call failed after {max_retries} retries: {e}")
        
        # If we get here, all retries failed
        raise APIError(f"All retry attempts failed. Last error: {last_exception}")
    
    def _calculate_backoff_time(self, attempt: int) -> float:
        """Calculate exponential backoff time"""
        base_delay = 1.0
        max_delay = 60.0
        delay = min(base_delay * (2 ** attempt), max_delay)
        return delay
    
    def test_connection(self) -> Dict[str, Any]:
        """Test API connection and return status"""
        test_results = {
            "api_key_valid": False,
            "flash_model_available": False,
            "pro_model_available": False,
            "test_generation_successful": False,
            "error_message": None
        }
        
        try:
            # Test API key
            test_results["api_key_valid"] = self.validate_api_key()
            
            if not test_results["api_key_valid"]:
                test_results["error_message"] = "Invalid API key"
                return test_results
            
            # Test Flash model
            try:
                flash_model = self.get_flash_model()
                test_results["flash_model_available"] = True
                logger.info("Flash model available")
            except Exception as e:
                test_results["error_message"] = f"Flash model error: {e}"
                logger.error(f"Flash model not available: {e}")
            
            # Test Pro model
            try:
                pro_model = self.get_pro_model()
                test_results["pro_model_available"] = True
                logger.info("Pro model available")
            except Exception as e:
                test_results["error_message"] = f"Pro model error: {e}"
                logger.error(f"Pro model not available: {e}")
            
            # Test generation with Flash model
            if test_results["flash_model_available"]:
                try:
                    test_prompt = "테스트입니다. '확인'이라고 답해주세요."
                    response = self.generate_content_with_retry(flash_model, test_prompt)
                    test_results["test_generation_successful"] = True
                    logger.info(f"Test generation successful: {response[:50]}...")
                except Exception as e:
                    test_results["error_message"] = f"Test generation failed: {e}"
                    logger.error(f"Test generation failed: {e}")
            
        except Exception as e:
            test_results["error_message"] = f"Connection test failed: {e}"
            logger.error(f"Connection test failed: {e}")
        
        return test_results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models"""
        try:
            models = list(genai.list_models())
            model_info = {
                "total_models": len(models),
                "flash_model": self.config.flash_model,
                "pro_model": self.config.pro_model,
                "available_models": []
            }
            
            for model in models:
                model_info["available_models"].append({
                    "name": model.name,
                    "display_name": getattr(model, 'display_name', 'N/A'),
                    "description": getattr(model, 'description', 'N/A')
                })
            
            return model_info
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"error": str(e)}
    
    def estimate_token_count(self, text: str, model_name: str = None) -> int:
        """Estimate token count for text (rough estimation)"""
        # Simple estimation: Korean characters are roughly 1.5 tokens each
        # English words are roughly 1.3 tokens each
        korean_chars = sum(1 for char in text if ord(char) >= 0xAC00 and ord(char) <= 0xD7A3)
        other_chars = len(text) - korean_chars
        
        estimated_tokens = int(korean_chars * 1.5 + other_chars * 0.8)
        return estimated_tokens
    
    def check_rate_limits(self) -> Dict[str, Any]:
        """Check current rate limit status (if available)"""
        # Note: Gemini API doesn't provide direct rate limit checking
        # This is a placeholder for future implementation
        return {
            "rate_limit_available": False,
            "message": "Rate limit checking not available for Gemini API"
        }