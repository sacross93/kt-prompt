"""
Enhanced API error handling and retry logic for Gemini API
"""
import time
import random
import json
import os
import re
from typing import Callable, Any, Optional, Dict, List
import logging
from functools import wraps
from datetime import datetime, timedelta
from models.exceptions import (
    APIError, RateLimitError, QuotaExceededError, 
    InvalidResponseError, GeminiOptimizerError
)
from services.advanced_parsing_handler import AdvancedParsingHandler

logger = logging.getLogger("gemini_optimizer.api_error_handler")

class APIErrorHandler:
    """Enhanced API error handler with advanced retry logic and recovery mechanisms"""
    
    def __init__(self, max_retries: int = 5, base_delay: float = 1.0, max_delay: float = 300.0, 
                 backoff_multiplier: float = 2.0, jitter_range: float = 0.1):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.jitter_range = jitter_range
        
        # Initialize advanced parsing handler
        self.parsing_handler = AdvancedParsingHandler()
        
        # Error tracking
        self.error_stats = {
            "total_errors": 0,
            "rate_limit_errors": 0,
            "network_errors": 0,
            "parsing_errors": 0,
            "successful_retries": 0,
            "failed_retries": 0,
            "last_error_time": None
        }
        
        # Error patterns for classification
        self.rate_limit_patterns = [
            "rate limit", "quota exceeded", "too many requests",
            "resource exhausted", "rate_limit_exceeded"
        ]
        
        self.network_error_patterns = [
            "network", "connection", "timeout", "unreachable",
            "dns", "socket", "ssl", "certificate"
        ]
        
        self.temporary_error_patterns = [
            "internal error", "server error", "service unavailable",
            "temporary", "retry", "503", "502", "500"
        ]
        
        logger.info(f"APIErrorHandler initialized with max_retries={max_retries}")
    
    def handle_rate_limit(self, retry_count: int, error_message: str = "") -> bool:
        """Handle rate limit errors"""
        if retry_count >= self.max_retries:
            logger.error(f"Rate limit exceeded after {self.max_retries} retries")
            return False
        
        # Extract wait time from error message if available
        wait_time = self._extract_wait_time(error_message)
        if wait_time is None:
            wait_time = self.exponential_backoff(retry_count)
        
        logger.warning(f"Rate limit hit (attempt {retry_count + 1}), waiting {wait_time:.1f} seconds...")
        time.sleep(wait_time)
        return True
    
    def handle_network_error(self, error: Exception, retry_count: int) -> bool:
        """Handle network-related errors"""
        if retry_count >= self.max_retries:
            logger.error(f"Network error after {self.max_retries} retries: {error}")
            return False
        
        wait_time = self.exponential_backoff(retry_count)
        logger.warning(f"Network error (attempt {retry_count + 1}), retrying in {wait_time:.1f} seconds: {error}")
        time.sleep(wait_time)
        return True
    
    def handle_response_format_error(self, response: str, retry_count: int, error_context: Optional[str] = None) -> Optional[List[str]]:
        """Enhanced response format error handling with multiple strategies"""
        self.error_stats["parsing_errors"] += 1
        logger.warning(f"Invalid response format (attempt {retry_count + 1}): {response[:100]}...")
        
        # Try enhanced response recovery first
        enhanced_response = self.parsing_handler.enhance_response_format_recovery(response, error_context)
        if enhanced_response != response:
            logger.info("Applied enhanced response format recovery")
            parsed_result = self.parsing_handler.try_multiple_parsing_strategies(enhanced_response)
            if parsed_result:
                logger.info(f"Successfully parsed enhanced response: {len(parsed_result)} items")
                return parsed_result
        
        # Try advanced parsing strategies on original response
        parsed_result = self.parsing_handler.try_multiple_parsing_strategies(response)
        if parsed_result:
            logger.info(f"Successfully parsed response using advanced strategies: {len(parsed_result)} items")
            return parsed_result
        
        # Try to extract partial results with adaptive confidence threshold
        partial_result = self.parsing_handler.extract_partial_results(response)
        confidence_threshold = max(0.3, 0.7 - (retry_count * 0.1))  # Lower threshold with more retries
        
        if partial_result.confidence > confidence_threshold:
            logger.info(f"Extracted partial results with {partial_result.confidence:.2f} confidence (threshold: {confidence_threshold:.2f})")
            return partial_result.parsed_items
        
        # Try intelligent response reconstruction
        reconstructed = self._attempt_intelligent_reconstruction(response, partial_result)
        if reconstructed:
            logger.info(f"Successfully reconstructed {len(reconstructed)} items from response")
            return reconstructed
        
        if retry_count >= self.max_retries:
            logger.error("Could not fix response format after maximum retries")
            # Generate comprehensive feedback for next attempt
            feedback = self._generate_comprehensive_feedback(response, partial_result, retry_count)
            logger.info(f"Generated comprehensive parsing feedback: {feedback}")
            return None
        
        return None
    
    def exponential_backoff(self, attempt: int, jitter: bool = True, adaptive: bool = True) -> float:
        """Enhanced exponential backoff with adaptive delays and jitter"""
        # Base exponential backoff
        delay = min(self.base_delay * (self.backoff_multiplier ** attempt), self.max_delay)
        
        # Adaptive delay based on error history
        if adaptive and self.error_stats["last_error_time"]:
            time_since_last_error = time.time() - self.error_stats["last_error_time"]
            if time_since_last_error < 60:  # Recent errors, increase delay
                delay *= 1.5
        
        # Add jitter to prevent thundering herd
        if jitter:
            jitter_amount = delay * self.jitter_range
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        # Ensure minimum delay
        return max(delay, 0.1)
    
    def classify_error(self, error: Exception) -> str:
        """Classify error type for appropriate handling"""
        error_message = str(error).lower()
        
        if any(pattern in error_message for pattern in self.rate_limit_patterns):
            return "rate_limit"
        elif any(pattern in error_message for pattern in self.network_error_patterns):
            return "network"
        elif any(pattern in error_message for pattern in self.temporary_error_patterns):
            return "temporary"
        elif "invalid" in error_message or "format" in error_message:
            return "format"
        else:
            return "unknown"
    
    def should_retry(self, error: Exception, retry_count: int) -> bool:
        """Determine if error should be retried"""
        if retry_count >= self.max_retries:
            return False
        
        error_type = self.classify_error(error)
        
        # Always retry rate limits and network errors
        if error_type in ["rate_limit", "network", "temporary"]:
            return True
        
        # Don't retry format errors or unknown errors
        if error_type in ["format", "unknown"]:
            return False
        
        return True
    
    def _extract_wait_time(self, error_message: str) -> Optional[float]:
        """Extract wait time from error message if available"""
        import re
        
        # Look for patterns like "retry after 30 seconds" or "wait 60s"
        patterns = [
            r'retry after (\d+)',
            r'wait (\d+)',
            r'try again in (\d+)',
            r'(\d+) seconds?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, error_message.lower())
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        return None
    
    def _attempt_response_fix(self, response: str) -> Optional[str]:
        """Attempt to fix malformed API response"""
        import re
        
        try:
            # Remove extra whitespace and normalize
            response = re.sub(r'\s+', ' ', response.strip())
            
            # Look for numbered classification patterns
            lines = response.split('\n')
            fixed_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Look for pattern: number. classification
                match = re.match(r'^(\d+)\.?\s*(.+)$', line)
                if match:
                    number = match.group(1)
                    classification = match.group(2).strip()
                    
                    # Try to fix classification format
                    fixed_classification = self._fix_classification_in_response(classification)
                    if fixed_classification:
                        fixed_lines.append(f"{number}. {fixed_classification}")
            
            if fixed_lines:
                return '\n'.join(fixed_lines)
            
        except Exception as e:
            logger.debug(f"Failed to fix response: {e}")
        
        return None
    
    def _fix_classification_in_response(self, classification: str) -> Optional[str]:
        """Fix classification format in response"""
        # Valid values for each category
        valid_types = ["사실형", "추론형", "대화형", "예측형"]
        valid_polarities = ["긍정", "부정", "미정"]
        valid_tenses = ["과거", "현재", "미래"]
        valid_certainties = ["확실", "불확실"]
        
        try:
            # Split by various separators
            parts = re.split(r'[,，\s]+', classification.strip())
            
            if len(parts) < 4:
                return None
            
            # Take first 4 parts and try to match
            parts = parts[:4]
            fixed_parts = []
            
            valid_categories = [valid_types, valid_polarities, valid_tenses, valid_certainties]
            
            for i, (part, valid_values) in enumerate(zip(parts, valid_categories)):
                matched_value = self._find_best_match(part, valid_values)
                if matched_value:
                    fixed_parts.append(matched_value)
                else:
                    return None
            
            return ','.join(fixed_parts)
            
        except Exception:
            return None
    
    def _find_best_match(self, value: str, valid_values: list) -> Optional[str]:
        """Find best match from valid values"""
        value = value.strip()
        
        # Exact match
        if value in valid_values:
            return value
        
        # Partial match
        for valid_val in valid_values:
            if value in valid_val or valid_val in value:
                return valid_val
        
        # Fuzzy match (simple character similarity)
        best_match = None
        best_score = 0
        
        for valid_val in valid_values:
            score = self._calculate_similarity(value, valid_val)
            if score > best_score and score > 0.6:  # Threshold for similarity
                best_score = score
                best_match = valid_val
        
        return best_match
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate simple character similarity between strings"""
        if not str1 or not str2:
            return 0.0
        
        # Simple character overlap ratio
        set1 = set(str1)
        set2 = set(str2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def handle_circuit_breaker(self, error_threshold: int = 10, time_window: int = 300) -> bool:
        """Implement circuit breaker pattern to prevent cascading failures"""
        current_time = time.time()
        
        # Check if we're in a failure window
        if (self.error_stats["last_error_time"] and 
            current_time - self.error_stats["last_error_time"] < time_window and
            self.error_stats["total_errors"] >= error_threshold):
            
            logger.warning(f"Circuit breaker activated: {self.error_stats['total_errors']} errors in {time_window}s")
            return False
        
        # Reset error count if outside time window
        if (self.error_stats["last_error_time"] and 
            current_time - self.error_stats["last_error_time"] >= time_window):
            self.error_stats["total_errors"] = 0
            logger.info("Circuit breaker reset: error window expired")
        
        return True
    
    def record_error(self, error_type: str):
        """Record error for tracking and circuit breaker"""
        self.error_stats["total_errors"] += 1
        self.error_stats["last_error_time"] = time.time()
        
        if error_type in self.error_stats:
            self.error_stats[error_type] += 1
    
    def record_success(self):
        """Record successful operation"""
        # Reset consecutive error count on success
        if self.error_stats["total_errors"] > 0:
            self.error_stats["successful_retries"] += 1
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error handling statistics"""
        stats = self.error_stats.copy()
        
        # Add calculated metrics
        total_attempts = stats["successful_retries"] + stats["failed_retries"]
        if total_attempts > 0:
            stats["success_rate"] = stats["successful_retries"] / total_attempts
        else:
            stats["success_rate"] = 1.0
        
        # Add time since last error
        if stats["last_error_time"]:
            stats["time_since_last_error"] = time.time() - stats["last_error_time"]
        
        return stats
    
    def should_use_circuit_breaker(self) -> bool:
        """Check if circuit breaker should be used"""
        return self.error_stats["total_errors"] >= 5  # Use circuit breaker after 5 errors
    
    def _attempt_intelligent_reconstruction(self, response: str, partial_result) -> Optional[List[str]]:
        """Attempt intelligent reconstruction of malformed response"""
        if not response or not partial_result:
            return None
        
        # If we have some successful parses, try to infer patterns
        if partial_result.parsed_items:
            # Use successful items as templates
            template_pattern = self._extract_pattern_from_successful_items(partial_result.parsed_items)
            
            # Try to apply pattern to failed items
            reconstructed = []
            for failed_item in partial_result.failed_items:
                reconstructed_item = self._apply_pattern_to_failed_item(failed_item, template_pattern)
                if reconstructed_item:
                    reconstructed.append(reconstructed_item)
            
            if reconstructed:
                return partial_result.parsed_items + reconstructed
        
        # Try statistical reconstruction based on common patterns
        return self._statistical_reconstruction(response)
    
    def _extract_pattern_from_successful_items(self, successful_items: List[str]) -> Dict[str, Any]:
        """Extract common patterns from successfully parsed items"""
        if not successful_items:
            return {}
        
        # Analyze structure of successful items
        patterns = {
            'separator': ',',
            'format': 'korean_terms',
            'length': 4,
            'common_terms': {}
        }
        
        # Count frequency of each classification term
        term_counts = {}
        for item in successful_items:
            parts = item.split(',')
            for i, part in enumerate(parts):
                category = ['type', 'polarity', 'tense', 'certainty'][i] if i < 4 else 'unknown'
                if category not in term_counts:
                    term_counts[category] = {}
                term_counts[category][part] = term_counts[category].get(part, 0) + 1
        
        patterns['common_terms'] = term_counts
        return patterns
    
    def _apply_pattern_to_failed_item(self, failed_item: str, pattern: Dict[str, Any]) -> Optional[str]:
        """Apply extracted pattern to reconstruct failed item"""
        if not failed_item or not pattern:
            return None
        
        # Try to extract any valid classification terms from failed item
        valid_terms = []
        
        # Check for Korean classification terms
        korean_terms = re.findall(r'[가-힣]+', failed_item)
        
        for term in korean_terms:
            # Check if term matches any known classification values
            for category, term_counts in pattern.get('common_terms', {}).items():
                if term in term_counts:
                    valid_terms.append(term)
                    break
        
        # If we found some valid terms, try to complete the classification
        if len(valid_terms) >= 2:  # Need at least 2 valid terms to attempt reconstruction
            # Fill in missing terms with most common ones from pattern
            while len(valid_terms) < 4:
                # Find most common term for missing category
                for category in ['type', 'polarity', 'tense', 'certainty']:
                    category_terms = pattern.get('common_terms', {}).get(category, {})
                    if category_terms:
                        most_common = max(category_terms.items(), key=lambda x: x[1])[0]
                        if most_common not in valid_terms:
                            valid_terms.append(most_common)
                            break
            
            if len(valid_terms) == 4:
                return ','.join(valid_terms[:4])
        
        return None
    
    def _statistical_reconstruction(self, response: str) -> Optional[List[str]]:
        """Statistical reconstruction based on response analysis"""
        # Count lines that look like they should be classifications
        lines = response.strip().split('\n')
        potential_classifications = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Count Korean terms that might be classifications
            korean_terms = re.findall(r'[가-힣]+', line)
            
            # If line has 2-6 Korean terms, it might be a classification
            if 2 <= len(korean_terms) <= 6:
                potential_classifications.append(line)
        
        # If we have potential classifications, try to standardize them
        if potential_classifications:
            standardized = []
            
            for line in potential_classifications:
                # Try to extract exactly 4 classification terms
                korean_terms = re.findall(r'[가-힣]+', line)
                
                if len(korean_terms) >= 4:
                    # Take first 4 terms and validate
                    candidate = ','.join(korean_terms[:4])
                    parts = candidate.split(',')
                    
                    # Basic validation - check if terms look like classifications
                    if self._looks_like_classification(parts):
                        standardized.append(candidate)
            
            return standardized if standardized else None
        
        return None
    
    def _looks_like_classification(self, parts: List[str]) -> bool:
        """Check if parts look like valid classification terms"""
        if len(parts) != 4:
            return False
        
        # Check if each part contains characters that appear in valid classifications
        valid_chars = set('사실추론대화예측긍정부정미과거현재미래확불')
        
        for part in parts:
            if not any(char in valid_chars for char in part):
                return False
            
            # Check minimum length
            if len(part) < 2:
                return False
        
        return True
    
    def _generate_comprehensive_feedback(self, response: str, partial_result, retry_count: int) -> str:
        """Generate comprehensive feedback for improving response format"""
        feedback_parts = [
            f"=== 응답 형식 개선 가이드 (시도 {retry_count + 1}) ===",
            "",
            "1. 기본 형식 요구사항:",
            "   - 각 줄: 번호. 문장유형,감정극성,시제,확실성",
            "   - 예시: 1. 사실형,긍정,현재,확실",
            "",
            "2. 유효한 분류 값들:",
            "   - 문장유형: 사실형, 추론형, 대화형, 예측형",
            "   - 감정극성: 긍정, 부정, 미정",
            "   - 시제: 과거, 현재, 미래",
            "   - 확실성: 확실, 불확실",
            ""
        ]
        
        # Add specific error analysis
        if partial_result:
            if partial_result.failed_items:
                feedback_parts.extend([
                    f"3. 파싱 실패 항목 분석 ({len(partial_result.failed_items)}개):",
                ])
                
                for i, failed_item in enumerate(partial_result.failed_items[:3]):
                    feedback_parts.append(f"   {i+1}. '{failed_item[:50]}...'")
                    
                    # Analyze what went wrong
                    issues = []
                    if ',' not in failed_item:
                        issues.append("쉼표 구분자 누락")
                    
                    korean_terms = re.findall(r'[가-힣]+', failed_item)
                    if len(korean_terms) < 4:
                        issues.append(f"분류 항목 부족 ({len(korean_terms)}/4)")
                    elif len(korean_terms) > 4:
                        issues.append(f"분류 항목 초과 ({len(korean_terms)}/4)")
                    
                    if issues:
                        feedback_parts.append(f"      문제점: {', '.join(issues)}")
                
                feedback_parts.append("")
            
            if partial_result.confidence < 0.5:
                feedback_parts.extend([
                    "4. 신뢰도 개선 방안:",
                    "   - 번호 매기기를 명확히 해주세요 (1., 2., 3., ...)",
                    "   - 각 분류 항목을 쉼표로 정확히 구분해주세요",
                    "   - 한국어 분류 용어만 사용해주세요",
                    "   - 각 줄에 정확히 4개의 분류 항목을 포함해주세요",
                    ""
                ])
        
        # Add retry-specific advice
        if retry_count > 2:
            feedback_parts.extend([
                "5. 추가 주의사항 (여러 번 시도 후):",
                "   - 응답 형식을 더욱 엄격히 준수해주세요",
                "   - 불필요한 설명이나 부가 정보는 제외해주세요",
                "   - 각 분류는 한 줄에 하나씩만 작성해주세요",
                ""
            ])
        
        feedback_parts.extend([
            "6. 올바른 응답 예시:",
            "1. 사실형,긍정,현재,확실",
            "2. 추론형,부정,과거,불확실",
            "3. 대화형,미정,미래,확실"
        ])
        
        return "\n".join(feedback_parts)
    
    def adaptive_retry_strategy(self, error_type: str, retry_count: int, error_history: List[str]) -> Dict[str, Any]:
        """Adaptive retry strategy based on error patterns"""
        strategy = {
            "wait_time": self.exponential_backoff(retry_count),
            "should_retry": True,
            "modifications": []
        }
        
        # Analyze error history for patterns
        recent_errors = error_history[-3:] if len(error_history) >= 3 else error_history
        
        if error_type == "rate_limit":
            # For rate limits, increase wait time more aggressively
            if len([e for e in recent_errors if "rate_limit" in e]) >= 2:
                strategy["wait_time"] *= 2  # Double wait time for repeated rate limits
                strategy["modifications"].append("increased_wait_time")
        
        elif error_type == "parsing":
            # For parsing errors, suggest format modifications
            if len([e for e in recent_errors if "parsing" in e]) >= 2:
                strategy["modifications"].extend([
                    "add_format_examples",
                    "simplify_prompt",
                    "explicit_format_request"
                ])
        
        elif error_type == "network":
            # For network errors, try alternative approaches
            if len([e for e in recent_errors if "network" in e]) >= 2:
                strategy["modifications"].extend([
                    "reduce_request_size",
                    "add_timeout_buffer",
                    "use_alternative_endpoint"
                ])
        
        # If too many retries, suggest giving up
        if retry_count >= self.max_retries - 1:
            strategy["should_retry"] = False
            strategy["modifications"].append("max_retries_reached")
        
        return strategy

def with_retry(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for automatic retry with error handling"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            error_handler = APIErrorHandler(max_retries, base_delay)
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    last_exception = e
                    
                    if not error_handler.should_retry(e, attempt):
                        break
                    
                    error_type = error_handler.classify_error(e)
                    
                    if error_type == "rate_limit":
                        if not error_handler.handle_rate_limit(attempt, str(e)):
                            break
                    elif error_type in ["network", "temporary"]:
                        if not error_handler.handle_network_error(e, attempt):
                            break
                    else:
                        # For other errors, just wait and retry
                        if attempt < max_retries:
                            wait_time = error_handler.exponential_backoff(attempt)
                            time.sleep(wait_time)
            
            # If we get here, all retries failed
            if isinstance(last_exception, GeminiOptimizerError):
                raise last_exception
            else:
                raise APIError(f"Operation failed after {max_retries} retries: {last_exception}")
        
        return wrapper
    return decorator