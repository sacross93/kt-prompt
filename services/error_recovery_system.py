"""
Comprehensive error recovery system integrating all error handling components
"""
import logging
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from services.api_error_handler import APIErrorHandler
from services.advanced_parsing_handler import AdvancedParsingHandler
from services.checkpoint_manager import CheckpointManager, RecoveryManager, CheckpointData
from models.exceptions import GeminiOptimizerError, APIError, InvalidResponseError

logger = logging.getLogger("gemini_optimizer.error_recovery")

@dataclass
class RecoveryContext:
    """Context information for error recovery"""
    operation_name: str
    attempt_count: int
    max_attempts: int
    last_error: Optional[Exception]
    checkpoint_data: Optional[CheckpointData]
    recovery_strategies: List[str]
    
class ErrorRecoverySystem:
    """Comprehensive error recovery system"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints", max_retries: int = 5):
        self.api_handler = APIErrorHandler(max_retries=max_retries)
        self.parsing_handler = AdvancedParsingHandler()
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
        self.recovery_manager = RecoveryManager(self.checkpoint_manager)
        
        # Recovery strategies
        self.recovery_strategies = {
            "api_retry": self._handle_api_retry,
            "parsing_recovery": self._handle_parsing_recovery,
            "checkpoint_restore": self._handle_checkpoint_restore,
            "graceful_degradation": self._handle_graceful_degradation,
            "circuit_breaker": self._handle_circuit_breaker
        }
        
        logger.info("ErrorRecoverySystem initialized")
    
    def execute_with_recovery(self, operation: Callable, operation_name: str, 
                            context: Dict[str, Any], max_attempts: int = 3) -> Any:
        """Execute operation with comprehensive error recovery"""
        recovery_context = RecoveryContext(
            operation_name=operation_name,
            attempt_count=0,
            max_attempts=max_attempts,
            last_error=None,
            checkpoint_data=None,
            recovery_strategies=[]
        )
        
        for attempt in range(max_attempts):
            recovery_context.attempt_count = attempt + 1
            
            try:
                logger.debug(f"Executing {operation_name} (attempt {attempt + 1}/{max_attempts})")
                
                # Check circuit breaker before attempting
                if not self.api_handler.handle_circuit_breaker():
                    raise APIError("Circuit breaker activated - too many recent failures")
                
                # Execute the operation
                result = operation(**context)
                
                # Record success
                self.api_handler.record_success()
                
                # Save checkpoint on successful operation
                if self._should_save_checkpoint(operation_name, attempt):
                    self._save_operation_checkpoint(operation_name, context, result)
                
                logger.debug(f"Operation {operation_name} completed successfully")
                return result
                
            except Exception as e:
                recovery_context.last_error = e
                logger.warning(f"Operation {operation_name} failed (attempt {attempt + 1}): {e}")
                
                # Record error
                error_type = self.api_handler.classify_error(e)
                self.api_handler.record_error(error_type)
                
                # Attempt recovery
                recovery_result = self._attempt_recovery(recovery_context, context)
                
                if recovery_result["should_retry"] and attempt < max_attempts - 1:
                    # Apply recovery actions
                    self._apply_recovery_actions(recovery_result["actions"], context)
                    
                    # Wait before retry
                    wait_time = self.api_handler.exponential_backoff(attempt)
                    logger.info(f"Waiting {wait_time:.1f}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Final attempt failed or no recovery possible
                    logger.error(f"Operation {operation_name} failed after {max_attempts} attempts")
                    
                    # Save error checkpoint
                    self._save_error_checkpoint(operation_name, context, e)
                    
                    # Re-raise the last exception
                    raise e
        
        # Should not reach here
        raise GeminiOptimizerError(f"Operation {operation_name} failed unexpectedly")
    
    def _attempt_recovery(self, context: RecoveryContext, operation_context: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to recover from error using available strategies"""
        recovery_actions = []
        should_retry = False
        
        error = context.last_error
        error_type = self.api_handler.classify_error(error)
        
        logger.debug(f"Attempting recovery for error type: {error_type}")
        
        # Get adaptive retry strategy
        error_history = operation_context.get('error_history', [])
        adaptive_strategy = self.api_handler.adaptive_retry_strategy(error_type, context.attempt_count - 1, error_history)
        
        # Strategy 1: API retry for network/rate limit errors
        if error_type in ["rate_limit", "network", "temporary"]:
            if adaptive_strategy["should_retry"] and self.api_handler.should_retry(error, context.attempt_count - 1):
                recovery_actions.append("api_retry")
                should_retry = True
                
                # Apply adaptive modifications
                for modification in adaptive_strategy["modifications"]:
                    recovery_actions.append(f"adaptive_{modification}")
        
        # Strategy 2: Enhanced parsing recovery for format errors
        elif error_type == "format" or isinstance(error, InvalidResponseError):
            if hasattr(error, 'response') and error.response:
                # Try enhanced parsing with error context
                error_context = f"parsing_error_attempt_{context.attempt_count}"
                
                try:
                    parsed_result = self.api_handler.handle_response_format_error(
                        error.response, 
                        context.attempt_count - 1, 
                        error_context
                    )
                    
                    if parsed_result:
                        recovery_actions.append("enhanced_parsing_recovery")
                        operation_context["recovered_result"] = parsed_result
                        logger.info(f"Successfully recovered {len(parsed_result)} items from parsing error")
                        return {"should_retry": False, "actions": recovery_actions, "result": parsed_result}
                except Exception as parse_error:
                    logger.warning(f"Parsing recovery failed: {parse_error}")
                
                # If parsing recovery failed but we haven't exceeded retries, try again
                if context.attempt_count < context.max_attempts:
                    recovery_actions.append("parsing_recovery_retry")
                    should_retry = True
                
                # If parsing failed, try incremental checkpoint recovery
                if context.attempt_count >= 2:
                    incremental_recovery = self._attempt_incremental_recovery(operation_context)
                    if incremental_recovery:
                        recovery_actions.append("incremental_recovery")
                        should_retry = True
        
        # Strategy 3: Progressive checkpoint restore
        if context.attempt_count >= 2 and not should_retry:
            recovery_info = self.recovery_manager.attempt_recovery()
            if recovery_info:
                recovery_actions.append("checkpoint_restore")
                context.checkpoint_data = recovery_info["checkpoint"]
                
                # Save progress checkpoint before restore
                self.checkpoint_manager.save_progress_checkpoint(
                    context.operation_name,
                    {
                        'recovery_attempt': context.attempt_count,
                        'error_type': error_type,
                        'recovery_actions': recovery_actions
                    }
                )
        
        # Strategy 4: Intelligent circuit breaker
        if self._should_activate_intelligent_circuit_breaker(error_type, context, operation_context):
            recovery_actions.append("intelligent_circuit_breaker")
            should_retry = False
        
        # Strategy 5: Adaptive graceful degradation
        if not should_retry and context.attempt_count >= context.max_attempts:
            degradation_level = self._calculate_degradation_level(context, operation_context)
            recovery_actions.append(f"graceful_degradation_level_{degradation_level}")
        
        return {
            "should_retry": should_retry,
            "actions": recovery_actions,
            "error_type": error_type,
            "adaptive_strategy": adaptive_strategy,
            "recommendations": self._get_enhanced_recovery_recommendations(error_type, context, adaptive_strategy)
        }
    
    def _apply_recovery_actions(self, actions: List[str], context: Dict[str, Any]):
        """Apply recovery actions to the operation context"""
        for action in actions:
            if action in self.recovery_strategies:
                try:
                    self.recovery_strategies[action](context)
                    logger.debug(f"Applied recovery action: {action}")
                except Exception as e:
                    logger.warning(f"Recovery action {action} failed: {e}")
    
    def _handle_api_retry(self, context: Dict[str, Any]):
        """Handle API retry recovery strategy"""
        # Adjust API parameters for retry
        if "api_params" in context:
            # Reduce batch size if applicable
            if "batch_size" in context["api_params"]:
                context["api_params"]["batch_size"] = max(1, context["api_params"]["batch_size"] // 2)
            
            # Add retry headers if applicable
            context["api_params"]["retry_attempt"] = context.get("retry_attempt", 0) + 1
    
    def _handle_parsing_recovery(self, context: Dict[str, Any]):
        """Handle parsing recovery strategy"""
        # Add parsing feedback to prompt if applicable
        if "prompt" in context and "parsing_errors" in context:
            feedback = self.parsing_handler.generate_parsing_feedback(context["parsing_errors"])
            context["parsing_feedback"] = feedback
    
    def _handle_checkpoint_restore(self, context: Dict[str, Any]):
        """Handle checkpoint restore recovery strategy"""
        recovery_info = self.recovery_manager.attempt_recovery()
        if recovery_info:
            checkpoint = recovery_info["checkpoint"]
            
            # Restore context from checkpoint
            context["restored_from_checkpoint"] = True
            context["checkpoint_iteration"] = checkpoint.iteration
            context["checkpoint_best_score"] = checkpoint.best_score
            
            logger.info(f"Restored from checkpoint: iteration {checkpoint.iteration}")
    
    def _handle_graceful_degradation(self, context: Dict[str, Any]):
        """Handle graceful degradation strategy"""
        # Reduce quality expectations or use fallback methods
        context["degraded_mode"] = True
        context["quality_threshold"] = context.get("quality_threshold", 0.7) * 0.9
        
        logger.info("Activated graceful degradation mode")
    
    def _handle_circuit_breaker(self, context: Dict[str, Any]):
        """Handle circuit breaker strategy"""
        # Temporarily disable certain features or reduce load
        context["circuit_breaker_active"] = True
        context["reduced_functionality"] = True
        
        logger.warning("Circuit breaker activated - reduced functionality mode")
    
    def _should_save_checkpoint(self, operation_name: str, attempt: int) -> bool:
        """Determine if checkpoint should be saved"""
        # Save checkpoint for critical operations or after multiple attempts
        critical_operations = ["optimize_prompt", "test_full_dataset", "analyze_results"]
        return operation_name in critical_operations or attempt > 0
    
    def _save_operation_checkpoint(self, operation_name: str, context: Dict[str, Any], result: Any):
        """Save checkpoint after successful operation"""
        try:
            checkpoint_data = self.checkpoint_manager.create_checkpoint(
                iteration=context.get("iteration", 0),
                current_prompt=context.get("current_prompt", ""),
                best_score=context.get("best_score", 0.0),
                best_prompt=context.get("best_prompt", ""),
                optimization_history=context.get("optimization_history", []),
                error_count=0,
                last_error=None,
                strategy_state=context.get("strategy_state", {}),
                progress_percentage=context.get("progress_percentage", 0.0)
            )
            
            self.checkpoint_manager.save_checkpoint(checkpoint_data)
            logger.debug(f"Saved checkpoint after successful {operation_name}")
            
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    def _save_error_checkpoint(self, operation_name: str, context: Dict[str, Any], error: Exception):
        """Save checkpoint after error for recovery"""
        try:
            checkpoint_data = self.checkpoint_manager.create_checkpoint(
                iteration=context.get("iteration", 0),
                current_prompt=context.get("current_prompt", ""),
                best_score=context.get("best_score", 0.0),
                best_prompt=context.get("best_prompt", ""),
                optimization_history=context.get("optimization_history", []),
                error_count=context.get("error_count", 0) + 1,
                last_error=str(error),
                strategy_state=context.get("strategy_state", {}),
                progress_percentage=context.get("progress_percentage", 0.0)
            )
            
            self.checkpoint_manager.save_checkpoint(checkpoint_data, force=True)
            logger.info(f"Saved error checkpoint after failed {operation_name}")
            
        except Exception as e:
            logger.error(f"Failed to save error checkpoint: {e}")
    
    def _get_recovery_recommendations(self, error_type: str, context: RecoveryContext) -> List[str]:
        """Get recovery recommendations based on error type and context"""
        recommendations = []
        
        if error_type == "rate_limit":
            recommendations.extend([
                "Reduce API call frequency",
                "Implement longer delays between requests",
                "Consider using batch processing"
            ])
        
        elif error_type == "network":
            recommendations.extend([
                "Check network connectivity",
                "Verify API endpoint availability",
                "Consider using alternative endpoints"
            ])
        
        elif error_type == "format":
            recommendations.extend([
                "Review response format requirements",
                "Improve prompt clarity",
                "Add format examples to prompt"
            ])
        
        elif error_type == "unknown":
            recommendations.extend([
                "Review error logs for patterns",
                "Consider contacting API support",
                "Implement additional error handling"
            ])
        
        # Add context-specific recommendations
        if context.attempt_count > 2:
            recommendations.append("Consider alternative approaches or manual intervention")
        
        if context.checkpoint_data:
            recommendations.append("Recovery from checkpoint is available")
        
        return recommendations
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        api_stats = self.api_handler.get_error_statistics()
        recovery_info = self.checkpoint_manager.get_recovery_info()
        
        return {
            "api_error_stats": api_stats,
            "recovery_info": recovery_info,
            "checkpoint_available": recovery_info is not None,
            "circuit_breaker_active": not self.api_handler.handle_circuit_breaker(),
            "system_health": self._calculate_system_health(api_stats)
        }
    
    def _calculate_system_health(self, stats: Dict[str, Any]) -> str:
        """Calculate overall system health status"""
        success_rate = stats.get("success_rate", 1.0)
        total_errors = stats.get("total_errors", 0)
        
        if success_rate >= 0.9 and total_errors < 5:
            return "healthy"
        elif success_rate >= 0.7 and total_errors < 15:
            return "warning"
        else:
            return "critical"
    
    def reset_error_state(self):
        """Reset error tracking state"""
        self.api_handler.error_stats = {
            "total_errors": 0,
            "rate_limit_errors": 0,
            "network_errors": 0,
            "parsing_errors": 0,
            "successful_retries": 0,
            "failed_retries": 0,
            "last_error_time": None
        }
        logger.info("Error state reset")
    
    def cleanup_resources(self):
        """Cleanup system resources"""
        try:
            # Cleanup old checkpoints
            self.checkpoint_manager.cleanup_old_checkpoints()
            
            # Reset error state if system is healthy
            status = self.get_system_status()
            if status["system_health"] == "healthy":
                self.reset_error_state()
            
            logger.info("Resource cleanup completed")
            
        except Exception as e:
            logger.warning(f"Resource cleanup failed: {e}")
    
    def _attempt_incremental_recovery(self, operation_context: Dict[str, Any]) -> bool:
        """Attempt incremental recovery using partial progress"""
        try:
            # Check if we have any partial progress to recover from
            operation_name = operation_context.get('operation_name', 'unknown')
            progress_checkpoint = self.checkpoint_manager.load_progress_checkpoint(operation_name)
            
            if progress_checkpoint:
                progress_data = progress_checkpoint.get('progress_data', {})
                
                # Apply partial progress to current context
                for key, value in progress_data.items():
                    if key not in operation_context:  # Don't overwrite existing context
                        operation_context[key] = value
                
                logger.info(f"Applied incremental recovery for {operation_name}")
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Incremental recovery failed: {e}")
            return False
    
    def _should_activate_intelligent_circuit_breaker(self, error_type: str, context: RecoveryContext, 
                                                   operation_context: Dict[str, Any]) -> bool:
        """Determine if intelligent circuit breaker should be activated"""
        # Basic circuit breaker check
        if self.api_handler.should_use_circuit_breaker():
            return True
        
        # Check for error patterns that suggest systemic issues
        error_history = operation_context.get('error_history', [])
        
        # If we have repeated errors of the same type
        recent_same_type_errors = [e for e in error_history[-5:] if error_type in e]
        if len(recent_same_type_errors) >= 3:
            logger.warning(f"Repeated {error_type} errors detected, activating circuit breaker")
            return True
        
        # If we're in a critical operation and have multiple failures
        if (operation_context.get('critical_operation', False) and 
            context.attempt_count >= 3):
            logger.warning("Critical operation with multiple failures, activating circuit breaker")
            return True
        
        return False
    
    def _calculate_degradation_level(self, context: RecoveryContext, operation_context: Dict[str, Any]) -> int:
        """Calculate appropriate degradation level (1-3, higher = more degraded)"""
        degradation_level = 1
        
        # Increase degradation based on attempt count
        if context.attempt_count >= 5:
            degradation_level = 3
        elif context.attempt_count >= 3:
            degradation_level = 2
        
        # Increase degradation based on error history
        error_history = operation_context.get('error_history', [])
        if len(error_history) >= 10:
            degradation_level = min(3, degradation_level + 1)
        
        # Increase degradation for critical errors
        if isinstance(context.last_error, (APIError, InvalidResponseError)):
            degradation_level = min(3, degradation_level + 1)
        
        return degradation_level
    
    def _get_enhanced_recovery_recommendations(self, error_type: str, context: RecoveryContext, 
                                             adaptive_strategy: Dict[str, Any]) -> List[str]:
        """Get enhanced recovery recommendations"""
        recommendations = self._get_recovery_recommendations(error_type, context)
        
        # Add adaptive strategy recommendations
        for modification in adaptive_strategy.get("modifications", []):
            if modification == "add_format_examples":
                recommendations.append("프롬프트에 더 많은 형식 예시를 추가하세요")
            elif modification == "simplify_prompt":
                recommendations.append("프롬프트를 더 간단하고 명확하게 수정하세요")
            elif modification == "reduce_request_size":
                recommendations.append("요청 크기를 줄여서 재시도하세요")
            elif modification == "increased_wait_time":
                recommendations.append("더 긴 대기 시간을 사용하세요")
        
        # Add context-specific recommendations
        if context.attempt_count >= 3:
            recommendations.append("대안적 접근 방법을 고려하세요")
        
        if context.checkpoint_data:
            recommendations.append(f"체크포인트에서 복구 가능 (반복 {context.checkpoint_data.iteration})")
        
        return recommendations
    
    def get_comprehensive_system_report(self) -> Dict[str, Any]:
        """Get comprehensive system status and performance report"""
        try:
            # Get basic system status
            basic_status = self.get_system_status()
            
            # Get checkpoint statistics
            checkpoint_stats = self.checkpoint_manager.get_checkpoint_statistics()
            
            # Get API handler statistics
            api_stats = self.api_handler.get_error_statistics()
            
            # Calculate system performance metrics
            performance_metrics = self._calculate_performance_metrics(api_stats, checkpoint_stats)
            
            # Generate recommendations
            system_recommendations = self._generate_system_recommendations(basic_status, checkpoint_stats, api_stats)
            
            return {
                'timestamp': time.time(),
                'basic_status': basic_status,
                'checkpoint_statistics': checkpoint_stats,
                'api_statistics': api_stats,
                'performance_metrics': performance_metrics,
                'system_recommendations': system_recommendations,
                'uptime_info': self._get_uptime_info()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate comprehensive system report: {e}")
            return {'error': str(e), 'timestamp': time.time()}
    
    def _calculate_performance_metrics(self, api_stats: Dict[str, Any], 
                                     checkpoint_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate system performance metrics"""
        metrics = {}
        
        # Error rate metrics
        total_operations = api_stats.get('successful_retries', 0) + api_stats.get('failed_retries', 0)
        if total_operations > 0:
            metrics['error_rate'] = api_stats.get('failed_retries', 0) / total_operations
            metrics['success_rate'] = api_stats.get('successful_retries', 0) / total_operations
        else:
            metrics['error_rate'] = 0.0
            metrics['success_rate'] = 1.0
        
        # Recovery effectiveness
        if api_stats.get('total_errors', 0) > 0:
            metrics['recovery_effectiveness'] = api_stats.get('successful_retries', 0) / api_stats.get('total_errors', 1)
        else:
            metrics['recovery_effectiveness'] = 1.0
        
        # Checkpoint efficiency
        metrics['checkpoint_frequency'] = checkpoint_stats.get('checkpoint_frequency', 0)
        metrics['average_score'] = checkpoint_stats.get('average_score', 0)
        metrics['score_trend'] = checkpoint_stats.get('score_trend', 'unknown')
        
        return metrics
    
    def _generate_system_recommendations(self, basic_status: Dict[str, Any], 
                                       checkpoint_stats: Dict[str, Any], 
                                       api_stats: Dict[str, Any]) -> List[str]:
        """Generate system-wide recommendations"""
        recommendations = []
        
        # Health-based recommendations
        health = basic_status.get('system_health', 'unknown')
        if health == 'critical':
            recommendations.append("시스템 상태가 위험합니다. 즉시 점검이 필요합니다.")
        elif health == 'warning':
            recommendations.append("시스템 성능이 저하되었습니다. 모니터링을 강화하세요.")
        
        # Error-based recommendations
        error_rate = api_stats.get('total_errors', 0)
        if error_rate > 10:
            recommendations.append("오류 발생률이 높습니다. API 설정을 검토하세요.")
        
        # Checkpoint-based recommendations
        if checkpoint_stats.get('total_checkpoints', 0) == 0:
            recommendations.append("체크포인트가 없습니다. 정기적인 저장을 설정하세요.")
        elif checkpoint_stats.get('score_trend') == 'declining':
            recommendations.append("성능이 하락 추세입니다. 전략을 재검토하세요.")
        
        # Storage recommendations
        storage_usage = checkpoint_stats.get('storage_usage', 0)
        if storage_usage > 100 * 1024 * 1024:  # 100MB
            recommendations.append("체크포인트 저장 공간이 많이 사용되었습니다. 정리를 고려하세요.")
        
        return recommendations
    
    def _get_uptime_info(self) -> Dict[str, Any]:
        """Get system uptime information"""
        # This is a simplified version - in a real system, you'd track actual uptime
        return {
            'session_start': time.time(),  # Would be set when system starts
            'current_time': time.time(),
            'estimated_uptime_seconds': 0  # Would calculate actual uptime
        }

# Decorator for automatic error recovery
def with_error_recovery(operation_name: str, max_attempts: int = 3, checkpoint_dir: str = "checkpoints"):
    """Decorator to add automatic error recovery to functions"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            recovery_system = ErrorRecoverySystem(checkpoint_dir=checkpoint_dir)
            
            # Prepare context from function arguments
            context = {
                "args": args,
                "kwargs": kwargs,
                "function_name": func.__name__
            }
            
            # Execute with recovery
            return recovery_system.execute_with_recovery(
                operation=lambda **ctx: func(*ctx["args"], **ctx["kwargs"]),
                operation_name=operation_name,
                context=context,
                max_attempts=max_attempts
            )
        
        return wrapper
    return decorator