"""
Advanced Iteration control and monitoring system for optimization process
자동화된 반복 최적화 컨트롤러 - 전체 최적화 사이클 관리
"""
import time
import json
import os
from typing import List, Dict, Any, Optional, Callable, Tuple
import logging
from datetime import datetime
from collections import defaultdict, deque

from models.data_models import (
    IterationState, OptimizationResult, OptimizationHistory, 
    StrategyResult, DiagnosisReport, ErrorAnalysis
)
from models.exceptions import ConvergenceError
from utils.logging_utils import (
    log_iteration_start, log_iteration_result, 
    log_convergence, log_optimization_complete
)
from config import OptimizationConfig

logger = logging.getLogger("gemini_optimizer.iteration_controller")

class ConvergenceDetector:
    """수렴 감지 및 분석 클래스"""
    
    def __init__(self, window_size: int = 5, variance_threshold: float = 0.001, 
                 improvement_threshold: float = 0.01):
        self.window_size = window_size
        self.variance_threshold = variance_threshold
        self.improvement_threshold = improvement_threshold
        self.performance_window = deque(maxlen=window_size)
        
    def add_performance(self, accuracy: float) -> None:
        """성능 데이터 추가"""
        self.performance_window.append(accuracy)
    
    def is_converged(self) -> bool:
        """수렴 여부 판단"""
        if len(self.performance_window) < self.window_size:
            return False
        
        # 분산 기반 수렴 감지
        scores = list(self.performance_window)
        mean_score = sum(scores) / len(scores)
        variance = sum((x - mean_score) ** 2 for x in scores) / len(scores)
        
        return variance < self.variance_threshold
    
    def get_trend(self) -> str:
        """성능 추이 분석"""
        if len(self.performance_window) < 3:
            return "insufficient_data"
        
        scores = list(self.performance_window)
        recent_trend = scores[-3:]
        
        # 최근 3회 추이 분석
        improvements = 0
        degradations = 0
        
        for i in range(1, len(recent_trend)):
            diff = recent_trend[i] - recent_trend[i-1]
            if diff > self.improvement_threshold:
                improvements += 1
            elif diff < -self.improvement_threshold:
                degradations += 1
        
        if improvements > degradations:
            return "improving"
        elif degradations > improvements:
            return "degrading"
        else:
            return "stable"
    
    def should_switch_strategy(self) -> bool:
        """전략 전환 필요 여부 판단"""
        if len(self.performance_window) < self.window_size:
            return False
        
        # 최근 성능이 정체되거나 하락하는 경우
        trend = self.get_trend()
        is_converged = self.is_converged()
        
        return is_converged and trend in ["stable", "degrading"]

class AdvancedIterationController:
    """
    고급 반복 최적화 컨트롤러
    - 전체 최적화 사이클 관리
    - 이전 반복의 학습 내용 누적 활용
    - 수렴 감지 및 대안 전략 자동 전환
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.current_state = None
        self.iteration_history = []
        self.start_time = None
        self.best_prompt_path = None
        
        # Convergence tracking
        self.no_improvement_count = 0
        self.improvement_threshold = config.convergence_threshold
        self.patience = config.patience
        
        # 메모리 시스템 - 이전 반복의 학습 내용 누적
        self.learning_memory = {
            "successful_patterns": [],      # 성공한 패턴들
            "failed_strategies": [],        # 실패한 전략들
            "error_patterns": defaultdict(list),  # 오류 패턴 누적
            "improvement_insights": [],     # 개선 인사이트
            "convergence_history": deque(maxlen=10),  # 수렴 히스토리
            "strategy_effectiveness": {},   # 전략별 효과성
            "boundary_cases": [],          # 경계 사례들
            "parsing_issues": []           # 파싱 문제들
        }
        
        # 전략 관리
        self.current_strategy = None
        self.strategy_queue = deque()
        self.strategy_attempts = defaultdict(int)
        self.strategy_results = defaultdict(list)
        
        # 수렴 감지
        self.convergence_detector = ConvergenceDetector(
            window_size=5,
            variance_threshold=0.001,
            improvement_threshold=config.convergence_threshold
        )
        
        # 최적화 히스토리
        self.optimization_history = OptimizationHistory(
            iterations=[],
            best_score=0.0,
            best_strategy=None,
            strategy_effectiveness={},
            convergence_status="not_started",
            total_improvements=0
        )
        
        logger.info(f"AdvancedIterationController initialized with target accuracy: {config.target_accuracy}")
        logger.info(f"Memory system and strategy management enabled")
    
    def initialize_optimization(self, total_samples: int, initial_strategy: str = None) -> IterationState:
        """고급 최적화 프로세스 초기화"""
        self.start_time = time.time()
        
        self.current_state = IterationState(
            iteration=0,
            current_accuracy=0.0,
            target_accuracy=self.config.target_accuracy,
            best_accuracy=0.0,
            best_prompt_version=0,
            is_converged=False,
            total_samples=total_samples,
            correct_predictions=0,
            error_count=total_samples
        )
        
        # 메모리 시스템 초기화
        self._load_previous_learning()
        
        # 전략 큐 초기화
        self._initialize_strategy_queue(initial_strategy)
        
        # 최적화 히스토리 초기화
        self.optimization_history.convergence_status = "running"
        
        logger.info(f"Advanced optimization initialized for {total_samples} samples")
        logger.info(f"Target accuracy: {self.config.target_accuracy:.4f}")
        logger.info(f"Max iterations: {self.config.max_iterations}")
        logger.info(f"Memory system loaded with {len(self.learning_memory['successful_patterns'])} patterns")
        
        return self.current_state
    
    def _load_previous_learning(self) -> None:
        """이전 학습 내용 로드"""
        memory_file = os.path.join(self.config.analysis_dir, "learning_memory.json")
        if os.path.exists(memory_file):
            try:
                with open(memory_file, 'r', encoding='utf-8') as f:
                    saved_memory = json.load(f)
                    
                # 기존 메모리와 병합
                for key, value in saved_memory.items():
                    if key in self.learning_memory:
                        if isinstance(self.learning_memory[key], list):
                            self.learning_memory[key].extend(value)
                        elif isinstance(self.learning_memory[key], dict):
                            self.learning_memory[key].update(value)
                
                logger.info(f"Previous learning loaded from {memory_file}")
            except Exception as e:
                logger.warning(f"Failed to load previous learning: {e}")
    
    def _save_learning_memory(self) -> None:
        """학습 메모리 저장"""
        memory_file = os.path.join(self.config.analysis_dir, "learning_memory.json")
        os.makedirs(self.config.analysis_dir, exist_ok=True)
        
        # deque를 list로 변환하여 JSON 직렬화 가능하게 만듦
        serializable_memory = {}
        for key, value in self.learning_memory.items():
            if isinstance(value, deque):
                serializable_memory[key] = list(value)
            elif isinstance(value, defaultdict):
                serializable_memory[key] = dict(value)
            else:
                serializable_memory[key] = value
        
        try:
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_memory, f, ensure_ascii=False, indent=2)
            logger.info(f"Learning memory saved to {memory_file}")
        except Exception as e:
            logger.error(f"Failed to save learning memory: {e}")
    
    def _initialize_strategy_queue(self, initial_strategy: str = None) -> None:
        """전략 큐 초기화"""
        # 기본 전략 순서 (효과성 기반)
        default_strategies = [
            "explicit_rules",    # 명시적 규칙 (가장 안정적)
            "few_shot",         # Few-shot learning
            "chain_of_thought", # CoT 추론
            "hybrid"            # 통합 접근
        ]
        
        # 이전 학습에서 효과적이었던 전략 우선
        if self.learning_memory["strategy_effectiveness"]:
            sorted_strategies = sorted(
                self.learning_memory["strategy_effectiveness"].items(),
                key=lambda x: x[1], reverse=True
            )
            effective_strategies = [s[0] for s in sorted_strategies if s[1] > 0.7]
            
            # 효과적인 전략을 앞에 배치
            strategy_order = effective_strategies + [s for s in default_strategies if s not in effective_strategies]
        else:
            strategy_order = default_strategies
        
        # 초기 전략이 지정된 경우 맨 앞에 배치
        if initial_strategy and initial_strategy not in strategy_order:
            strategy_order.insert(0, initial_strategy)
        elif initial_strategy:
            strategy_order.remove(initial_strategy)
            strategy_order.insert(0, initial_strategy)
        
        self.strategy_queue = deque(strategy_order)
        self.current_strategy = self.strategy_queue.popleft() if self.strategy_queue else "explicit_rules"
        
        logger.info(f"Strategy queue initialized: {list(self.strategy_queue)}")
        logger.info(f"Starting with strategy: {self.current_strategy}")
    
    def start_iteration(self, iteration: int) -> None:
        """Start a new iteration"""
        if self.current_state is None:
            raise ConvergenceError("Optimization not initialized")
        
        self.current_state.iteration = iteration
        log_iteration_start(logger, iteration, self.config.target_accuracy)
    
    def update_results(self, accuracy: float, correct_count: int, error_count: int, 
                      error_analysis: ErrorAnalysis = None, diagnosis: DiagnosisReport = None) -> None:
        """고급 결과 업데이트 - 학습 내용 누적"""
        if self.current_state is None:
            raise ConvergenceError("Optimization not initialized")
        
        self.current_state.current_accuracy = accuracy
        self.current_state.correct_predictions = correct_count
        self.current_state.error_count = error_count
        
        # 수렴 감지기에 성능 추가
        self.convergence_detector.add_performance(accuracy)
        
        # Check if this is the best result so far
        improved = self.current_state.update_best()
        
        if improved:
            self.no_improvement_count = 0
            self.optimization_history.total_improvements += 1
            logger.info(f"New best accuracy achieved: {accuracy:.4f}")
            
            # 성공 패턴 학습
            self._learn_successful_pattern(accuracy, self.current_strategy)
        else:
            # Check if improvement is significant
            improvement = accuracy - self.current_state.best_accuracy
            if improvement < self.improvement_threshold:
                self.no_improvement_count += 1
                logger.info(f"No significant improvement ({improvement:.4f} < {self.improvement_threshold:.4f})")
            else:
                self.no_improvement_count = 0
        
        # 전략 결과 기록
        self._record_strategy_result(accuracy, error_count)
        
        # 오류 분석 학습
        if error_analysis:
            self._learn_from_errors(error_analysis)
        
        # 진단 결과 학습
        if diagnosis:
            self._learn_from_diagnosis(diagnosis)
        
        # Log iteration results
        log_iteration_result(
            logger, 
            self.current_state.iteration, 
            accuracy, 
            error_count, 
            self.current_state.total_samples
        )
        
        # Store iteration history with enhanced data
        iteration_data = {
            "iteration": self.current_state.iteration,
            "accuracy": accuracy,
            "correct_count": correct_count,
            "error_count": error_count,
            "is_best": improved,
            "strategy": self.current_strategy,
            "timestamp": time.time(),
            "convergence_trend": self.convergence_detector.get_trend()
        }
        
        self.iteration_history.append(iteration_data)
        self.optimization_history.iterations.append(iteration_data)
        
        # 최적화 히스토리 업데이트
        if improved:
            self.optimization_history.best_score = accuracy
            self.optimization_history.best_strategy = self.current_strategy
    
    def _learn_successful_pattern(self, accuracy: float, strategy: str) -> None:
        """성공 패턴 학습"""
        pattern = {
            "strategy": strategy,
            "accuracy": accuracy,
            "iteration": self.current_state.iteration,
            "timestamp": datetime.now().isoformat()
        }
        
        self.learning_memory["successful_patterns"].append(pattern)
        
        # 전략 효과성 업데이트
        if strategy not in self.learning_memory["strategy_effectiveness"]:
            self.learning_memory["strategy_effectiveness"][strategy] = accuracy
        else:
            # 가중 평균으로 업데이트
            current_eff = self.learning_memory["strategy_effectiveness"][strategy]
            self.learning_memory["strategy_effectiveness"][strategy] = (current_eff + accuracy) / 2
        
        logger.info(f"Learned successful pattern: {strategy} -> {accuracy:.4f}")
    
    def _record_strategy_result(self, accuracy: float, error_count: int) -> None:
        """전략 결과 기록"""
        result = {
            "accuracy": accuracy,
            "error_count": error_count,
            "iteration": self.current_state.iteration,
            "timestamp": datetime.now().isoformat()
        }
        
        self.strategy_results[self.current_strategy].append(result)
        self.strategy_attempts[self.current_strategy] += 1
        
        # 전략 효과성 계산
        strategy_scores = [r["accuracy"] for r in self.strategy_results[self.current_strategy]]
        avg_effectiveness = sum(strategy_scores) / len(strategy_scores)
        self.optimization_history.strategy_effectiveness[self.current_strategy] = avg_effectiveness
    
    def _learn_from_errors(self, error_analysis: ErrorAnalysis) -> None:
        """오류 분석에서 학습"""
        # 오류 패턴 누적
        for pattern, cases in error_analysis.error_patterns.items():
            self.learning_memory["error_patterns"][pattern].extend(cases)
        
        # 경계 사례 누적
        self.learning_memory["boundary_cases"].extend(error_analysis.boundary_cases)
        
        # 파싱 실패 사례 누적
        self.learning_memory["parsing_issues"].extend(error_analysis.parsing_failures)
        
        logger.info(f"Learned from {error_analysis.total_errors} errors")
    
    def _learn_from_diagnosis(self, diagnosis: DiagnosisReport) -> None:
        """진단 결과에서 학습"""
        # 개선 인사이트 누적
        insights = {
            "root_causes": diagnosis.root_causes,
            "recommended_fixes": diagnosis.recommended_fixes,
            "priority_areas": diagnosis.priority_areas,
            "strategy": self.current_strategy,
            "iteration": self.current_state.iteration,
            "timestamp": datetime.now().isoformat()
        }
        
        self.learning_memory["improvement_insights"].append(insights)
        
        # 실패한 전략 기록 (성능이 낮은 경우)
        if self.current_state.current_accuracy < 0.6:
            failure_record = {
                "strategy": self.current_strategy,
                "accuracy": self.current_state.current_accuracy,
                "root_causes": diagnosis.root_causes,
                "timestamp": datetime.now().isoformat()
            }
            self.learning_memory["failed_strategies"].append(failure_record)
        
        logger.info(f"Learned from diagnosis: {len(diagnosis.root_causes)} root causes identified")
    
    def check_convergence(self) -> bool:
        """고급 수렴 검사 - 대안 전략 자동 전환 포함"""
        if self.current_state is None:
            return False
        
        # Check if target accuracy is reached
        if self.current_state.is_target_reached():
            self.current_state.is_converged = True
            self.optimization_history.convergence_status = "target_reached"
            log_convergence(logger, self.current_state.current_accuracy, self.current_state.iteration)
            return True
        
        # Check if maximum iterations reached
        if self.current_state.iteration >= self.config.max_iterations:
            logger.info(f"Maximum iterations ({self.config.max_iterations}) reached")
            self.current_state.is_converged = True
            self.optimization_history.convergence_status = "max_iterations"
            return True
        
        # 수렴 감지기를 통한 전략 전환 필요성 검사
        if self.convergence_detector.should_switch_strategy():
            if self._switch_to_alternative_strategy():
                logger.info(f"Switched to alternative strategy: {self.current_strategy}")
                self.no_improvement_count = 0  # 카운터 리셋
                return False  # 새 전략으로 계속 진행
            else:
                logger.info("No more alternative strategies available")
                self.current_state.is_converged = True
                self.optimization_history.convergence_status = "no_alternatives"
                return True
        
        # Check for early stopping due to no improvement
        if self.no_improvement_count >= self.patience:
            # 대안 전략 시도
            if self._switch_to_alternative_strategy():
                logger.info(f"Switched to alternative strategy due to no improvement: {self.current_strategy}")
                self.no_improvement_count = 0
                return False
            else:
                logger.info(f"Early stopping: No improvement for {self.patience} iterations and no alternatives")
                self.current_state.is_converged = True
                self.optimization_history.convergence_status = "early_stopping"
                return True
        
        return False
    
    def _switch_to_alternative_strategy(self) -> bool:
        """대안 전략으로 전환"""
        # 큐에서 다음 전략 가져오기
        if self.strategy_queue:
            new_strategy = self.strategy_queue.popleft()
            
            # 이미 시도한 전략인지 확인
            if self.strategy_attempts[new_strategy] >= 2:  # 최대 2회까지만 시도
                return self._switch_to_alternative_strategy()  # 재귀적으로 다음 전략 시도
            
            self.current_strategy = new_strategy
            logger.info(f"Strategy switched to: {new_strategy}")
            return True
        
        # 큐가 비어있으면 학습된 효과적인 전략 재시도
        effective_strategies = [
            strategy for strategy, effectiveness in self.learning_memory["strategy_effectiveness"].items()
            if effectiveness > 0.7 and self.strategy_attempts[strategy] < 3
        ]
        
        if effective_strategies:
            # 가장 효과적인 전략 선택
            best_strategy = max(effective_strategies, 
                              key=lambda s: self.learning_memory["strategy_effectiveness"][s])
            self.current_strategy = best_strategy
            logger.info(f"Retrying effective strategy: {best_strategy}")
            return True
        
        return False  # 더 이상 시도할 전략이 없음
    
    def get_current_strategy(self) -> str:
        """현재 사용 중인 전략 반환"""
        return self.current_strategy
    
    def get_next_strategy_recommendation(self) -> Optional[str]:
        """다음 권장 전략 반환"""
        if self.strategy_queue:
            return self.strategy_queue[0]
        
        # 학습된 효과적인 전략 중 덜 시도된 것 추천
        effective_strategies = [
            (strategy, effectiveness) for strategy, effectiveness 
            in self.learning_memory["strategy_effectiveness"].items()
            if effectiveness > 0.6 and self.strategy_attempts[strategy] < 2
        ]
        
        if effective_strategies:
            return max(effective_strategies, key=lambda x: x[1])[0]
        
        return None
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """누적된 학습 인사이트 반환"""
        insights = {
            "successful_patterns_count": len(self.learning_memory["successful_patterns"]),
            "failed_strategies_count": len(self.learning_memory["failed_strategies"]),
            "error_patterns_count": sum(len(cases) for cases in self.learning_memory["error_patterns"].values()),
            "improvement_insights_count": len(self.learning_memory["improvement_insights"]),
            "strategy_effectiveness": self.learning_memory["strategy_effectiveness"].copy(),
            "most_effective_strategy": None,
            "common_error_patterns": [],
            "key_improvement_areas": []
        }
        
        # 가장 효과적인 전략
        if self.learning_memory["strategy_effectiveness"]:
            insights["most_effective_strategy"] = max(
                self.learning_memory["strategy_effectiveness"].items(),
                key=lambda x: x[1]
            )
        
        # 공통 오류 패턴 (상위 3개)
        error_pattern_counts = {
            pattern: len(cases) for pattern, cases in self.learning_memory["error_patterns"].items()
        }
        if error_pattern_counts:
            sorted_patterns = sorted(error_pattern_counts.items(), key=lambda x: x[1], reverse=True)
            insights["common_error_patterns"] = sorted_patterns[:3]
        
        # 주요 개선 영역
        if self.learning_memory["improvement_insights"]:
            all_priority_areas = []
            for insight in self.learning_memory["improvement_insights"]:
                all_priority_areas.extend(insight.get("priority_areas", []))
            
            # 빈도 계산
            area_counts = {}
            for area in all_priority_areas:
                area_counts[area] = area_counts.get(area, 0) + 1
            
            sorted_areas = sorted(area_counts.items(), key=lambda x: x[1], reverse=True)
            insights["key_improvement_areas"] = sorted_areas[:5]
        
        return insights
    
    def run_optimization_cycle(self, test_function: Callable, analyze_function: Callable = None) -> bool:
        """자동화된 최적화 사이클 실행"""
        try:
            logger.info(f"Starting optimization cycle - Iteration {self.current_state.iteration}")
            logger.info(f"Current strategy: {self.current_strategy}")
            
            # 테스트 실행
            test_result = test_function(self.current_strategy)
            
            # 결과 업데이트
            error_analysis = None
            diagnosis = None
            
            if analyze_function and test_result.get("accuracy", 0) < self.config.target_accuracy:
                # 분석 실행
                analysis_result = analyze_function(test_result)
                error_analysis = analysis_result.get("error_analysis")
                diagnosis = analysis_result.get("diagnosis")
            
            self.update_results(
                accuracy=test_result.get("accuracy", 0),
                correct_count=test_result.get("correct_count", 0),
                error_count=test_result.get("error_count", 0),
                error_analysis=error_analysis,
                diagnosis=diagnosis
            )
            
            # 진행 상황 출력
            self.print_progress()
            
            # 학습 메모리 저장
            self._save_learning_memory()
            
            return True
            
        except Exception as e:
            logger.error(f"Error in optimization cycle: {e}")
            return False
    
    def should_continue(self) -> bool:
        """Check if optimization should continue"""
        return not self.check_convergence()
    
    def get_progress_info(self) -> str:
        """고급 진행 상황 정보"""
        if self.current_state is None:
            return "Optimization not started"
        
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        insights = self.get_learning_insights()
        
        progress_info = f"""
{'='*60}
Advanced Optimization Progress
{'='*60}
Iteration: {self.current_state.iteration}/{self.config.max_iterations}
Current Strategy: {self.current_strategy}
Current Accuracy: {self.current_state.current_accuracy:.4f}
Best Accuracy: {self.current_state.best_accuracy:.4f}
Target Accuracy: {self.current_state.target_accuracy:.4f}
Correct Predictions: {self.current_state.correct_predictions}/{self.current_state.total_samples}
Errors: {self.current_state.error_count}

Convergence Status:
- No Improvement Count: {self.no_improvement_count}/{self.patience}
- Performance Trend: {self.convergence_detector.get_trend()}
- Convergence Status: {self.optimization_history.convergence_status}

Strategy Management:
- Strategies Attempted: {len(self.strategy_attempts)}
- Current Strategy Attempts: {self.strategy_attempts[self.current_strategy]}
- Remaining Strategies: {len(self.strategy_queue)}
- Next Recommended: {self.get_next_strategy_recommendation() or 'None'}

Learning Insights:
- Successful Patterns: {insights['successful_patterns_count']}
- Error Patterns Learned: {insights['error_patterns_count']}
- Most Effective Strategy: {insights['most_effective_strategy'][0] if insights['most_effective_strategy'] else 'None'}
- Total Improvements: {self.optimization_history.total_improvements}

Timing:
- Elapsed Time: {elapsed_time:.1f} seconds
- Estimated Remaining: {self.estimate_remaining_time():.1f} seconds

Status: {'Converged' if self.current_state.is_converged else 'Running'}
{'='*60}
"""
        return progress_info
    
    def print_progress(self) -> None:
        """Print current progress to console"""
        print(self.get_progress_info())
    
    def get_iteration_summary(self) -> Dict[str, Any]:
        """Get summary of current iteration"""
        if self.current_state is None:
            return {}
        
        return {
            "iteration": self.current_state.iteration,
            "current_accuracy": self.current_state.current_accuracy,
            "best_accuracy": self.current_state.best_accuracy,
            "target_accuracy": self.current_state.target_accuracy,
            "correct_predictions": self.current_state.correct_predictions,
            "total_samples": self.current_state.total_samples,
            "error_count": self.current_state.error_count,
            "no_improvement_count": self.no_improvement_count,
            "is_converged": self.current_state.is_converged,
            "target_reached": self.current_state.is_target_reached()
        }
    
    def finalize_optimization(self, final_prompt_path: str = None) -> OptimizationResult:
        """고급 최적화 완료 및 결과 반환"""
        if self.current_state is None:
            raise ConvergenceError("Optimization not initialized")
        
        end_time = time.time()
        execution_time = end_time - self.start_time if self.start_time else 0
        
        # 최종 학습 메모리 저장
        self._save_learning_memory()
        
        # 최적화 히스토리 완료
        self.optimization_history.convergence_status = "completed"
        
        # 상세 결과 생성
        result = OptimizationResult(
            final_accuracy=self.current_state.current_accuracy,
            best_accuracy=self.current_state.best_accuracy,
            best_prompt_version=self.current_state.best_prompt_version,
            total_iterations=self.current_state.iteration,
            convergence_achieved=self.current_state.is_target_reached(),
            final_prompt_path=final_prompt_path or "unknown",
            execution_time=execution_time
        )
        
        # 최적화 요약 리포트 생성
        self._generate_optimization_report(result)
        
        log_optimization_complete(logger, result)
        
        return result
    
    def _generate_optimization_report(self, result: OptimizationResult) -> None:
        """최적화 요약 리포트 생성"""
        insights = self.get_learning_insights()
        
        report = f"""
=== Advanced Optimization Complete ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

FINAL RESULTS:
- Final Accuracy: {result.final_accuracy:.4f}
- Best Accuracy: {result.best_accuracy:.4f}
- Target Achieved: {result.convergence_achieved}
- Total Iterations: {result.total_iterations}
- Execution Time: {result.execution_time:.2f} seconds
- Best Strategy: {self.optimization_history.best_strategy}

LEARNING SUMMARY:
- Successful Patterns Discovered: {insights['successful_patterns_count']}
- Error Patterns Analyzed: {insights['error_patterns_count']}
- Strategies Tested: {len(self.strategy_attempts)}
- Total Improvements: {self.optimization_history.total_improvements}

STRATEGY EFFECTIVENESS:
"""
        
        for strategy, effectiveness in sorted(insights['strategy_effectiveness'].items(), 
                                           key=lambda x: x[1], reverse=True):
            attempts = self.strategy_attempts[strategy]
            report += f"- {strategy}: {effectiveness:.4f} ({attempts} attempts)\n"
        
        report += f"""
COMMON ERROR PATTERNS:
"""
        for pattern, count in insights['common_error_patterns']:
            report += f"- {pattern}: {count} occurrences\n"
        
        report += f"""
KEY IMPROVEMENT AREAS:
"""
        for area, frequency in insights['key_improvement_areas']:
            report += f"- {area}: mentioned {frequency} times\n"
        
        report += f"""
CONVERGENCE ANALYSIS:
- Final Status: {self.optimization_history.convergence_status}
- Performance Trend: {self.convergence_detector.get_trend()}
- Convergence Achieved: {self.convergence_detector.is_converged()}

RECOMMENDATIONS:
"""
        
        if result.convergence_achieved:
            report += "- Target accuracy achieved! Consider GPT-4o validation.\n"
        elif result.best_accuracy >= 0.8:
            report += "- High performance achieved. Fine-tuning may yield further improvements.\n"
        else:
            report += "- Consider additional strategies or data augmentation.\n"
            
        if insights['most_effective_strategy']:
            strategy, effectiveness = insights['most_effective_strategy']
            report += f"- Most effective strategy was '{strategy}' with {effectiveness:.4f} effectiveness.\n"
        
        # 리포트 저장
        report_file = os.path.join(self.config.analysis_dir, "advanced_optimization_report.md")
        os.makedirs(self.config.analysis_dir, exist_ok=True)
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Optimization report saved to {report_file}")
        except Exception as e:
            logger.error(f"Failed to save optimization report: {e}")
        
        print(report)
    
    def get_optimization_history(self) -> OptimizationHistory:
        """최적화 히스토리 반환"""
        return self.optimization_history
    
    def export_learning_memory(self) -> Dict[str, Any]:
        """학습 메모리 내보내기"""
        # deque와 defaultdict를 일반 자료구조로 변환
        exportable_memory = {}
        for key, value in self.learning_memory.items():
            if isinstance(value, deque):
                exportable_memory[key] = list(value)
            elif isinstance(value, defaultdict):
                exportable_memory[key] = dict(value)
            else:
                exportable_memory[key] = value
        
        return exportable_memory
    
    def import_learning_memory(self, memory_data: Dict[str, Any]) -> None:
        """학습 메모리 가져오기"""
        for key, value in memory_data.items():
            if key in self.learning_memory:
                if isinstance(self.learning_memory[key], list):
                    self.learning_memory[key].extend(value)
                elif isinstance(self.learning_memory[key], dict):
                    self.learning_memory[key].update(value)
        
        logger.info("Learning memory imported successfully")
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get detailed optimization statistics"""
        if not self.iteration_history:
            return {}
        
        accuracies = [h["accuracy"] for h in self.iteration_history]
        
        stats = {
            "total_iterations": len(self.iteration_history),
            "initial_accuracy": accuracies[0] if accuracies else 0.0,
            "final_accuracy": accuracies[-1] if accuracies else 0.0,
            "best_accuracy": max(accuracies) if accuracies else 0.0,
            "worst_accuracy": min(accuracies) if accuracies else 0.0,
            "average_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
            "improvement_iterations": sum(1 for h in self.iteration_history if h["is_best"]),
            "convergence_achieved": self.current_state.is_converged if self.current_state else False,
            "target_reached": self.current_state.is_target_reached() if self.current_state else False
        }
        
        # Calculate improvement trend
        if len(accuracies) > 1:
            stats["total_improvement"] = accuracies[-1] - accuracies[0]
            stats["improvement_rate"] = stats["total_improvement"] / len(accuracies)
        else:
            stats["total_improvement"] = 0.0
            stats["improvement_rate"] = 0.0
        
        return stats
    
    def export_iteration_history(self) -> List[Dict[str, Any]]:
        """Export iteration history for analysis"""
        return self.iteration_history.copy()
    
    def reset_optimization(self) -> None:
        """Reset optimization state for new run"""
        self.current_state = None
        self.iteration_history = []
        self.start_time = None
        self.best_prompt_path = None
        self.no_improvement_count = 0
        
        logger.info("Optimization state reset")
    
    def set_best_prompt_path(self, path: str) -> None:
        """Set path to best prompt"""
        self.best_prompt_path = path
        if self.current_state:
            self.current_state.best_prompt_version = self.current_state.iteration
    
    def get_convergence_info(self) -> Dict[str, Any]:
        """Get convergence information"""
        return {
            "is_converged": self.current_state.is_converged if self.current_state else False,
            "target_reached": self.current_state.is_target_reached() if self.current_state else False,
            "no_improvement_count": self.no_improvement_count,
            "patience": self.patience,
            "improvement_threshold": self.improvement_threshold,
            "iterations_remaining": max(0, self.config.max_iterations - (self.current_state.iteration if self.current_state else 0))
        }
    
    def estimate_remaining_time(self) -> float:
        """Estimate remaining optimization time"""
        if not self.iteration_history or not self.start_time:
            return 0.0
        
        elapsed_time = time.time() - self.start_time
        completed_iterations = len(self.iteration_history)
        
        if completed_iterations == 0:
            return 0.0
        
        avg_time_per_iteration = elapsed_time / completed_iterations
        remaining_iterations = max(0, self.config.max_iterations - completed_iterations)
        
        # Consider early stopping
        if self.no_improvement_count >= self.patience - 1:
            remaining_iterations = min(remaining_iterations, 1)
        
        return avg_time_per_iteration * remaining_iterations
    
    def get_performance_trend(self, window_size: int = 3) -> str:
        """Get performance trend analysis"""
        if len(self.iteration_history) < window_size:
            return "insufficient_data"
        
        recent_accuracies = [h["accuracy"] for h in self.iteration_history[-window_size:]]
        
        # Calculate trend
        if len(recent_accuracies) < 2:
            return "stable"
        
        improvements = 0
        degradations = 0
        
        for i in range(1, len(recent_accuracies)):
            diff = recent_accuracies[i] - recent_accuracies[i-1]
            if diff > self.improvement_threshold:
                improvements += 1
            elif diff < -self.improvement_threshold:
                degradations += 1
        
        if improvements > degradations:
            return "improving"
        elif degradations > improvements:
            return "degrading"
        else:
            return "stable"
# Backward compatibility alias
IterationController = AdvancedIterationController