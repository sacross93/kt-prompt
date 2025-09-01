"""
Data models for Gemini Prompt Optimizer
"""
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from enum import Enum

class ClassificationType(Enum):
    """Classification types for Korean sentences"""
    FACTUAL = "사실형"      # 사실형
    INFERENTIAL = "추론형"  # 추론형
    CONVERSATIONAL = "대화형"  # 대화형
    PREDICTIVE = "예측형"   # 예측형

class Polarity(Enum):
    """Polarity classification"""
    POSITIVE = "긍정"  # 긍정
    NEGATIVE = "부정"  # 부정
    NEUTRAL = "미정"   # 미정

class Tense(Enum):
    """Tense classification"""
    PAST = "과거"      # 과거
    PRESENT = "현재"   # 현재
    FUTURE = "미래"    # 미래

class Certainty(Enum):
    """Certainty classification"""
    CERTAIN = "확실"     # 확실
    UNCERTAIN = "불확실"  # 불확실

@dataclass
class Sample:
    """Sample data structure for CSV data"""
    id: int
    sentence: str
    type: str      # 사실형/추론형/대화형/예측형
    polarity: str  # 긍정/부정/미정
    tense: str     # 과거/현재/미래
    certainty: str # 확실/불확실
    
    def get_expected_output(self) -> str:
        """Get expected output in the required format"""
        return f"{self.type},{self.polarity},{self.tense},{self.certainty}"
    
    def __str__(self) -> str:
        return f"Sample({self.id}: {self.sentence[:50]}...)"

@dataclass
class ErrorCase:
    """Error case for analysis"""
    question_id: int
    sentence: str
    expected: str
    predicted: str
    error_type: str  # type/polarity/tense/certainty or combination
    
    def get_error_details(self) -> Dict[str, str]:
        """Get detailed error information"""
        expected_parts = self.expected.split(',')
        predicted_parts = self.predicted.split(',')
        
        if len(expected_parts) != 4 or len(predicted_parts) != 4:
            return {"format_error": "Invalid format"}
        
        errors = {}
        categories = ["type", "polarity", "tense", "certainty"]
        
        for i, category in enumerate(categories):
            if expected_parts[i] != predicted_parts[i]:
                errors[category] = f"Expected: {expected_parts[i]}, Got: {predicted_parts[i]}"
        
        return errors
    
    def __str__(self) -> str:
        return f"ErrorCase({self.question_id}: {self.error_type})"

@dataclass
class AnalysisReport:
    """Analysis report from Gemini Pro"""
    total_errors: int
    error_patterns: Dict[str, int]
    improvement_suggestions: List[str]
    prompt_modifications: List[str]
    confidence_score: float
    analysis_text: str
    
    def get_summary(self) -> str:
        """Get summary of the analysis"""
        return f"""
Analysis Summary:
- Total Errors: {self.total_errors}
- Error Patterns: {self.error_patterns}
- Confidence Score: {self.confidence_score:.2f}
- Suggestions: {len(self.improvement_suggestions)} items
- Modifications: {len(self.prompt_modifications)} items
"""

@dataclass
class IterationState:
    """State of optimization iteration"""
    iteration: int
    current_accuracy: float
    target_accuracy: float
    best_accuracy: float
    best_prompt_version: int
    is_converged: bool
    total_samples: int
    correct_predictions: int
    error_count: int
    
    def get_progress_info(self) -> str:
        """Get progress information string"""
        return f"""
Iteration {self.iteration}:
- Current Accuracy: {self.current_accuracy:.4f}
- Target Accuracy: {self.target_accuracy:.4f}
- Best Accuracy: {self.best_accuracy:.4f}
- Correct: {self.correct_predictions}/{self.total_samples}
- Errors: {self.error_count}
- Converged: {self.is_converged}
"""
    
    def is_target_reached(self) -> bool:
        """Check if target accuracy is reached"""
        return self.current_accuracy >= self.target_accuracy
    
    def update_best(self) -> bool:
        """Update best accuracy if current is better"""
        if self.current_accuracy > self.best_accuracy:
            self.best_accuracy = self.current_accuracy
            self.best_prompt_version = self.iteration
            return True
        return False

@dataclass
class OptimizationResult:
    """Final optimization result"""
    final_accuracy: float
    best_accuracy: float
    best_prompt_version: int
    total_iterations: int
    convergence_achieved: bool
    final_prompt_path: str
    execution_time: float
    
    def get_final_report(self) -> str:
        """Get final optimization report"""
        return f"""
=== Optimization Complete ===
Final Accuracy: {self.final_accuracy:.4f}
Best Accuracy: {self.best_accuracy:.4f}
Best Prompt Version: {self.best_prompt_version}
Total Iterations: {self.total_iterations}
Convergence Achieved: {self.convergence_achieved}
Execution Time: {self.execution_time:.2f} seconds
Final Prompt: {self.final_prompt_path}
"""

@dataclass
class BaselineInsights:
    """기본 분석 결과"""
    success_factors: List[str]
    improvement_areas: List[str]
    best_features: List[str]
    performance_data: Dict[str, float]

@dataclass
class ErrorAnalysis:
    """고급 오류 분석 모델"""
    total_errors: int
    attribute_errors: Dict[str, int]       # 속성별 오류 수
    error_patterns: Dict[str, List[str]]   # 패턴별 오류 사례
    boundary_cases: List[ErrorCase]        # 경계 사례들
    parsing_failures: List[str]            # 파싱 실패 사례
    confidence_scores: Dict[str, float]    # 속성별 신뢰도

@dataclass
class DiagnosisReport:
    """진단 리포트 모델"""
    root_causes: List[str]                 # 근본 원인들
    attribute_issues: Dict[str, List[str]] # 속성별 문제점
    prompt_weaknesses: List[str]           # 프롬프트 약점들
    recommended_fixes: List[str]           # 권장 수정사항
    priority_areas: List[str]              # 우선순위 영역

@dataclass
class StrategyResult:
    """전략 실행 결과"""
    strategy_name: str
    accuracy: float
    execution_time: float
    error_count: int
    parsing_failures: int
    timestamp: str
    prompt_version: str
    detailed_metrics: Dict[str, float]
    
    def get_effectiveness_score(self) -> float:
        """전략의 종합 효과 점수 계산"""
        # 정확도 70%, 파싱 성공률 20%, 실행 시간 10% 가중치
        parsing_success_rate = 1.0 - (self.parsing_failures / max(1, self.error_count + 100))
        time_efficiency = min(1.0, 60.0 / max(1.0, self.execution_time))  # 60초 기준
        
        effectiveness = (
            self.accuracy * 0.7 +
            parsing_success_rate * 0.2 +
            time_efficiency * 0.1
        )
        return effectiveness

@dataclass
class OptimizationHistory:
    """최적화 히스토리 모델"""
    iterations: List[Dict[str, Any]]
    best_score: float
    best_strategy: Optional[str]
    strategy_effectiveness: Dict[str, float]
    convergence_status: str
    total_improvements: int
    
    def get_recent_performance(self, window: int = 3) -> List[float]:
        """최근 N회 성능 반환"""
        if len(self.iterations) < window:
            return [iter_data.get('accuracy', 0.0) for iter_data in self.iterations]
        return [iter_data.get('accuracy', 0.0) for iter_data in self.iterations[-window:]]
    
    def is_converged(self, threshold: float = 0.01, window: int = 3) -> bool:
        """수렴 여부 판단"""
        recent_scores = self.get_recent_performance(window)
        if len(recent_scores) < window:
            return False
        
        # Calculate variance manually
        if not recent_scores:
            return False
        mean_val = sum(recent_scores) / len(recent_scores)
        variance = sum((x - mean_val) ** 2 for x in recent_scores) / len(recent_scores)
        return variance < threshold