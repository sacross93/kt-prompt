"""
프롬프트 전략 관리 시스템
다양한 최적화 전략을 관리하고 효과를 정량적으로 비교하는 시스템
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    # Fallback for numpy functions
    class np:
        @staticmethod
        def mean(values):
            return sum(values) / len(values) if values else 0.0
        
        @staticmethod
        def var(values):
            if not values:
                return 0.0
            mean_val = sum(values) / len(values)
            return sum((x - mean_val) ** 2 for x in values) / len(values)

from advanced_generator import PromptStrategy

@dataclass
class StrategyResult:
    """전략 실행 결과"""
    strategy: PromptStrategy
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
class StrategyCombo:
    """전략 조합"""
    strategies: List[PromptStrategy]
    combo_name: str
    description: str
    expected_synergy: float
    
    def get_combo_id(self) -> str:
        """조합 고유 ID 생성"""
        strategy_names = sorted([s.value for s in self.strategies])
        return "_".join(strategy_names)

@dataclass
class OptimizationHistory:
    """최적화 히스토리"""
    iterations: List[Dict[str, Any]]
    best_score: float
    best_strategy: Optional[PromptStrategy]
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
        
        variance = np.var(recent_scores)
        return variance < threshold

class StrategyManager:
    """프롬프트 전략 관리자"""
    
    def __init__(self, analysis_dir: str = "prompt/analysis"):
        self.analysis_dir = analysis_dir
        self.strategy_results: Dict[str, List[StrategyResult]] = defaultdict(list)
        self.strategy_combinations: List[StrategyCombo] = []
        self.optimization_history = OptimizationHistory(
            iterations=[],
            best_score=0.0,
            best_strategy=None,
            strategy_effectiveness={},
            convergence_status="not_started",
            total_improvements=0
        )
        
        # 기존 데이터 로드
        self._load_existing_data()
        self._initialize_strategy_combinations()
    
    def _load_existing_data(self):
        """기존 전략 효과 데이터 로드"""
        effectiveness_file = f"{self.analysis_dir}/strategy_effectiveness.json"
        if os.path.exists(effectiveness_file):
            try:
                with open(effectiveness_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                for strategy_name, strategy_data in data.get("strategies", {}).items():
                    if strategy_data.get("effectiveness") is not None:
                        self.optimization_history.strategy_effectiveness[strategy_name] = strategy_data["effectiveness"]
                        
            except Exception as e:
                print(f"기존 데이터 로드 실패: {e}")
    
    def _initialize_strategy_combinations(self):
        """전략 조합 초기화"""
        self.strategy_combinations = [
            StrategyCombo(
                strategies=[PromptStrategy.FEW_SHOT],
                combo_name="few_shot_only",
                description="Few-shot learning 단독 적용",
                expected_synergy=0.05
            ),
            StrategyCombo(
                strategies=[PromptStrategy.CHAIN_OF_THOUGHT],
                combo_name="cot_only", 
                description="Chain-of-Thought 단독 적용",
                expected_synergy=0.03
            ),
            StrategyCombo(
                strategies=[PromptStrategy.EXPLICIT_RULES],
                combo_name="rules_only",
                description="명시적 규칙 단독 적용",
                expected_synergy=0.04
            ),
            StrategyCombo(
                strategies=[PromptStrategy.FEW_SHOT, PromptStrategy.EXPLICIT_RULES],
                combo_name="few_shot_rules",
                description="Few-shot + 명시적 규칙 조합",
                expected_synergy=0.08
            ),
            StrategyCombo(
                strategies=[PromptStrategy.CHAIN_OF_THOUGHT, PromptStrategy.EXPLICIT_RULES],
                combo_name="cot_rules",
                description="CoT + 명시적 규칙 조합",
                expected_synergy=0.07
            ),
            StrategyCombo(
                strategies=[PromptStrategy.FEW_SHOT, PromptStrategy.CHAIN_OF_THOUGHT],
                combo_name="few_shot_cot",
                description="Few-shot + CoT 조합",
                expected_synergy=0.06
            ),
            StrategyCombo(
                strategies=[PromptStrategy.HYBRID],
                combo_name="hybrid_all",
                description="모든 기법 통합 적용",
                expected_synergy=0.10
            )
        ]
    
    def get_available_strategies(self) -> List[PromptStrategy]:
        """사용 가능한 전략 목록 반환"""
        return list(PromptStrategy)
    
    def get_available_combinations(self) -> List[StrategyCombo]:
        """사용 가능한 전략 조합 반환"""
        return self.strategy_combinations
    
    def record_strategy_result(self, result: StrategyResult):
        """전략 실행 결과 기록"""
        strategy_name = result.strategy.value
        self.strategy_results[strategy_name].append(result)
        
        # 효과성 업데이트
        effectiveness = result.get_effectiveness_score()
        self.optimization_history.strategy_effectiveness[strategy_name] = effectiveness
        
        # 최고 성능 업데이트
        if result.accuracy > self.optimization_history.best_score:
            self.optimization_history.best_score = result.accuracy
            self.optimization_history.best_strategy = result.strategy
            self.optimization_history.total_improvements += 1
        
        # 히스토리에 추가
        iteration_data = {
            "strategy": strategy_name,
            "accuracy": result.accuracy,
            "effectiveness": effectiveness,
            "timestamp": result.timestamp,
            "execution_time": result.execution_time
        }
        self.optimization_history.iterations.append(iteration_data)
        
        print(f"전략 '{strategy_name}' 결과 기록: 정확도 {result.accuracy:.4f}, 효과성 {effectiveness:.4f}")
    
    def evaluate_strategy_effectiveness(self, strategy: PromptStrategy) -> float:
        """전략의 효과성 평가"""
        strategy_name = strategy.value
        
        if strategy_name in self.strategy_results and self.strategy_results[strategy_name]:
            # 최근 결과들의 평균 효과성
            recent_results = self.strategy_results[strategy_name][-3:]  # 최근 3회
            effectiveness_scores = [result.get_effectiveness_score() for result in recent_results]
            return np.mean(effectiveness_scores)
        
        # 기존 데이터가 있으면 사용
        if strategy_name in self.optimization_history.strategy_effectiveness:
            return self.optimization_history.strategy_effectiveness[strategy_name]
        
        # 기본값 반환
        return 0.0
    
    def select_next_strategy(self, history: OptimizationHistory) -> PromptStrategy:
        """다음 시도할 전략 선택"""
        # 수렴 상태 확인
        if history.is_converged():
            self.optimization_history.convergence_status = "converged"
            return self._select_exploration_strategy()
        
        # 성능 개선이 없는 경우 다른 전략 시도
        recent_performance = history.get_recent_performance(3)
        if len(recent_performance) >= 3 and all(score <= history.best_score for score in recent_performance[-2:]):
            return self._select_alternative_strategy()
        
        # 기본적으로 가장 효과적인 전략 선택
        return self._select_best_strategy()
    
    def _select_best_strategy(self) -> PromptStrategy:
        """가장 효과적인 전략 선택"""
        if not self.optimization_history.strategy_effectiveness:
            return PromptStrategy.EXPLICIT_RULES  # 기본 전략
        
        best_strategy_name = max(
            self.optimization_history.strategy_effectiveness.keys(),
            key=lambda k: self.optimization_history.strategy_effectiveness[k]
        )
        
        try:
            return PromptStrategy(best_strategy_name)
        except ValueError:
            return PromptStrategy.EXPLICIT_RULES
    
    def _select_alternative_strategy(self) -> PromptStrategy:
        """대안 전략 선택 (성능 정체 시)"""
        # 아직 시도하지 않은 전략 우선
        tried_strategies = set(self.optimization_history.strategy_effectiveness.keys())
        available_strategies = [s for s in PromptStrategy if s.value not in tried_strategies]
        
        if available_strategies:
            return available_strategies[0]
        
        # 모든 전략을 시도했다면 효과성이 낮은 순서로 재시도
        sorted_strategies = sorted(
            self.optimization_history.strategy_effectiveness.items(),
            key=lambda x: x[1]
        )
        
        if sorted_strategies:
            try:
                return PromptStrategy(sorted_strategies[0][0])
            except ValueError:
                pass
        
        return PromptStrategy.HYBRID
    
    def _select_exploration_strategy(self) -> PromptStrategy:
        """탐색적 전략 선택 (수렴 시)"""
        # 조합 전략 시도
        unused_combos = [
            combo for combo in self.strategy_combinations
            if combo.get_combo_id() not in self.optimization_history.strategy_effectiveness
        ]
        
        if unused_combos:
            # 예상 시너지가 높은 조합 우선
            best_combo = max(unused_combos, key=lambda c: c.expected_synergy)
            if len(best_combo.strategies) == 1:
                return best_combo.strategies[0]
            else:
                return PromptStrategy.HYBRID  # 복합 전략은 HYBRID로 처리
        
        return PromptStrategy.HYBRID
    
    def find_optimal_combination(self, strategies: List[PromptStrategy]) -> StrategyCombo:
        """최적 전략 조합 탐색"""
        if not strategies:
            return self.strategy_combinations[0]  # 기본 조합
        
        # 단일 전략인 경우
        if len(strategies) == 1:
            strategy = strategies[0]
            for combo in self.strategy_combinations:
                if len(combo.strategies) == 1 and combo.strategies[0] == strategy:
                    return combo
        
        # 복합 전략 조합 찾기
        strategy_set = set(strategies)
        for combo in self.strategy_combinations:
            if set(combo.strategies) == strategy_set:
                return combo
        
        # 새로운 조합 생성
        combo_name = "_".join([s.value for s in sorted(strategies, key=lambda x: x.value)])
        return StrategyCombo(
            strategies=strategies,
            combo_name=combo_name,
            description=f"사용자 정의 조합: {combo_name}",
            expected_synergy=0.05 * len(strategies)
        )
    
    def get_strategy_comparison(self) -> Dict[str, Any]:
        """전략별 성능 비교 데이터 반환"""
        comparison = {
            "strategy_rankings": [],
            "performance_trends": {},
            "effectiveness_scores": self.optimization_history.strategy_effectiveness.copy(),
            "usage_statistics": {}
        }
        
        # 전략별 순위
        sorted_strategies = sorted(
            self.optimization_history.strategy_effectiveness.items(),
            key=lambda x: x[1],
            reverse=True
        )
        comparison["strategy_rankings"] = sorted_strategies
        
        # 성능 추이
        for strategy_name, results in self.strategy_results.items():
            if results:
                comparison["performance_trends"][strategy_name] = [
                    {"accuracy": r.accuracy, "timestamp": r.timestamp} for r in results
                ]
        
        # 사용 통계
        for strategy_name, results in self.strategy_results.items():
            comparison["usage_statistics"][strategy_name] = {
                "usage_count": len(results),
                "avg_accuracy": np.mean([r.accuracy for r in results]) if results else 0.0,
                "avg_execution_time": np.mean([r.execution_time for r in results]) if results else 0.0
            }
        
        return comparison
    
    def save_strategy_data(self):
        """전략 데이터 저장"""
        # 전략 효과성 저장
        effectiveness_data = {
            "strategies": {},
            "last_updated": datetime.now().isoformat(),
            "optimization_history": {
                "best_score": self.optimization_history.best_score,
                "best_strategy": self.optimization_history.best_strategy.value if self.optimization_history.best_strategy else None,
                "total_improvements": self.optimization_history.total_improvements,
                "convergence_status": self.optimization_history.convergence_status
            }
        }
        
        for strategy_name, effectiveness in self.optimization_history.strategy_effectiveness.items():
            usage_count = len(self.strategy_results.get(strategy_name, []))
            effectiveness_data["strategies"][strategy_name] = {
                "effectiveness": effectiveness,
                "used_count": usage_count,
                "description": self._get_strategy_description(strategy_name)
            }
        
        effectiveness_file = f"{self.analysis_dir}/strategy_effectiveness.json"
        with open(effectiveness_file, 'w', encoding='utf-8') as f:
            json.dump(effectiveness_data, f, ensure_ascii=False, indent=2)
        
        # 상세 결과 저장
        detailed_results = {}
        for strategy_name, results in self.strategy_results.items():
            serializable_results = []
            for result in results:
                result_dict = asdict(result)
                # PromptStrategy enum을 문자열로 변환
                if 'strategy' in result_dict and hasattr(result_dict['strategy'], 'value'):
                    result_dict['strategy'] = result_dict['strategy'].value
                serializable_results.append(result_dict)
            detailed_results[strategy_name] = serializable_results
        
        results_file = f"{self.analysis_dir}/strategy_detailed_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        
        print(f"전략 데이터가 저장되었습니다: {effectiveness_file}, {results_file}")
    
    def _get_strategy_description(self, strategy_name: str) -> str:
        """전략 설명 반환"""
        descriptions = {
            "baseline": "기본 프롬프트 구조",
            "few_shot": "Few-shot learning 예시 추가",
            "chain_of_thought": "Chain-of-Thought 추론 과정",
            "explicit_rules": "명시적 분류 규칙 강화",
            "hybrid": "모든 기법 통합 적용"
        }
        return descriptions.get(strategy_name, "사용자 정의 전략")
    
    def generate_strategy_report(self) -> str:
        """전략 분석 리포트 생성"""
        comparison = self.get_strategy_comparison()
        
        report = f"""
=== 프롬프트 전략 분석 리포트 ===
생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**최적화 현황:**
- 최고 성능: {self.optimization_history.best_score:.4f}
- 최적 전략: {self.optimization_history.best_strategy.value if self.optimization_history.best_strategy else 'None'}
- 총 개선 횟수: {self.optimization_history.total_improvements}
- 수렴 상태: {self.optimization_history.convergence_status}

**전략별 순위:**
"""
        
        for i, (strategy, effectiveness) in enumerate(comparison["strategy_rankings"], 1):
            usage_stats = comparison["usage_statistics"].get(strategy, {})
            report += f"{i}. {strategy}: {effectiveness:.4f} "
            report += f"(사용 {usage_stats.get('usage_count', 0)}회, "
            report += f"평균 정확도 {usage_stats.get('avg_accuracy', 0):.4f})\n"
        
        report += f"""
**성능 추이:**
- 총 반복 횟수: {len(self.optimization_history.iterations)}
- 최근 3회 평균: {np.mean(self.optimization_history.get_recent_performance(3)):.4f}

**권장사항:**
"""
        
        if self.optimization_history.is_converged():
            report += "- 현재 수렴 상태입니다. 새로운 전략 조합을 시도해보세요.\n"
        else:
            next_strategy = self.select_next_strategy(self.optimization_history)
            report += f"- 다음 권장 전략: {next_strategy.value}\n"
        
        if self.optimization_history.best_score < 0.7:
            report += "- 목표 성능(0.7) 미달성. 추가 최적화가 필요합니다.\n"
        elif self.optimization_history.best_score >= 0.8:
            report += "- 우수한 성능 달성! GPT-4o 최종 검증을 진행하세요.\n"
        
        return report

if __name__ == "__main__":
    # 테스트 실행
    manager = StrategyManager()
    
    print("=== 전략 관리자 테스트 ===")
    print(f"사용 가능한 전략: {[s.value for s in manager.get_available_strategies()]}")
    print(f"전략 조합: {len(manager.get_available_combinations())}개")
    
    # 샘플 결과 기록
    sample_result = StrategyResult(
        strategy=PromptStrategy.EXPLICIT_RULES,
        accuracy=0.72,
        execution_time=45.0,
        error_count=28,
        parsing_failures=2,
        timestamp=datetime.now().isoformat(),
        prompt_version="v1",
        detailed_metrics={"type_accuracy": 0.85, "polarity_accuracy": 0.70}
    )
    
    manager.record_strategy_result(sample_result)
    
    # 다음 전략 선택
    next_strategy = manager.select_next_strategy(manager.optimization_history)
    print(f"다음 권장 전략: {next_strategy.value}")
    
    # 리포트 생성
    report = manager.generate_strategy_report()
    print(report)
    
    # 데이터 저장
    manager.save_strategy_data()