"""
StrategyManager와 기존 시스템 통합 예제
실제 최적화 파이프라인에서 StrategyManager를 사용하는 방법을 보여줍니다.
"""

import sys
import os
from datetime import datetime

sys.path.append('.')
sys.path.append('models')

from strategy_manager import StrategyManager, StrategyResult
from advanced_generator import AdvancedPromptGenerator, PromptStrategy

class OptimizationPipeline:
    """최적화 파이프라인 통합 예제"""
    
    def __init__(self):
        self.strategy_manager = StrategyManager()
        self.prompt_generator = AdvancedPromptGenerator()
        self.target_accuracy = 0.7
        self.max_iterations = 10
        
    def run_optimization_cycle(self):
        """전체 최적화 사이클 실행"""
        print("=== 프롬프트 최적화 파이프라인 시작 ===\n")
        
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            print(f"--- 반복 {iteration} ---")
            
            # 1. 다음 전략 선택
            next_strategy = self.strategy_manager.select_next_strategy(
                self.strategy_manager.optimization_history
            )
            print(f"선택된 전략: {next_strategy.value}")
            
            # 2. 전략에 따른 프롬프트 생성
            prompt = self.prompt_generator.create_enhanced_prompt(next_strategy)
            prompt_version = f"v{iteration}_{next_strategy.value}"
            
            # 3. 시뮬레이션된 테스트 실행 (실제로는 Gemini API 호출)
            test_result = self._simulate_test(next_strategy, iteration)
            
            # 4. 결과 기록
            strategy_result = StrategyResult(
                strategy=next_strategy,
                accuracy=test_result['accuracy'],
                execution_time=test_result['execution_time'],
                error_count=test_result['error_count'],
                parsing_failures=test_result['parsing_failures'],
                timestamp=datetime.now().isoformat(),
                prompt_version=prompt_version,
                detailed_metrics=test_result['detailed_metrics']
            )
            
            self.strategy_manager.record_strategy_result(strategy_result)
            
            print(f"결과: 정확도 {test_result['accuracy']:.3f}, "
                  f"효과성 {strategy_result.get_effectiveness_score():.3f}")
            
            # 5. 목표 달성 확인
            if test_result['accuracy'] >= self.target_accuracy:
                print(f"🎉 목표 정확도 {self.target_accuracy} 달성!")
                break
            
            # 6. 수렴 확인
            if self.strategy_manager.optimization_history.is_converged():
                print("⚠️ 성능 수렴 감지. 탐색적 전략으로 전환합니다.")
            
            print()
        
        # 최종 결과
        self._print_final_results()
        
    def _simulate_test(self, strategy: PromptStrategy, iteration: int):
        """테스트 결과 시뮬레이션 (실제로는 Gemini API 호출)"""
        # 전략별 기본 성능 (시뮬레이션)
        base_performance = {
            PromptStrategy.BASELINE: 0.65,
            PromptStrategy.EXPLICIT_RULES: 0.68,
            PromptStrategy.FEW_SHOT: 0.71,
            PromptStrategy.CHAIN_OF_THOUGHT: 0.69,
            PromptStrategy.HYBRID: 0.74
        }
        
        # 반복에 따른 개선 (학습 효과)
        improvement = min(0.05, iteration * 0.01)
        base_acc = base_performance.get(strategy, 0.65)
        accuracy = min(0.85, base_acc + improvement)
        
        # 노이즈 추가 (실제 테스트의 변동성)
        import random
        noise = random.uniform(-0.02, 0.02)
        accuracy = max(0.5, min(0.9, accuracy + noise))
        
        return {
            'accuracy': accuracy,
            'execution_time': random.uniform(20, 60),
            'error_count': int(100 * (1 - accuracy)),
            'parsing_failures': random.randint(0, 3),
            'detailed_metrics': {
                'type_accuracy': accuracy + random.uniform(-0.05, 0.05),
                'polarity_accuracy': accuracy + random.uniform(-0.05, 0.05),
                'tense_accuracy': accuracy + random.uniform(-0.05, 0.05),
                'certainty_accuracy': accuracy + random.uniform(-0.05, 0.05)
            }
        }
    
    def _print_final_results(self):
        """최종 결과 출력"""
        print("=== 최적화 완료 ===")
        
        history = self.strategy_manager.optimization_history
        print(f"최고 성능: {history.best_score:.3f}")
        print(f"최적 전략: {history.best_strategy.value if history.best_strategy else 'None'}")
        print(f"총 개선 횟수: {history.total_improvements}")
        print(f"총 반복 횟수: {len(history.iterations)}")
        
        # 전략별 성능 요약
        print("\n전략별 성능 요약:")
        comparison = self.strategy_manager.get_strategy_comparison()
        for strategy, effectiveness in comparison["strategy_rankings"]:
            usage_stats = comparison["usage_statistics"].get(strategy, {})
            print(f"  {strategy}: 효과성 {effectiveness:.3f}, "
                  f"평균 정확도 {usage_stats.get('avg_accuracy', 0):.3f}")
        
        # 권장사항
        print(f"\n권장사항:")
        if history.best_score >= self.target_accuracy:
            print("- 목표 성능 달성! GPT-4o 최종 검증을 진행하세요.")
        else:
            next_strategy = self.strategy_manager.select_next_strategy(history)
            print(f"- 다음 시도 권장 전략: {next_strategy.value}")
        
        # 데이터 저장
        self.strategy_manager.save_strategy_data()
        print("\n전략 데이터가 저장되었습니다.")

def demonstrate_strategy_selection_logic():
    """전략 선택 로직 시연"""
    print("=== 전략 선택 로직 시연 ===\n")
    
    manager = StrategyManager()
    
    # 시나리오별 전략 선택 테스트
    scenarios = [
        {
            "name": "초기 상태",
            "history": manager.optimization_history,
            "description": "아무 전략도 시도하지 않은 상태"
        },
        {
            "name": "성능 개선 중",
            "history": type(manager.optimization_history)(
                iterations=[
                    {"accuracy": 0.65, "strategy": "baseline"},
                    {"accuracy": 0.68, "strategy": "explicit_rules"},
                    {"accuracy": 0.71, "strategy": "few_shot"}
                ],
                best_score=0.71,
                best_strategy="few_shot",
                strategy_effectiveness={"baseline": 0.65, "explicit_rules": 0.68, "few_shot": 0.71},
                convergence_status="improving",
                total_improvements=2
            ),
            "description": "지속적으로 성능이 개선되고 있는 상태"
        },
        {
            "name": "성능 정체",
            "history": type(manager.optimization_history)(
                iterations=[
                    {"accuracy": 0.70, "strategy": "baseline"},
                    {"accuracy": 0.701, "strategy": "explicit_rules"},
                    {"accuracy": 0.699, "strategy": "few_shot"},
                    {"accuracy": 0.700, "strategy": "cot"}
                ],
                best_score=0.701,
                best_strategy="explicit_rules",
                strategy_effectiveness={"baseline": 0.70, "explicit_rules": 0.701, "few_shot": 0.699, "cot": 0.700},
                convergence_status="stagnant",
                total_improvements=1
            ),
            "description": "성능 개선이 정체된 상태"
        }
    ]
    
    for scenario in scenarios:
        print(f"시나리오: {scenario['name']}")
        print(f"설명: {scenario['description']}")
        
        selected_strategy = manager.select_next_strategy(scenario['history'])
        print(f"선택된 전략: {selected_strategy.value}")
        
        # 전략 선택 이유 설명
        if not scenario['history'].strategy_effectiveness:
            print("이유: 초기 상태이므로 기본 전략 선택")
        elif scenario['history'].is_converged():
            print("이유: 수렴 상태 감지, 탐색적 전략 선택")
        else:
            best_strategy = max(scenario['history'].strategy_effectiveness.items(), key=lambda x: x[1])
            print(f"이유: 현재 최고 성능 전략 ({best_strategy[0]}: {best_strategy[1]:.3f}) 기반 선택")
        
        print()

def demonstrate_combination_optimization():
    """전략 조합 최적화 시연"""
    print("=== 전략 조합 최적화 시연 ===\n")
    
    manager = StrategyManager()
    
    # 다양한 조합 시나리오
    combination_scenarios = [
        {
            "strategies": [PromptStrategy.FEW_SHOT],
            "description": "단일 전략"
        },
        {
            "strategies": [PromptStrategy.FEW_SHOT, PromptStrategy.EXPLICIT_RULES],
            "description": "이중 조합"
        },
        {
            "strategies": [PromptStrategy.FEW_SHOT, PromptStrategy.CHAIN_OF_THOUGHT, PromptStrategy.EXPLICIT_RULES],
            "description": "삼중 조합"
        }
    ]
    
    for scenario in combination_scenarios:
        print(f"조합 시나리오: {scenario['description']}")
        print(f"전략들: {[s.value for s in scenario['strategies']]}")
        
        optimal_combo = manager.find_optimal_combination(scenario['strategies'])
        print(f"최적 조합: {optimal_combo.combo_name}")
        print(f"예상 시너지: {optimal_combo.expected_synergy:.2f}")
        print(f"설명: {optimal_combo.description}")
        print()

def main():
    """메인 실행 함수"""
    print("StrategyManager 통합 시스템 데모\n")
    
    # 1. 전략 선택 로직 시연
    demonstrate_strategy_selection_logic()
    
    # 2. 전략 조합 최적화 시연
    demonstrate_combination_optimization()
    
    # 3. 전체 최적화 파이프라인 실행
    pipeline = OptimizationPipeline()
    pipeline.run_optimization_cycle()

if __name__ == "__main__":
    main()