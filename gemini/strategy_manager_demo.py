"""
StrategyManager 데모 및 통합 테스트
전략 관리 시스템의 모든 기능을 시연하는 예제
"""

import sys
import os
from datetime import datetime, timedelta

sys.path.append('.')
sys.path.append('models')

from strategy_manager import StrategyManager, StrategyResult, StrategyCombo
from advanced_generator import PromptStrategy

def create_sample_results():
    """샘플 전략 결과 생성"""
    results = []
    
    # 기본 전략 결과
    results.append(StrategyResult(
        strategy=PromptStrategy.BASELINE,
        accuracy=0.70,
        execution_time=25.0,
        error_count=30,
        parsing_failures=3,
        timestamp=(datetime.now() - timedelta(days=5)).isoformat(),
        prompt_version="v1",
        detailed_metrics={
            "type_accuracy": 0.75,
            "polarity_accuracy": 0.68,
            "tense_accuracy": 0.72,
            "certainty_accuracy": 0.65
        }
    ))
    
    # 명시적 규칙 전략 결과
    results.append(StrategyResult(
        strategy=PromptStrategy.EXPLICIT_RULES,
        accuracy=0.72,
        execution_time=30.0,
        error_count=28,
        parsing_failures=2,
        timestamp=(datetime.now() - timedelta(days=4)).isoformat(),
        prompt_version="v2",
        detailed_metrics={
            "type_accuracy": 0.78,
            "polarity_accuracy": 0.70,
            "tense_accuracy": 0.74,
            "certainty_accuracy": 0.67
        }
    ))
    
    # Few-shot 전략 결과
    results.append(StrategyResult(
        strategy=PromptStrategy.FEW_SHOT,
        accuracy=0.75,
        execution_time=35.0,
        error_count=25,
        parsing_failures=1,
        timestamp=(datetime.now() - timedelta(days=3)).isoformat(),
        prompt_version="v3",
        detailed_metrics={
            "type_accuracy": 0.80,
            "polarity_accuracy": 0.72,
            "tense_accuracy": 0.76,
            "certainty_accuracy": 0.72
        }
    ))
    
    # Chain-of-Thought 전략 결과
    results.append(StrategyResult(
        strategy=PromptStrategy.CHAIN_OF_THOUGHT,
        accuracy=0.73,
        execution_time=40.0,
        error_count=27,
        parsing_failures=2,
        timestamp=(datetime.now() - timedelta(days=2)).isoformat(),
        prompt_version="v4",
        detailed_metrics={
            "type_accuracy": 0.77,
            "polarity_accuracy": 0.71,
            "tense_accuracy": 0.75,
            "certainty_accuracy": 0.69
        }
    ))
    
    # 하이브리드 전략 결과
    results.append(StrategyResult(
        strategy=PromptStrategy.HYBRID,
        accuracy=0.78,
        execution_time=45.0,
        error_count=22,
        parsing_failures=0,
        timestamp=(datetime.now() - timedelta(days=1)).isoformat(),
        prompt_version="v5",
        detailed_metrics={
            "type_accuracy": 0.82,
            "polarity_accuracy": 0.75,
            "tense_accuracy": 0.78,
            "certainty_accuracy": 0.76
        }
    ))
    
    return results

def demonstrate_strategy_management():
    """전략 관리 시스템 시연"""
    print("=== 프롬프트 전략 관리 시스템 데모 ===\n")
    
    # 1. StrategyManager 초기화
    print("1. StrategyManager 초기화")
    manager = StrategyManager()
    print(f"   - 사용 가능한 전략: {len(manager.get_available_strategies())}개")
    print(f"   - 전략 조합: {len(manager.get_available_combinations())}개")
    print()
    
    # 2. 사용 가능한 전략 목록 출력
    print("2. 사용 가능한 전략 목록:")
    for strategy in manager.get_available_strategies():
        print(f"   - {strategy.value}")
    print()
    
    # 3. 전략 조합 목록 출력
    print("3. 전략 조합 목록:")
    for combo in manager.get_available_combinations():
        strategy_names = [s.value for s in combo.strategies]
        print(f"   - {combo.combo_name}: {strategy_names}")
        print(f"     설명: {combo.description}")
        print(f"     예상 시너지: {combo.expected_synergy:.2f}")
    print()
    
    # 4. 샘플 결과 기록
    print("4. 전략 실행 결과 기록:")
    sample_results = create_sample_results()
    
    for result in sample_results:
        manager.record_strategy_result(result)
        effectiveness = result.get_effectiveness_score()
        print(f"   - {result.strategy.value}: 정확도 {result.accuracy:.3f}, 효과성 {effectiveness:.3f}")
    print()
    
    # 5. 전략별 효과성 평가
    print("5. 전략별 효과성 평가:")
    for strategy in manager.get_available_strategies():
        effectiveness = manager.evaluate_strategy_effectiveness(strategy)
        print(f"   - {strategy.value}: {effectiveness:.3f}")
    print()
    
    # 6. 다음 전략 선택
    print("6. 다음 권장 전략 선택:")
    next_strategy = manager.select_next_strategy(manager.optimization_history)
    print(f"   - 권장 전략: {next_strategy.value}")
    print()
    
    # 7. 최적 전략 조합 찾기
    print("7. 최적 전략 조합 탐색:")
    top_strategies = [PromptStrategy.FEW_SHOT, PromptStrategy.EXPLICIT_RULES]
    optimal_combo = manager.find_optimal_combination(top_strategies)
    print(f"   - 조합명: {optimal_combo.combo_name}")
    print(f"   - 전략들: {[s.value for s in optimal_combo.strategies]}")
    print(f"   - 예상 시너지: {optimal_combo.expected_synergy:.2f}")
    print()
    
    # 8. 전략 비교 분석
    print("8. 전략 성능 비교 분석:")
    comparison = manager.get_strategy_comparison()
    
    print("   전략별 순위:")
    for i, (strategy, effectiveness) in enumerate(comparison["strategy_rankings"], 1):
        usage_stats = comparison["usage_statistics"].get(strategy, {})
        print(f"   {i}. {strategy}: {effectiveness:.3f} "
              f"(사용 {usage_stats.get('usage_count', 0)}회)")
    
    print("\n   사용 통계:")
    for strategy, stats in comparison["usage_statistics"].items():
        print(f"   - {strategy}: 평균 정확도 {stats['avg_accuracy']:.3f}, "
              f"평균 실행시간 {stats['avg_execution_time']:.1f}초")
    print()
    
    # 9. 수렴 상태 확인
    print("9. 최적화 상태 분석:")
    print(f"   - 최고 성능: {manager.optimization_history.best_score:.3f}")
    print(f"   - 최적 전략: {manager.optimization_history.best_strategy.value if manager.optimization_history.best_strategy else 'None'}")
    print(f"   - 총 개선 횟수: {manager.optimization_history.total_improvements}")
    print(f"   - 수렴 여부: {manager.optimization_history.is_converged()}")
    print()
    
    # 10. 종합 리포트 생성
    print("10. 종합 분석 리포트:")
    report = manager.generate_strategy_report()
    print(report)
    
    # 11. 데이터 저장
    print("11. 전략 데이터 저장:")
    manager.save_strategy_data()
    print("   - 전략 효과성 데이터 저장 완료")
    print("   - 상세 결과 데이터 저장 완료")
    print()
    
    return manager

def demonstrate_advanced_scenarios():
    """고급 시나리오 시연"""
    print("=== 고급 시나리오 시연 ===\n")
    
    manager = StrategyManager()
    
    # 시나리오 1: 성능 정체 상황
    print("시나리오 1: 성능 정체 상황에서의 전략 선택")
    stagnant_history = manager.optimization_history
    stagnant_history.iterations = [
        {"accuracy": 0.70, "strategy": "baseline"},
        {"accuracy": 0.701, "strategy": "explicit_rules"},
        {"accuracy": 0.699, "strategy": "few_shot"},
        {"accuracy": 0.700, "strategy": "cot"}
    ]
    stagnant_history.best_score = 0.701
    
    alternative_strategy = manager._select_alternative_strategy()
    print(f"   - 대안 전략: {alternative_strategy.value}")
    print()
    
    # 시나리오 2: 수렴 상황
    print("시나리오 2: 수렴 상황에서의 탐색적 전략 선택")
    converged_history = manager.optimization_history
    converged_history.iterations = [
        {"accuracy": 0.750, "strategy": "baseline"},
        {"accuracy": 0.751, "strategy": "explicit_rules"},
        {"accuracy": 0.752, "strategy": "few_shot"},
        {"accuracy": 0.751, "strategy": "cot"},
        {"accuracy": 0.752, "strategy": "hybrid"}
    ]
    
    exploration_strategy = manager._select_exploration_strategy()
    print(f"   - 탐색적 전략: {exploration_strategy.value}")
    print()
    
    # 시나리오 3: 사용자 정의 조합
    print("시나리오 3: 사용자 정의 전략 조합")
    custom_strategies = [PromptStrategy.CHAIN_OF_THOUGHT, PromptStrategy.FEW_SHOT, PromptStrategy.EXPLICIT_RULES]
    custom_combo = manager.find_optimal_combination(custom_strategies)
    print(f"   - 조합명: {custom_combo.combo_name}")
    print(f"   - 설명: {custom_combo.description}")
    print(f"   - 예상 시너지: {custom_combo.expected_synergy:.2f}")
    print()

def main():
    """메인 실행 함수"""
    try:
        # 기본 시연
        manager = demonstrate_strategy_management()
        
        # 고급 시나리오
        demonstrate_advanced_scenarios()
        
        print("=== 데모 완료 ===")
        print("전략 관리 시스템이 성공적으로 구현되었습니다!")
        
        # 최종 상태 요약
        print(f"\n최종 상태:")
        print(f"- 기록된 전략 결과: {sum(len(results) for results in manager.strategy_results.values())}개")
        print(f"- 최고 성능: {manager.optimization_history.best_score:.3f}")
        print(f"- 활성 전략: {len(manager.optimization_history.strategy_effectiveness)}개")
        
    except Exception as e:
        print(f"데모 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()