"""
StrategyManager 테스트
"""

import unittest
import os
import json
import tempfile
import shutil
from datetime import datetime
from unittest.mock import patch, MagicMock

import sys
sys.path.append('gemini')
sys.path.append('models')

from strategy_manager import StrategyManager, StrategyResult, StrategyCombo, OptimizationHistory
from advanced_generator import PromptStrategy
from data_models import OptimizationHistory as DataOptimizationHistory

class TestStrategyManager(unittest.TestCase):
    """StrategyManager 테스트 클래스"""
    
    def setUp(self):
        """테스트 설정"""
        # 임시 디렉토리 생성
        self.test_dir = tempfile.mkdtemp()
        self.analysis_dir = os.path.join(self.test_dir, "analysis")
        os.makedirs(self.analysis_dir, exist_ok=True)
        
        # 테스트용 전략 효과성 데이터 생성
        test_data = {
            "strategies": {
                "baseline": {
                    "effectiveness": 0.7,
                    "used_count": 1,
                    "description": "기본 프롬프트 구조"
                },
                "explicit_rules": {
                    "effectiveness": 0.72,
                    "used_count": 2,
                    "description": "명시적 분류 규칙 강화"
                }
            }
        }
        
        with open(os.path.join(self.analysis_dir, "strategy_effectiveness.json"), 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        self.manager = StrategyManager(self.analysis_dir)
    
    def tearDown(self):
        """테스트 정리"""
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """초기화 테스트"""
        self.assertIsInstance(self.manager, StrategyManager)
        self.assertEqual(len(self.manager.get_available_strategies()), 5)  # 5개 전략
        self.assertGreater(len(self.manager.get_available_combinations()), 0)
        
        # 기존 데이터 로드 확인
        self.assertIn("baseline", self.manager.optimization_history.strategy_effectiveness)
        self.assertEqual(self.manager.optimization_history.strategy_effectiveness["baseline"], 0.7)
    
    def test_strategy_result_recording(self):
        """전략 결과 기록 테스트"""
        result = StrategyResult(
            strategy=PromptStrategy.FEW_SHOT,
            accuracy=0.75,
            execution_time=30.0,
            error_count=25,
            parsing_failures=1,
            timestamp=datetime.now().isoformat(),
            prompt_version="v1",
            detailed_metrics={"type_accuracy": 0.8, "polarity_accuracy": 0.7}
        )
        
        initial_iterations = len(self.manager.optimization_history.iterations)
        self.manager.record_strategy_result(result)
        
        # 결과 기록 확인
        self.assertEqual(len(self.manager.optimization_history.iterations), initial_iterations + 1)
        self.assertIn("few_shot", self.manager.optimization_history.strategy_effectiveness)
        self.assertEqual(self.manager.optimization_history.best_score, 0.75)
        self.assertEqual(self.manager.optimization_history.best_strategy, PromptStrategy.FEW_SHOT)
    
    def test_effectiveness_calculation(self):
        """효과성 계산 테스트"""
        result = StrategyResult(
            strategy=PromptStrategy.CHAIN_OF_THOUGHT,
            accuracy=0.8,
            execution_time=45.0,
            error_count=20,
            parsing_failures=0,
            timestamp=datetime.now().isoformat(),
            prompt_version="v1",
            detailed_metrics={}
        )
        
        effectiveness = result.get_effectiveness_score()
        
        # 효과성 점수가 정확도보다 약간 높아야 함 (파싱 성공률과 시간 효율성 포함)
        self.assertGreater(effectiveness, 0.8)
        self.assertLess(effectiveness, 1.0)
    
    def test_strategy_selection(self):
        """전략 선택 테스트"""
        # 기본 전략 선택
        strategy = self.manager.select_next_strategy(self.manager.optimization_history)
        self.assertIsInstance(strategy, PromptStrategy)
        
        # 수렴 상태에서의 전략 선택
        converged_history = OptimizationHistory(
            iterations=[
                {"accuracy": 0.75, "strategy": "baseline"},
                {"accuracy": 0.751, "strategy": "explicit_rules"},
                {"accuracy": 0.752, "strategy": "few_shot"}
            ],
            best_score=0.752,
            best_strategy=None,
            strategy_effectiveness={"baseline": 0.75, "explicit_rules": 0.751},
            convergence_status="converged",
            total_improvements=2
        )
        
        strategy = self.manager.select_next_strategy(converged_history)
        self.assertIsInstance(strategy, PromptStrategy)
    
    def test_combination_finding(self):
        """전략 조합 찾기 테스트"""
        # 단일 전략 조합
        combo = self.manager.find_optimal_combination([PromptStrategy.FEW_SHOT])
        self.assertEqual(len(combo.strategies), 1)
        self.assertEqual(combo.strategies[0], PromptStrategy.FEW_SHOT)
        
        # 복합 전략 조합
        combo = self.manager.find_optimal_combination([PromptStrategy.FEW_SHOT, PromptStrategy.EXPLICIT_RULES])
        self.assertIn(PromptStrategy.FEW_SHOT, combo.strategies)
        self.assertIn(PromptStrategy.EXPLICIT_RULES, combo.strategies)
    
    def test_strategy_comparison(self):
        """전략 비교 테스트"""
        # 테스트 결과 추가
        result1 = StrategyResult(
            strategy=PromptStrategy.BASELINE,
            accuracy=0.7,
            execution_time=20.0,
            error_count=30,
            parsing_failures=2,
            timestamp=datetime.now().isoformat(),
            prompt_version="v1",
            detailed_metrics={}
        )
        
        result2 = StrategyResult(
            strategy=PromptStrategy.FEW_SHOT,
            accuracy=0.75,
            execution_time=35.0,
            error_count=25,
            parsing_failures=1,
            timestamp=datetime.now().isoformat(),
            prompt_version="v2",
            detailed_metrics={}
        )
        
        self.manager.record_strategy_result(result1)
        self.manager.record_strategy_result(result2)
        
        comparison = self.manager.get_strategy_comparison()
        
        # 비교 데이터 구조 확인
        self.assertIn("strategy_rankings", comparison)
        self.assertIn("performance_trends", comparison)
        self.assertIn("effectiveness_scores", comparison)
        self.assertIn("usage_statistics", comparison)
        
        # 순위 확인 (few_shot이 더 높은 성능)
        rankings = comparison["strategy_rankings"]
        self.assertGreater(len(rankings), 0)
        
        # 사용 통계 확인
        usage_stats = comparison["usage_statistics"]
        self.assertIn("baseline", usage_stats)
        self.assertIn("few_shot", usage_stats)
    
    def test_data_persistence(self):
        """데이터 저장/로드 테스트"""
        # 결과 추가
        result = StrategyResult(
            strategy=PromptStrategy.HYBRID,
            accuracy=0.78,
            execution_time=50.0,
            error_count=22,
            parsing_failures=0,
            timestamp=datetime.now().isoformat(),
            prompt_version="v3",
            detailed_metrics={"overall": 0.78}
        )
        
        self.manager.record_strategy_result(result)
        self.manager.save_strategy_data()
        
        # 파일 생성 확인
        effectiveness_file = os.path.join(self.analysis_dir, "strategy_effectiveness.json")
        results_file = os.path.join(self.analysis_dir, "strategy_detailed_results.json")
        
        self.assertTrue(os.path.exists(effectiveness_file))
        self.assertTrue(os.path.exists(results_file))
        
        # 데이터 내용 확인
        with open(effectiveness_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.assertIn("strategies", data)
            self.assertIn("hybrid", data["strategies"])
            self.assertIn("optimization_history", data)
    
    def test_report_generation(self):
        """리포트 생성 테스트"""
        # 테스트 데이터 추가
        result = StrategyResult(
            strategy=PromptStrategy.EXPLICIT_RULES,
            accuracy=0.73,
            execution_time=40.0,
            error_count=27,
            parsing_failures=1,
            timestamp=datetime.now().isoformat(),
            prompt_version="v1",
            detailed_metrics={}
        )
        
        self.manager.record_strategy_result(result)
        
        report = self.manager.generate_strategy_report()
        
        # 리포트 내용 확인
        self.assertIn("프롬프트 전략 분석 리포트", report)
        self.assertIn("최적화 현황", report)
        self.assertIn("전략별 순위", report)
        self.assertIn("성능 추이", report)
        self.assertIn("권장사항", report)
        
        # 성능 정보 포함 확인
        self.assertIn("0.73", report)  # 정확도
        self.assertIn("explicit_rules", report)  # 전략명
    
    def test_convergence_detection(self):
        """수렴 감지 테스트"""
        history = OptimizationHistory(
            iterations=[
                {"accuracy": 0.750, "strategy": "baseline"},
                {"accuracy": 0.751, "strategy": "explicit_rules"},
                {"accuracy": 0.752, "strategy": "few_shot"},
                {"accuracy": 0.751, "strategy": "cot"},
                {"accuracy": 0.752, "strategy": "hybrid"}
            ],
            best_score=0.752,
            best_strategy=None,
            strategy_effectiveness={},
            convergence_status="running",
            total_improvements=0
        )
        
        # 수렴 상태 확인 (분산이 작음)
        self.assertTrue(history.is_converged(threshold=0.01, window=3))
        
        # 비수렴 상태 테스트
        history.iterations.append({"accuracy": 0.780, "strategy": "new_strategy"})
        self.assertFalse(history.is_converged(threshold=0.01, window=3))
    
    def test_strategy_combinations(self):
        """전략 조합 테스트"""
        combinations = self.manager.get_available_combinations()
        
        # 조합 개수 확인
        self.assertGreater(len(combinations), 5)
        
        # 각 조합의 구조 확인
        for combo in combinations:
            self.assertIsInstance(combo, StrategyCombo)
            self.assertGreater(len(combo.strategies), 0)
            self.assertIsInstance(combo.combo_name, str)
            self.assertIsInstance(combo.description, str)
            self.assertIsInstance(combo.expected_synergy, float)
        
        # 특정 조합 확인
        hybrid_combos = [c for c in combinations if "hybrid" in c.combo_name]
        self.assertGreater(len(hybrid_combos), 0)

if __name__ == '__main__':
    unittest.main()