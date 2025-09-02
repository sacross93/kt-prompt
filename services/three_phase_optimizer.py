"""
3단계 프롬프트 최적화기

1단계: 정확도 최우선 최적화 (0.8점 이상 목표)
2단계: 한글 비율 최적화 (90% 이상 목표)  
3단계: 길이 압축 최적화 (3000자 제한)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging
import asyncio

from .kt_score_calculator import KTScoreCalculator, KTScoreBreakdown
from .korean_ratio_optimizer import KoreanRatioOptimizer
from .length_compressor import LengthCompressor
from .accuracy_optimizer import AccuracyOptimizer
from .gemini_flash_classifier import GeminiFlashClassifier
from .kt_score_monitor import KTScoreMonitor
from .auto_accuracy_optimizer import AutoAccuracyOptimizer
from .auto_length_optimizer import AutoLengthOptimizer
from .auto_optimization_summary import AutoOptimizationSummary

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """3단계 최적화 결과"""
    phase1_prompt: str                 # 1단계: 정확도 최적화 결과
    phase2_prompt: str                 # 2단계: 한글 비율 최적화 결과  
    phase3_prompt: str                 # 3단계: 길이 압축 최적화 결과
    phase1_scores: KTScoreBreakdown    # 1단계 점수
    phase2_scores: KTScoreBreakdown    # 2단계 점수
    phase3_scores: KTScoreBreakdown    # 3단계 점수
    optimization_log: List[str]        # 최적화 과정 로그
    final_kt_score: float              # 최종 KT 점수

class ThreePhaseOptimizer:
    """3단계 프롬프트 최적화기"""
    
    def __init__(self, samples_csv_path: str = "data/samples.csv"):
        from config import OptimizationConfig
        
        self.samples_csv_path = samples_csv_path
        self.config = OptimizationConfig.from_env()
        self.kt_calculator = KTScoreCalculator()
        self.korean_optimizer = KoreanRatioOptimizer()
        self.length_compressor = LengthCompressor()
        self.accuracy_optimizer = AccuracyOptimizer(samples_csv_path)
        self.kt_monitor = KTScoreMonitor()
        
        # 자동화된 최적화기들 (나중에 초기화)
        self.auto_accuracy_optimizer = None
        self.auto_length_optimizer = None
        self.auto_summary = AutoOptimizationSummary()
        
        # GeminiFlashClassifier는 나중에 프롬프트와 함께 초기화
        self.gemini_tester = None
        
        self.optimization_log = []
        
    def log(self, message: str):
        """로그 메시지 추가"""
        logger.info(message)
        self.optimization_log.append(message)
        print(f"[3단계 최적화] {message}")
    
    async def execute_full_optimization(self, base_prompt: str) -> OptimizationResult:
        """전체 3단계 최적화 실행"""
        self.log("=== 3단계 프롬프트 최적화 시작 ===")
        
        # 1단계: 정확도 최우선 최적화
        self.log("1단계: 정확도 최우선 최적화 시작")
        phase1_prompt, phase1_accuracy = await self.optimize_phase1_accuracy(base_prompt)
        phase1_scores = self.kt_calculator.calculate_full_score(phase1_accuracy, phase1_prompt)
        self.kt_monitor.record_score("1단계_정확도최적화", phase1_scores)
        self.log(f"1단계 완료 - 정확도: {phase1_accuracy:.4f}, KT점수: {phase1_scores.total_score:.4f}")
        
        # 2단계: 한글 비율 최적화
        self.log("2단계: 한글 비율 최적화 시작")
        phase2_prompt = await self.optimize_phase2_korean_ratio(phase1_prompt)
        phase2_accuracy = await self._test_accuracy(phase2_prompt)
        phase2_scores = self.kt_calculator.calculate_full_score(phase2_accuracy, phase2_prompt)
        self.kt_monitor.record_score("2단계_한글비율최적화", phase2_scores)
        self.log(f"2단계 완료 - 한글비율: {phase2_scores.korean_char_ratio:.4f}, KT점수: {phase2_scores.total_score:.4f}")
        
        # 3단계: 길이 압축 최적화
        self.log("3단계: 길이 압축 최적화 시작")
        phase3_prompt = await self.optimize_phase3_length(phase2_prompt)
        phase3_accuracy = await self._test_accuracy(phase3_prompt)
        phase3_scores = self.kt_calculator.calculate_full_score(phase3_accuracy, phase3_prompt)
        self.kt_monitor.record_score("3단계_길이압축최적화", phase3_scores)
        self.log(f"3단계 완료 - 길이: {phase3_scores.prompt_length}자, KT점수: {phase3_scores.total_score:.4f}")
        
        # 결과 정리
        result = OptimizationResult(
            phase1_prompt=phase1_prompt,
            phase2_prompt=phase2_prompt,
            phase3_prompt=phase3_prompt,
            phase1_scores=phase1_scores,
            phase2_scores=phase2_scores,
            phase3_scores=phase3_scores,
            optimization_log=self.optimization_log.copy(),
            final_kt_score=phase3_scores.total_score
        )
        
        self.log("=== 3단계 최적화 완료 ===")
        self._log_final_summary(result)
        
        # 모니터링 리포트 생성
        monitoring_report = self.kt_monitor.generate_progress_report()
        self.log("\n=== KT 점수 모니터링 리포트 ===")
        self.log(monitoring_report)
        
        # 시각화 출력
        visualization = self.kt_monitor.create_score_visualization()
        self.log("\n=== 점수 변화 시각화 ===")
        self.log(visualization)
        
        return result
    
    async def optimize_phase1_accuracy(self, base_prompt: str, target_accuracy: float = 0.8) -> Tuple[str, float]:
        """1단계: 정확도 최우선 최적화"""
        self.log(f"목표 정확도: {target_accuracy} 이상")
        
        # 기존 프롬프트 성능 측정
        current_accuracy = await self._test_accuracy(base_prompt)
        self.log(f"기준 프롬프트 정확도: {current_accuracy:.4f}")
        
        if current_accuracy >= target_accuracy:
            self.log("이미 목표 정확도 달성")
            return base_prompt, current_accuracy
        
        # 정확도 최적화 수행
        optimized_prompt = await self.accuracy_optimizer.optimize_for_accuracy(
            base_prompt, target_accuracy
        )
        
        # 최적화된 프롬프트 성능 측정
        final_accuracy = await self._test_accuracy(optimized_prompt)
        self.log(f"최적화 후 정확도: {final_accuracy:.4f}")
        
        return optimized_prompt, final_accuracy
    
    async def optimize_phase2_korean_ratio(self, prompt: str, target_ratio: float = 0.9) -> str:
        """2단계: 한글 비율 최적화"""
        current_ratio = self.kt_calculator.calculate_korean_ratio(prompt)
        self.log(f"현재 한글 비율: {current_ratio:.4f}, 목표: {target_ratio}")
        
        if current_ratio >= target_ratio:
            self.log("이미 목표 한글 비율 달성")
            return prompt
        
        # 한글 비율 최적화 수행
        optimized_prompt = self.korean_optimizer.optimize_to_target_ratio(prompt, target_ratio)
        
        final_ratio = self.kt_calculator.calculate_korean_ratio(optimized_prompt)
        self.log(f"최적화 후 한글 비율: {final_ratio:.4f}")
        
        return optimized_prompt
    
    async def optimize_phase3_length(self, prompt: str, max_length: int = 3000) -> str:
        """3단계: 길이 압축 최적화"""
        current_length = len(prompt)
        self.log(f"현재 길이: {current_length}자, 최대: {max_length}자")
        
        if current_length <= max_length * 0.7:  # 이미 충분히 짧으면
            self.log("이미 충분히 짧은 프롬프트")
            return prompt
        
        # 길이 압축 최적화 수행
        compressed_prompt = await self.length_compressor.optimize_information_density(
            prompt, max_length
        )
        
        final_length = len(compressed_prompt)
        self.log(f"압축 후 길이: {final_length}자 ({current_length - final_length}자 단축)")
        
        return compressed_prompt
    
    def validate_phase_completion(self, phase: int, metrics: Dict[str, float]) -> bool:
        """단계별 완료 조건 검증"""
        if phase == 1:
            # 1단계: 정확도 0.8 이상
            return metrics.get('accuracy', 0.0) >= 0.8
        elif phase == 2:
            # 2단계: 한글 비율 90% 이상
            return metrics.get('korean_ratio', 0.0) >= 0.9
        elif phase == 3:
            # 3단계: 3000자 이하
            return metrics.get('length', 3001) <= 3000
        
        return False
    
    async def _test_accuracy(self, prompt: str) -> float:
        """프롬프트 정확도 테스트"""
        try:
            # GeminiFlashClassifier 초기화 (프롬프트와 함께)
            if self.gemini_tester is None or self.gemini_tester.get_system_prompt() != prompt:
                self.gemini_tester = GeminiFlashClassifier(self.config, prompt)
            
            results = await self.gemini_tester.test_prompt_performance(prompt, self.samples_csv_path)
            return results.get('accuracy', 0.0)
        except Exception as e:
            self.log(f"정확도 테스트 오류: {e}")
            return 0.0
    
    def _log_final_summary(self, result: OptimizationResult):
        """최종 결과 요약 로그"""
        self.log("\n=== 최종 결과 요약 ===")
        
        # 단계별 점수 비교
        scores = [result.phase1_scores, result.phase2_scores, result.phase3_scores]
        phases = ["1단계(정확도)", "2단계(한글화)", "3단계(압축)"]
        
        for i, (phase, score) in enumerate(zip(phases, scores)):
            self.log(f"{phase}: KT점수 {score.total_score:.4f} "
                    f"(정확도:{score.accuracy_score:.3f}, 한글:{score.korean_ratio_score:.3f}, "
                    f"길이:{score.length_score:.3f})")
        
        # 최종 성과
        final_score = result.phase3_scores
        self.log(f"\n최종 KT 점수: {final_score.total_score:.4f}")
        self.log(f"- 정확도: {final_score.accuracy_score/0.8:.4f} (가중치 적용: {final_score.accuracy_score:.4f})")
        self.log(f"- 한글비율: {final_score.korean_char_ratio:.4f} (가중치 적용: {final_score.korean_ratio_score:.4f})")
        self.log(f"- 길이점수: {final_score.length_score:.4f} (길이: {final_score.prompt_length}자)")
        
        # 목표 달성 여부
        if final_score.total_score >= 0.9:
            self.log("🎉 목표 점수 0.9점 달성!")
        else:
            self.log(f"목표까지 {0.9 - final_score.total_score:.4f}점 부족")
    
    def get_optimization_report(self, result: OptimizationResult) -> str:
        """최적화 리포트 생성"""
        report = "# 3단계 프롬프트 최적화 리포트\n\n"
        
        # 단계별 결과
        report += "## 단계별 결과\n\n"
        phases = [
            ("1단계: 정확도 최적화", result.phase1_scores),
            ("2단계: 한글 비율 최적화", result.phase2_scores),
            ("3단계: 길이 압축 최적화", result.phase3_scores)
        ]
        
        for phase_name, scores in phases:
            report += f"### {phase_name}\n"
            report += f"- KT 총점: {scores.total_score:.4f}\n"
            report += f"- 정확도: {scores.accuracy_score/0.8:.4f} (가중치: {scores.accuracy_score:.4f})\n"
            report += f"- 한글비율: {scores.korean_char_ratio:.4f} (가중치: {scores.korean_ratio_score:.4f})\n"
            report += f"- 길이점수: {scores.length_score:.4f} (길이: {scores.prompt_length}자)\n\n"
        
        # 최종 성과
        report += "## 최종 성과\n\n"
        final = result.phase3_scores
        report += f"**최종 KT 점수: {final.total_score:.4f}**\n\n"
        
        if final.total_score >= 0.9:
            report += "✅ **목표 점수 0.9점 달성!**\n\n"
        else:
            needed = 0.9 - final.total_score
            report += f"❌ 목표까지 {needed:.4f}점 부족\n\n"
        
        # 개선 제안
        if final.improvement_suggestions:
            report += "## 추가 개선 제안\n\n"
            for i, suggestion in enumerate(final.improvement_suggestions, 1):
                report += f"{i}. {suggestion}\n"
        
        return report    

    async def execute_automated_optimization(self, base_prompt: str) -> OptimizationResult:
        """자동화된 3단계 최적화 실행 (Gemini Pro 기반)"""
        self.log("=== 자동화된 3단계 프롬프트 최적화 시작 ===")
        
        # 1단계: 자동화된 정확도 최적화 (Gemini Pro 기반)
        self.log("1단계: 자동화된 정확도 최적화 시작")
        
        # 자동 정확도 최적화기 초기화
        if self.auto_accuracy_optimizer is None:
            self.auto_accuracy_optimizer = AutoAccuracyOptimizer(self.samples_csv_path)
        
        # 임시 파일로 기본 프롬프트 저장
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(base_prompt)
            temp_prompt_path = f.name
        
        try:
            # 자동 정확도 최적화 실행
            phase1_prompt_path = await self.auto_accuracy_optimizer.optimize_accuracy_automatically(
                temp_prompt_path, target_accuracy=0.8, max_iterations=5
            )
            
            with open(phase1_prompt_path, 'r', encoding='utf-8') as f:
                phase1_prompt = f.read().strip()
            
            phase1_accuracy = await self._test_accuracy(phase1_prompt)
            phase1_scores = self.kt_calculator.calculate_full_score(phase1_accuracy, phase1_prompt)
            self.kt_monitor.record_score("1단계_자동정확도최적화", phase1_scores)
            self.log(f"1단계 완료 - 정확도: {phase1_accuracy:.4f}, KT점수: {phase1_scores.total_score:.4f}")
            
            # 2단계: 한글 비율 최적화 (기존 방식)
            self.log("2단계: 한글 비율 최적화 시작")
            phase2_prompt = await self.optimize_phase2_korean_ratio(phase1_prompt)
            phase2_accuracy = await self._test_accuracy(phase2_prompt)
            phase2_scores = self.kt_calculator.calculate_full_score(phase2_accuracy, phase2_prompt)
            self.kt_monitor.record_score("2단계_한글비율최적화", phase2_scores)
            self.log(f"2단계 완료 - 한글비율: {phase2_scores.korean_char_ratio:.4f}, KT점수: {phase2_scores.total_score:.4f}")
            
            # 3단계: 자동화된 길이 압축 최적화 (Gemini Pro 기반)
            self.log("3단계: 자동화된 길이 압축 최적화 시작")
            
            # 자동 길이 최적화기 초기화
            if self.auto_length_optimizer is None:
                self.auto_length_optimizer = AutoLengthOptimizer(self.samples_csv_path)
            
            # 임시 파일로 2단계 프롬프트 저장
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                f.write(phase2_prompt)
                phase2_temp_path = f.name
            
            try:
                phase3_prompt_path, optimization_info = await self.auto_length_optimizer.optimize_with_accuracy_monitoring(
                    phase2_temp_path, target_length=3000, max_accuracy_loss=0.05
                )
                
                with open(phase3_prompt_path, 'r', encoding='utf-8') as f:
                    phase3_prompt = f.read().strip()
                
                phase3_accuracy = optimization_info['compressed_accuracy']
                phase3_scores = self.kt_calculator.calculate_full_score(phase3_accuracy, phase3_prompt)
                self.kt_monitor.record_score("3단계_자동길이압축최적화", phase3_scores)
                self.log(f"3단계 완료 - 길이: {phase3_scores.prompt_length}자, KT점수: {phase3_scores.total_score:.4f}")
                
            finally:
                # 임시 파일 정리
                import os
                if os.path.exists(phase2_temp_path):
                    os.unlink(phase2_temp_path)
            
            # 결과 정리
            result = OptimizationResult(
                phase1_prompt=phase1_prompt,
                phase2_prompt=phase2_prompt,
                phase3_prompt=phase3_prompt,
                phase1_scores=phase1_scores,
                phase2_scores=phase2_scores,
                phase3_scores=phase3_scores,
                optimization_log=self.optimization_log.copy(),
                final_kt_score=phase3_scores.total_score
            )
            
            self.log("=== 자동화된 3단계 최적화 완료 ===")
            self._log_final_summary(result)
            
            # 자동 최적화 리포트 추가
            auto_accuracy_report = self.auto_accuracy_optimizer.get_optimization_report()
            self.log("\n=== 자동 정확도 최적화 리포트 ===")
            self.log(auto_accuracy_report)
            
            # 종합 자동화 리포트 생성
            accuracy_history = self.auto_accuracy_optimizer.optimization_history
            accuracy_history_data = [
                {
                    'accuracy': record.accuracy,
                    'kt_score': record.kt_score,
                    'error_count': record.error_count,
                    'timestamp': record.timestamp,
                    'improvements': record.improvements
                }
                for record in accuracy_history
            ]
            
            # 길이 최적화 정보 (있는 경우)
            length_info = optimization_info if 'optimization_info' in locals() else None
            
            # 종합 리포트 생성 및 저장
            artifacts = self.auto_summary.save_optimization_artifacts(
                phase3_prompt, accuracy_history_data, length_info, phase3_scores.total_score
            )
            
            self.log("\n=== 🤖 자동화된 최적화 종합 리포트 ===")
            comprehensive_report = self.auto_summary.generate_comprehensive_report(
                accuracy_history_data, length_info, phase3_scores.total_score
            )
            self.log(comprehensive_report)
            
            # 성능 시각화
            performance_viz = self.auto_summary.create_performance_visualization(accuracy_history_data)
            self.log("\n=== 📊 성능 변화 시각화 ===")
            self.log(performance_viz)
            
            # 저장된 파일 정보
            self.log("\n=== 💾 저장된 결과물 ===")
            for artifact_type, file_path in artifacts.items():
                self.log(f"- {artifact_type}: {file_path}")
            
            # 모니터링 리포트 생성
            monitoring_report = self.kt_monitor.generate_progress_report()
            self.log("\n=== KT 점수 모니터링 리포트 ===")
            self.log(monitoring_report)
            
            # 시각화 출력
            visualization = self.kt_monitor.create_score_visualization()
            self.log("\n=== 점수 변화 시각화 ===")
            self.log(visualization)
            
            return result
            
        finally:
            # 임시 파일 정리
            import os
            if os.path.exists(temp_prompt_path):
                os.unlink(temp_prompt_path)