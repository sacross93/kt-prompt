"""
자동화된 길이 압축 최적화기

Gemini 2.5 Pro를 사용하여 프롬프트 내용을 보존하면서 지능적으로 압축
"""

import os
from typing import Tuple, Dict
import logging

from .gemini_pro_analyzer import GeminiProAnalyzer
from .gemini_flash_classifier import GeminiFlashClassifier
from .kt_score_calculator import KTScoreCalculator
from config import OptimizationConfig

logger = logging.getLogger(__name__)

class AutoLengthOptimizer:
    """자동화된 길이 압축 최적화기"""
    
    def __init__(self, samples_csv_path: str = "data/samples.csv"):
        self.samples_csv_path = samples_csv_path
        self.config = OptimizationConfig.from_env()
        self.kt_calculator = KTScoreCalculator()
        
    async def optimize_length_intelligently(
        self, 
        prompt_path: str, 
        target_length: int = 3000,
        min_accuracy_retention: float = 0.95
    ) -> Tuple[str, float, float]:
        """지능적 길이 압축 최적화
        
        Returns:
            Tuple[압축된_프롬프트_경로, 압축_후_정확도, 압축률]
        """
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            original_prompt = f.read().strip()
        
        original_length = len(original_prompt)
        
        logger.info(f"지능적 길이 압축 시작 - 원본: {original_length}자, 목표: {target_length}자")
        
        # 압축이 필요한지 확인
        if original_length <= target_length:
            logger.info("이미 목표 길이 이하입니다.")
            return prompt_path, 1.0, 1.0
        
        # 원본 정확도 측정
        original_accuracy = await self._test_accuracy(original_prompt)
        logger.info(f"원본 정확도: {original_accuracy:.4f}")
        
        # Gemini 2.5 Pro로 지능적 압축 요청
        compression_prompt = self._create_compression_prompt(original_prompt, target_length)
        
        analyzer = GeminiProAnalyzer(self.config)
        compressed_prompt = await analyzer.compress_prompt_intelligently(compression_prompt)
        
        compressed_length = len(compressed_prompt)
        compression_ratio = compressed_length / original_length
        
        logger.info(f"압축 완료 - {original_length}자 → {compressed_length}자 (압축률: {compression_ratio:.2%})")
        
        # 압축된 프롬프트 정확도 테스트
        compressed_accuracy = await self._test_accuracy(compressed_prompt)
        accuracy_retention = compressed_accuracy / original_accuracy if original_accuracy > 0 else 0
        
        logger.info(f"압축 후 정확도: {compressed_accuracy:.4f} (보존률: {accuracy_retention:.2%})")
        
        # 정확도 보존률 확인
        if accuracy_retention < min_accuracy_retention:
            logger.warning(f"정확도 보존률이 낮습니다: {accuracy_retention:.2%} < {min_accuracy_retention:.2%}")
        
        # 압축된 프롬프트 저장
        output_dir = os.path.dirname(prompt_path)
        base_name = os.path.splitext(os.path.basename(prompt_path))[0]
        compressed_path = os.path.join(output_dir, f"{base_name}_compressed.txt")
        
        with open(compressed_path, 'w', encoding='utf-8') as f:
            f.write(compressed_prompt)
        
        logger.info(f"압축된 프롬프트 저장: {compressed_path}")
        
        return compressed_path, compressed_accuracy, compression_ratio
    
    def _create_compression_prompt(self, original_prompt: str, target_length: int) -> str:
        """압축 요청 프롬프트 생성"""
        
        current_length = len(original_prompt)
        reduction_needed = current_length - target_length
        reduction_percentage = (reduction_needed / current_length) * 100
        
        compression_prompt = f"""
당신은 프롬프트 압축 전문가입니다. 주어진 프롬프트의 핵심 내용과 성능을 최대한 보존하면서 길이를 줄여주세요.

현재 프롬프트:
```
{original_prompt}
```

압축 요구사항:
- 현재 길이: {current_length}자
- 목표 길이: {target_length}자 이하
- 줄여야 할 길이: {reduction_needed}자 ({reduction_percentage:.1f}% 감소)

압축 전략:
1. **핵심 분류 규칙 보존**: 유형/극성/시제/확실성 분류 기준은 반드시 유지
2. **중복 제거**: 반복되는 설명이나 예시 통합
3. **간결한 표현**: 장황한 설명을 핵심만 남기고 압축
4. **예시 최적화**: 가장 효과적인 예시만 선별하여 유지
5. **불필요한 수식어 제거**: "매우", "정말", "반드시" 등 강조 표현 최소화

주의사항:
- 분류 정확도에 영향을 주는 핵심 규칙은 절대 삭제하지 마세요
- 출력 형식 지시사항은 반드시 유지하세요
- 한글 비율을 현재 수준 이상으로 유지하세요

압축된 프롬프트만 출력해주세요 (설명이나 주석 없이):
"""
        
        return compression_prompt
    
    async def _test_accuracy(self, prompt_text: str) -> float:
        """프롬프트 정확도 테스트"""
        try:
            classifier = GeminiFlashClassifier(self.config, prompt_text)
            results = await classifier.test_prompt_performance(prompt_text, self.samples_csv_path)
            return results.get('accuracy', 0.0)
        except Exception as e:
            logger.error(f"정확도 테스트 실패: {e}")
            return 0.0
    
    async def optimize_with_accuracy_monitoring(
        self, 
        prompt_path: str, 
        target_length: int = 3000,
        max_accuracy_loss: float = 0.05
    ) -> Tuple[str, Dict]:
        """정확도 모니터링과 함께 길이 최적화
        
        Returns:
            Tuple[최적화된_프롬프트_경로, 최적화_결과_정보]
        """
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            original_prompt = f.read().strip()
        
        # 원본 성능 측정
        original_accuracy = await self._test_accuracy(original_prompt)
        original_kt_score = self.kt_calculator.calculate_full_score(original_accuracy, original_prompt)
        
        logger.info(f"원본 성능 - 정확도: {original_accuracy:.4f}, KT점수: {original_kt_score.total_score:.4f}")
        
        # 지능적 압축 수행
        compressed_path, compressed_accuracy, compression_ratio = await self.optimize_length_intelligently(
            prompt_path, target_length
        )
        
        # 압축된 프롬프트 성능 측정
        with open(compressed_path, 'r', encoding='utf-8') as f:
            compressed_prompt = f.read().strip()
        
        compressed_kt_score = self.kt_calculator.calculate_full_score(compressed_accuracy, compressed_prompt)
        
        # 성능 변화 분석
        accuracy_change = compressed_accuracy - original_accuracy
        kt_score_change = compressed_kt_score.total_score - original_kt_score.total_score
        
        optimization_info = {
            'original_length': len(original_prompt),
            'compressed_length': len(compressed_prompt),
            'compression_ratio': compression_ratio,
            'original_accuracy': original_accuracy,
            'compressed_accuracy': compressed_accuracy,
            'accuracy_change': accuracy_change,
            'original_kt_score': original_kt_score.total_score,
            'compressed_kt_score': compressed_kt_score.total_score,
            'kt_score_change': kt_score_change,
            'success': abs(accuracy_change) <= max_accuracy_loss
        }
        
        logger.info(f"압축 결과 - 길이: {compression_ratio:.2%}, 정확도 변화: {accuracy_change:+.4f}, KT점수 변화: {kt_score_change:+.4f}")
        
        return compressed_path, optimization_info