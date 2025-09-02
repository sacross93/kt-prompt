"""
반복적 정확도 최적화기

Gemini 2.5 Flash로 테스트 → Gemini 2.5 Pro로 분석 → 프롬프트 개선 → 반복
"""

import asyncio
import os
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

from .gemini_flash_classifier import GeminiFlashClassifier
from .gemini_client import GeminiClient
from config import OptimizationConfig

logger = logging.getLogger(__name__)

@dataclass
class ErrorCase:
    """오류 사례"""
    sentence: str
    predicted: str
    expected: str
    error_type: str

@dataclass
class AnalysisResult:
    """분석 결과"""
    root_causes: List[str]
    improvement_suggestions: List[str]
    revised_prompt: str

class IterativeAccuracyOptimizer:
    """반복적 정확도 최적화기"""
    
    def __init__(self, samples_csv_path: str = "data/samples.csv"):
        self.samples_csv_path = samples_csv_path
        self.config = OptimizationConfig.from_env()
        self.gemini_client = GeminiClient(self.config)
        
        # 모델 초기화
        self.flash_model = self.gemini_client.get_flash_model()
        self.pro_model = self.gemini_client.get_pro_model()
        
        self.iteration_count = 0
        self.max_iterations = 10
        self.target_accuracy = 0.8
        
    async def optimize_accuracy_iteratively(self, initial_prompt: str, target_accuracy: float = 0.8) -> Tuple[str, float]:
        """반복적 정확도 최적화"""
        self.target_accuracy = target_accuracy
        current_prompt = initial_prompt
        best_prompt = initial_prompt
        best_accuracy = 0.0
        
        logger.info(f"🎯 목표 정확도: {target_accuracy:.2%}")
        logger.info(f"🔄 최대 반복 횟수: {self.max_iterations}")
        
        for iteration in range(self.max_iterations):
            self.iteration_count = iteration + 1
            logger.info(f"\n=== 반복 {self.iteration_count}/{self.max_iterations} ===")
            
            # 1단계: Gemini 2.5 Flash로 테스트
            logger.info("1️⃣ Gemini 2.5 Flash 테스트 중...")
            accuracy, error_cases = await self._test_with_flash(current_prompt)
            
            logger.info(f"📊 현재 정확도: {accuracy:.4f} ({accuracy:.1%})")
            
            # 최고 성능 업데이트
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_prompt = current_prompt
                logger.info(f"🏆 최고 성능 갱신: {best_accuracy:.4f}")
            
            # 목표 달성 확인
            if accuracy >= target_accuracy:
                logger.info(f"🎉 목표 정확도 달성! {accuracy:.4f} >= {target_accuracy:.4f}")
                return current_prompt, accuracy
            
            # 개선 여지가 없으면 중단
            if len(error_cases) == 0:
                logger.info("❌ 더 이상 개선할 오류가 없습니다.")
                break
            
            # 2단계: Gemini 2.5 Pro로 오류 분석
            logger.info("2️⃣ Gemini 2.5 Pro 오류 분석 중...")
            analysis = await self._analyze_errors_with_pro(current_prompt, error_cases)
            
            # 3단계: 프롬프트 개선
            logger.info("3️⃣ 프롬프트 개선 중...")
            current_prompt = analysis.revised_prompt
            
            # 개선 내용 로그
            logger.info("📝 개선 사항:")
            for suggestion in analysis.improvement_suggestions[:3]:  # 상위 3개만
                logger.info(f"   - {suggestion}")
            
            # 프롬프트 저장
            await self._save_iteration_prompt(current_prompt, iteration + 1, accuracy)
        
        logger.info(f"\n🏁 최적화 완료 - 최고 성능: {best_accuracy:.4f}")
        return best_prompt, best_accuracy
    
    async def _test_with_flash(self, prompt: str) -> Tuple[float, List[ErrorCase]]:
        """Gemini 2.5 Flash로 테스트 및 오류 수집"""
        try:
            # GeminiFlashClassifier 초기화
            classifier = GeminiFlashClassifier(self.config, prompt)
            
            # 성능 테스트 (100개 샘플)
            results = await classifier.test_prompt_performance(prompt, self.samples_csv_path)
            
            accuracy = results.get('accuracy', 0.0)
            errors = results.get('errors', [])
            
            # 오류 사례 변환
            error_cases = []
            for error in errors[:20]:  # 최대 20개 오류만 분석
                error_case = ErrorCase(
                    sentence=f"문장 {error['index']}",  # 실제 문장은 나중에 추가
                    predicted=error['predicted'],
                    expected=error['expected'],
                    error_type=self._classify_error_type(error['predicted'], error['expected'])
                )
                error_cases.append(error_case)
            
            return accuracy, error_cases
            
        except Exception as e:
            logger.error(f"Flash 테스트 실패: {e}")
            return 0.0, []
    
    def _classify_error_type(self, predicted: str, expected: str) -> str:
        """오류 유형 분류"""
        pred_parts = predicted.split(',')
        exp_parts = expected.split(',')
        
        if len(pred_parts) != 4 or len(exp_parts) != 4:
            return "형식오류"
        
        error_attrs = []
        attrs = ['유형', '극성', '시제', '확실성']
        
        for i, attr in enumerate(attrs):
            if pred_parts[i].strip() != exp_parts[i].strip():
                error_attrs.append(attr)
        
        return '+'.join(error_attrs) if error_attrs else "기타"
    
    async def _analyze_errors_with_pro(self, current_prompt: str, error_cases: List[ErrorCase]) -> AnalysisResult:
        """Gemini 2.5 Pro로 오류 분석 및 프롬프트 개선"""
        
        # 오류 패턴 분석
        error_summary = self._summarize_errors(error_cases)
        
        # Gemini 2.5 Pro 분석 프롬프트 생성
        analysis_prompt = f"""
당신은 한국어 문장 분류 프롬프트 최적화 전문가입니다.

현재 프롬프트:
```
{current_prompt}
```

발생한 오류들:
{error_summary}

위 오류들을 분석하여 다음을 제공해주세요:

1. 근본 원인 분석 (3-5개):
   - 왜 이런 오류가 발생했는지 구체적 원인

2. 개선 제안사항 (3-5개):
   - 어떻게 프롬프트를 수정해야 하는지 구체적 방법

3. 개선된 프롬프트:
   - 위 분석을 바탕으로 완전히 새로운 프롬프트 작성
   - 기존 프롬프트의 좋은 부분은 유지하되 문제점은 해결
   - 한국어로 작성하고 명확하고 간결하게

응답 형식:
## 근본 원인 분석
1. [원인1]
2. [원인2]
...

## 개선 제안사항  
1. [제안1]
2. [제안2]
...

## 개선된 프롬프트
```
[새로운 프롬프트 전체 내용]
```
"""
        
        try:
            # Gemini 2.5 Pro로 분석 요청
            response = self.gemini_client.generate_content_with_retry(
                self.pro_model, analysis_prompt
            )
            
            # 응답 파싱
            root_causes, suggestions, revised_prompt = self._parse_pro_response(response)
            
            return AnalysisResult(
                root_causes=root_causes,
                improvement_suggestions=suggestions,
                revised_prompt=revised_prompt
            )
            
        except Exception as e:
            logger.error(f"Pro 분석 실패: {e}")
            # 실패 시 기본 개선안 반환
            return AnalysisResult(
                root_causes=["분석 실패"],
                improvement_suggestions=["기본 개선 적용"],
                revised_prompt=current_prompt  # 원본 유지
            )
    
    def _summarize_errors(self, error_cases: List[ErrorCase]) -> str:
        """오류 요약"""
        if not error_cases:
            return "오류 없음"
        
        # 오류 유형별 집계
        error_types = {}
        for case in error_cases:
            error_types[case.error_type] = error_types.get(case.error_type, 0) + 1
        
        summary = f"총 {len(error_cases)}개 오류:\n"
        
        # 상위 오류 유형들
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5]:
            summary += f"- {error_type}: {count}개\n"
        
        # 구체적 오류 예시 (상위 5개)
        summary += "\n구체적 오류 예시:\n"
        for i, case in enumerate(error_cases[:5]):
            summary += f"{i+1}. 예측: {case.predicted} → 정답: {case.expected} (유형: {case.error_type})\n"
        
        return summary
    
    def _parse_pro_response(self, response: str) -> Tuple[List[str], List[str], str]:
        """Gemini Pro 응답 파싱"""
        try:
            lines = response.split('\n')
            
            root_causes = []
            suggestions = []
            revised_prompt = ""
            
            current_section = None
            in_prompt = False
            
            for line in lines:
                line = line.strip()
                
                if "근본 원인" in line or "원인 분석" in line:
                    current_section = "causes"
                elif "개선 제안" in line or "제안사항" in line:
                    current_section = "suggestions"
                elif "개선된 프롬프트" in line or "새로운 프롬프트" in line:
                    current_section = "prompt"
                elif line.startswith("```"):
                    in_prompt = not in_prompt
                    continue
                
                if current_section == "causes" and line.startswith(("1.", "2.", "3.", "4.", "5.", "-")):
                    root_causes.append(line[2:].strip())
                elif current_section == "suggestions" and line.startswith(("1.", "2.", "3.", "4.", "5.", "-")):
                    suggestions.append(line[2:].strip())
                elif current_section == "prompt" and in_prompt:
                    revised_prompt += line + "\n"
            
            # 프롬프트가 비어있으면 응답 전체를 프롬프트로 사용
            if not revised_prompt.strip():
                # 마지막 ``` 이후 내용을 프롬프트로 추출
                prompt_start = response.rfind("```")
                if prompt_start != -1:
                    revised_prompt = response[prompt_start+3:].strip()
                else:
                    revised_prompt = response  # 전체 응답 사용
            
            return root_causes, suggestions, revised_prompt.strip()
            
        except Exception as e:
            logger.error(f"응답 파싱 실패: {e}")
            return [], [], response  # 실패 시 전체 응답을 프롬프트로 사용
    
    async def _save_iteration_prompt(self, prompt: str, iteration: int, accuracy: float):
        """반복별 프롬프트 저장"""
        try:
            os.makedirs("prompt/gemini/iterations", exist_ok=True)
            filename = f"prompt/gemini/iterations/iteration_{iteration:02d}_acc_{accuracy:.4f}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(prompt)
            
            logger.info(f"💾 프롬프트 저장: {filename}")
            
        except Exception as e:
            logger.error(f"프롬프트 저장 실패: {e}")