"""
자동화된 정확도 최적화 시스템

Gemini 2.5 Flash로 테스트 → Gemini 2.5 Pro로 분석 및 개선 → 반복
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

from .gemini_flash_classifier import GeminiFlashClassifier
from .gemini_pro_analyzer import GeminiProAnalyzer
from .kt_score_calculator import KTScoreCalculator
from config import OptimizationConfig

logger = logging.getLogger(__name__)

@dataclass
class ErrorAnalysis:
    """오답 분석 결과"""
    wrong_questions: List[str]
    wrong_predictions: List[str]
    correct_answers: List[str]
    error_patterns: List[str]
    improvement_suggestions: List[str]

@dataclass
class OptimizationIteration:
    """최적화 반복 기록"""
    iteration: int
    timestamp: str
    prompt_path: str
    accuracy: float
    kt_score: float
    error_count: int
    improvements: List[str]

class AutoAccuracyOptimizer:
    """자동화된 정확도 최적화 시스템"""
    
    def __init__(self, samples_csv_path: str = "data/samples.csv", output_dir: str = "prompt/auto_optimized"):
        self.samples_csv_path = samples_csv_path
        self.output_dir = output_dir
        self.config = OptimizationConfig.from_env()
        self.kt_calculator = KTScoreCalculator()
        
        # 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 최적화 히스토리
        self.optimization_history: List[OptimizationIteration] = []
        
    async def optimize_accuracy_automatically(
        self, 
        initial_prompt_path: str, 
        target_accuracy: float = 0.8,
        max_iterations: int = 10
    ) -> str:
        """자동화된 정확도 최적화 실행"""
        
        logger.info(f"자동 정확도 최적화 시작 - 목표: {target_accuracy}, 최대 반복: {max_iterations}")
        
        current_prompt_path = initial_prompt_path
        
        for iteration in range(1, max_iterations + 1):
            logger.info(f"\n=== 반복 {iteration}/{max_iterations} 시작 ===")
            
            # 1단계: Gemini 2.5 Flash로 테스트
            accuracy, errors = await self._test_with_flash(current_prompt_path)
            
            logger.info(f"현재 정확도: {accuracy:.4f} (목표: {target_accuracy})")
            
            # 목표 달성 시 종료
            if accuracy >= target_accuracy:
                logger.info(f"🎉 목표 정확도 달성! 최종 프롬프트: {current_prompt_path}")
                return current_prompt_path
            
            # 2단계: Gemini 2.5 Pro로 오답 분석
            error_analysis = await self._analyze_errors_with_pro(current_prompt_path, errors)
            
            # 3단계: Gemini 2.5 Pro로 개선된 프롬프트 생성
            improved_prompt_path = await self._generate_improved_prompt_with_pro(
                current_prompt_path, error_analysis, iteration
            )
            
            # 히스토리 기록
            self._record_iteration(iteration, current_prompt_path, accuracy, len(errors), error_analysis.improvement_suggestions)
            
            current_prompt_path = improved_prompt_path
            
            logger.info(f"반복 {iteration} 완료 - 다음 프롬프트: {current_prompt_path}")
        
        logger.warning(f"최대 반복 수 도달. 최종 프롬프트: {current_prompt_path}")
        return current_prompt_path
    
    async def _test_with_flash(self, prompt_path: str) -> Tuple[float, List[Dict]]:
        """Gemini 2.5 Flash로 정확도 테스트 및 오답 수집"""
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_text = f.read().strip()
        
        # Flash 분류기 초기화
        classifier = GeminiFlashClassifier(self.config, prompt_text)
        
        # 테스트 실행 (더 많은 샘플로)
        results = await classifier.test_prompt_performance(prompt_text, self.samples_csv_path)
        
        accuracy = results.get('accuracy', 0.0)
        
        # 오답 수집
        errors = []
        if 'detailed_results' in results:
            for result in results['detailed_results']:
                if not result.get('correct', False):
                    errors.append({
                        'question': result.get('question', ''),
                        'predicted': result.get('predicted', ''),
                        'actual': result.get('actual', ''),
                        'explanation': result.get('explanation', '')
                    })
        
        logger.info(f"테스트 완료 - 정확도: {accuracy:.4f}, 오답: {len(errors)}개")
        return accuracy, errors
    
    async def _analyze_errors_with_pro(self, prompt_path: str, errors: List[Dict]) -> ErrorAnalysis:
        """Gemini 2.5 Pro로 오답 분석"""
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            current_prompt = f.read().strip()
        
        # 오답 분석 요청 프롬프트 생성
        analysis_prompt = self._create_error_analysis_prompt(current_prompt, errors)
        
        # Gemini 2.5 Pro로 분석
        analyzer = GeminiProAnalyzer(self.config)
        analysis_result = await analyzer.analyze_prompt_errors(analysis_prompt)
        
        # 결과 파싱
        error_analysis = self._parse_error_analysis(analysis_result, errors)
        
        logger.info(f"오답 분석 완료 - 패턴: {len(error_analysis.error_patterns)}개, 제안: {len(error_analysis.improvement_suggestions)}개")
        return error_analysis
    
    def _create_error_analysis_prompt(self, current_prompt: str, errors: List[Dict]) -> str:
        """오답 분석용 프롬프트 생성"""
        
        error_examples = ""
        for i, error in enumerate(errors[:10], 1):  # 최대 10개 오답만
            error_examples += f"""
오답 {i}:
- 문장: "{error['question']}"
- 예측: {error['predicted']}
- 정답: {error['actual']}
"""
        
        analysis_prompt = f"""
당신은 한국어 문장 분류 프롬프트 개선 전문가입니다.

현재 프롬프트:
```
{current_prompt}
```

이 프롬프트로 분류한 결과 다음과 같은 오답들이 발생했습니다:
{error_examples}

다음을 분석해주세요:

1. **오답 패턴 분석**: 어떤 유형의 문장에서 주로 틀리는가?
2. **프롬프트 문제점**: 현재 프롬프트의 어떤 부분이 이런 오답을 유발하는가?
3. **개선 방향**: 이런 오답을 줄이기 위해 프롬프트를 어떻게 수정해야 하는가?

분석 결과를 다음 형식으로 제공해주세요:

## 오답 패턴
- 패턴 1: ...
- 패턴 2: ...

## 프롬프트 문제점
- 문제점 1: ...
- 문제점 2: ...

## 개선 제안
- 제안 1: ...
- 제안 2: ...
"""
        
        return analysis_prompt
    
    def _parse_error_analysis(self, analysis_result: str, errors: List[Dict]) -> ErrorAnalysis:
        """분석 결과 파싱"""
        
        # 간단한 파싱 (실제로는 더 정교한 파싱 필요)
        error_patterns = []
        improvement_suggestions = []
        
        lines = analysis_result.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if '## 오답 패턴' in line:
                current_section = 'patterns'
            elif '## 개선 제안' in line:
                current_section = 'suggestions'
            elif line.startswith('- ') and current_section == 'patterns':
                error_patterns.append(line[2:])
            elif line.startswith('- ') and current_section == 'suggestions':
                improvement_suggestions.append(line[2:])
        
        return ErrorAnalysis(
            wrong_questions=[e['question'] for e in errors],
            wrong_predictions=[e['predicted'] for e in errors],
            correct_answers=[e['actual'] for e in errors],
            error_patterns=error_patterns,
            improvement_suggestions=improvement_suggestions
        )
    
    async def _generate_improved_prompt_with_pro(
        self, 
        current_prompt_path: str, 
        error_analysis: ErrorAnalysis, 
        iteration: int
    ) -> str:
        """Gemini 2.5 Pro로 개선된 프롬프트 생성"""
        
        with open(current_prompt_path, 'r', encoding='utf-8') as f:
            current_prompt = f.read().strip()
        
        # 프롬프트 개선 요청 생성
        improvement_prompt = self._create_improvement_prompt(current_prompt, error_analysis)
        
        # Gemini 2.5 Pro로 개선된 프롬프트 생성
        analyzer = GeminiProAnalyzer(self.config)
        improved_prompt = await analyzer.generate_improved_prompt(improvement_prompt)
        
        # 새 프롬프트 파일 저장
        new_prompt_path = os.path.join(self.output_dir, f"auto_optimized_v{iteration}.txt")
        with open(new_prompt_path, 'w', encoding='utf-8') as f:
            f.write(improved_prompt)
        
        logger.info(f"개선된 프롬프트 생성: {new_prompt_path}")
        return new_prompt_path
    
    def _create_improvement_prompt(self, current_prompt: str, error_analysis: ErrorAnalysis) -> str:
        """프롬프트 개선 요청 생성"""
        
        patterns_text = '\n'.join([f"- {p}" for p in error_analysis.error_patterns])
        suggestions_text = '\n'.join([f"- {s}" for s in error_analysis.improvement_suggestions])
        
        improvement_prompt = f"""
당신은 한국어 문장 분류 프롬프트 개선 전문가입니다.

현재 프롬프트:
```
{current_prompt}
```

분석된 오답 패턴:
{patterns_text}

개선 제안사항:
{suggestions_text}

위 분석을 바탕으로 더 정확한 분류를 위한 개선된 프롬프트를 생성해주세요.

개선 시 고려사항:
1. 오답 패턴을 해결할 수 있는 명확한 규칙 추가
2. 애매한 표현 제거 및 구체적인 기준 제시
3. 실제 데이터 분포 반영 (사실형 82%, 긍정 95%, 과거 48%, 확실 92%)
4. 한글 비율 유지 (현재 수준 이상)
5. 길이는 3000자 이내 유지

개선된 프롬프트만 출력해주세요 (설명 없이):
"""
        
        return improvement_prompt
    
    def _record_iteration(
        self, 
        iteration: int, 
        prompt_path: str, 
        accuracy: float, 
        error_count: int, 
        improvements: List[str]
    ):
        """반복 기록"""
        
        # KT 점수 계산
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_text = f.read().strip()
        
        kt_score_breakdown = self.kt_calculator.calculate_full_score(accuracy, prompt_text)
        
        record = OptimizationIteration(
            iteration=iteration,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            prompt_path=prompt_path,
            accuracy=accuracy,
            kt_score=kt_score_breakdown.total_score,
            error_count=error_count,
            improvements=improvements
        )
        
        self.optimization_history.append(record)
        
        # 히스토리 저장
        history_file = os.path.join(self.output_dir, "optimization_history.json")
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump([record.__dict__ for record in self.optimization_history], f, 
                     ensure_ascii=False, indent=2)
    
    def get_optimization_report(self) -> str:
        """최적화 리포트 생성"""
        if not self.optimization_history:
            return "최적화 기록이 없습니다."
        
        report = "# 자동 정확도 최적화 리포트\n\n"
        
        # 전체 진행 상황
        first_record = self.optimization_history[0]
        last_record = self.optimization_history[-1]
        
        accuracy_improvement = last_record.accuracy - first_record.accuracy
        kt_improvement = last_record.kt_score - first_record.kt_score
        
        report += f"## 전체 진행 상황\n"
        report += f"- 총 반복 수: {len(self.optimization_history)}회\n"
        report += f"- 시작 정확도: {first_record.accuracy:.4f}\n"
        report += f"- 최종 정확도: {last_record.accuracy:.4f}\n"
        report += f"- 정확도 개선: {accuracy_improvement:+.4f}\n"
        report += f"- KT 점수 개선: {kt_improvement:+.4f}\n\n"
        
        # 반복별 상세 기록
        report += "## 반복별 상세 기록\n\n"
        for record in self.optimization_history:
            report += f"### 반복 {record.iteration} ({record.timestamp})\n"
            report += f"- 프롬프트: {record.prompt_path}\n"
            report += f"- 정확도: {record.accuracy:.4f}\n"
            report += f"- KT 점수: {record.kt_score:.4f}\n"
            report += f"- 오답 수: {record.error_count}개\n"
            
            if record.improvements:
                report += "- 개선사항:\n"
                for improvement in record.improvements:
                    report += f"  - {improvement}\n"
            report += "\n"
        
        return report