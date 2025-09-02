"""
정확도 최적화기

samples.csv 기반으로 Few-shot 예시를 선택하고 Chain-of-Thought를 적용하여 정확도 향상
"""

import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import logging
import random

logger = logging.getLogger(__name__)

@dataclass
class Sample:
    """샘플 데이터"""
    sentence: str
    type: str
    polarity: str
    tense: str
    certainty: str

class AccuracyOptimizer:
    """정확도 최적화기"""
    
    def __init__(self, samples_csv_path: str):
        self.samples_csv_path = samples_csv_path
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Sample]:
        """samples.csv 로드"""
        try:
            df = pd.read_csv(self.samples_csv_path)
            samples = []
            
            for _, row in df.iterrows():
                # output 컬럼을 파싱하여 속성 추출
                output_parts = row['output'].strip('"').split(',')
                if len(output_parts) >= 4:
                    sample = Sample(
                        sentence=row['user_prompt'],
                        type=output_parts[0],
                        polarity=output_parts[1],
                        tense=output_parts[2],
                        certainty=output_parts[3]
                    )
                    samples.append(sample)
                
            logger.info(f"샘플 데이터 로드 완료: {len(samples)}개")
            return samples
            
        except Exception as e:
            logger.error(f"샘플 데이터 로드 실패: {e}")
            return []
    
    async def optimize_for_accuracy(self, base_prompt: str, target_accuracy: float = 0.8) -> str:
        """정확도 최적화"""
        logger.info(f"정확도 최적화 시작 - 목표: {target_accuracy}")
        
        # 1단계: Few-shot 예시 추가
        enhanced_prompt = self._add_few_shot_examples(base_prompt)
        
        # 2단계: Chain-of-Thought 추론 과정 추가
        cot_prompt = self._add_chain_of_thought(enhanced_prompt)
        
        # 3단계: 명시적 분류 규칙 강화
        final_prompt = self._enhance_classification_rules(cot_prompt)
        
        return final_prompt
    
    def _add_few_shot_examples(self, prompt: str, num_examples: int = 6) -> str:
        """Few-shot 예시 추가"""
        # 각 속성별로 대표 예시 선택
        examples_by_attribute = self._select_representative_examples()
        
        # 예시 텍스트 생성
        examples_text = "\n\n## 분류 예시\n\n"
        
        for i, sample in enumerate(examples_by_attribute[:num_examples], 1):
            examples_text += f"예시 {i}:\n"
            examples_text += f"문장: \"{sample.sentence}\"\n"
            examples_text += f"분류 결과:\n"
            examples_text += f"- 유형: {sample.type}\n"
            examples_text += f"- 극성: {sample.polarity}\n" 
            examples_text += f"- 시제: {sample.tense}\n"
            examples_text += f"- 확실성: {sample.certainty}\n\n"
        
        # 프롬프트에 예시 삽입
        if "## 분류 예시" in prompt:
            # 기존 예시 섹션 교체
            parts = prompt.split("## 분류 예시")
            if len(parts) > 1:
                # 다음 섹션까지 찾기
                next_section = parts[1].find("##")
                if next_section != -1:
                    prompt = parts[0] + examples_text + "##" + parts[1][next_section+2:]
                else:
                    prompt = parts[0] + examples_text
        else:
            # 새로 추가
            prompt += examples_text
            
        return prompt
    
    def _select_representative_examples(self) -> List[Sample]:
        """대표적인 예시 선택 - 실제 데이터에서 직접 선택"""
        if not self.samples:
            return []
        
        # 각 속성 조합별로 예시 수집
        attribute_combinations = {}
        
        for sample in self.samples:
            key = (sample.type, sample.polarity, sample.tense, sample.certainty)
            if key not in attribute_combinations:
                attribute_combinations[key] = []
            attribute_combinations[key].append(sample)
        
        # 빈도가 높은 조합부터 선택 (실제 데이터 분포 반영)
        sorted_combinations = sorted(
            attribute_combinations.items(), 
            key=lambda x: len(x[1]), 
            reverse=True
        )
        
        selected = []
        for combination, samples in sorted_combinations[:8]:  # 상위 8개 조합
            # 문장 길이가 적당하고 명확한 것 선택 (30-80자)
            suitable_samples = [
                s for s in samples 
                if 30 <= len(s.sentence) <= 80 and not any(
                    ambiguous in s.sentence.lower() 
                    for ambiguous in ['아마도', '혹시', '것 같다', '듯하다']
                )
            ]
            
            if suitable_samples:
                # 가장 명확한 예시 선택
                best_sample = suitable_samples[0]
                selected.append(best_sample)
            elif samples:
                selected.append(samples[0])
        
        logger.info(f"선택된 예시: {len(selected)}개")
        return selected
    
    def _add_chain_of_thought(self, prompt: str) -> str:
        """Chain-of-Thought 추론 과정 추가"""
        cot_section = """
## 분류 추론 과정

다음 단계를 따라 체계적으로 분석하세요:

### 1단계: 문장 구조 분석
- 주어, 서술어, 목적어 파악
- 핵심 키워드 식별
- 문맥과 의도 파악

### 2단계: 각 속성별 판단
**유형 분석:**
- 사실 진술인가? → 사실
- 개인적 견해나 감정인가? → 의견  
- 추측이나 가능성 표현인가? → 추측

**극성 분석:**
- 긍정적 감정/평가 표현 → 긍정
- 부정적 감정/평가 표현 → 부정
- 중립적이거나 객관적 서술 → 중립

**시제 분석:**
- 현재 상황이나 일반적 사실 → 현재
- 과거 사건이나 경험 → 과거
- 미래 계획이나 예측 → 미래

**확실성 분석:**
- 단정적 표현, 명확한 사실 → 확실
- 추측, 가능성, 의문 표현 → 불확실

### 3단계: 최종 분류 결정
각 속성별 분석 결과를 종합하여 최종 분류 결정

"""
        
        # CoT 섹션 추가
        if "## 분류 추론 과정" in prompt:
            # 기존 섹션 교체
            parts = prompt.split("## 분류 추론 과정")
            if len(parts) > 1:
                next_section = parts[1].find("##")
                if next_section != -1:
                    prompt = parts[0] + cot_section + "##" + parts[1][next_section+2:]
                else:
                    prompt = parts[0] + cot_section
        else:
            # 예시 섹션 뒤에 추가
            if "## 분류 예시" in prompt:
                parts = prompt.split("## 분류 예시")
                examples_end = parts[1].find("##")
                if examples_end != -1:
                    prompt = parts[0] + "## 분류 예시" + parts[1][:examples_end] + cot_section + "##" + parts[1][examples_end+2:]
                else:
                    prompt = parts[0] + "## 분류 예시" + parts[1] + cot_section
            else:
                prompt += cot_section
                
        return prompt
    
    def _enhance_classification_rules(self, prompt: str) -> str:
        """명시적 분류 규칙 강화"""
        rules_section = """
## 상세 분류 규칙

### 유형 분류 규칙
**사실 (Fact):**
- 객관적으로 검증 가능한 정보
- 통계, 데이터, 역사적 사실
- "~이다", "~했다" 등 단정적 서술

**의견 (Opinion):**
- 개인적 견해, 감정, 평가
- "좋다", "나쁘다", "아름답다" 등 주관적 표현
- "생각한다", "느낀다" 등 의견 표현

**추측 (Speculation):**
- 불확실한 미래나 가능성
- "~것 같다", "~일 수도", "아마도" 등
- 조건부 표현, 가정법

### 극성 분류 규칙
**긍정 (Positive):**
- 좋음, 만족, 기쁨, 성공 등 긍정적 감정
- 칭찬, 추천, 선호 표현
- "훌륭하다", "좋다", "만족한다"

**부정 (Negative):**
- 나쁨, 불만, 슬픔, 실패 등 부정적 감정  
- 비판, 불평, 거부 표현
- "나쁘다", "싫다", "실망한다"

**중립 (Neutral):**
- 감정적 색채가 없는 객관적 서술
- 단순 정보 전달, 사실 나열
- 중성적 표현

### 시제 분류 규칙
**현재 (Present):**
- 현재 진행 중인 상황
- 일반적 사실, 습관
- "~한다", "~이다", "~고 있다"

**과거 (Past):**
- 이미 끝난 사건, 경험
- "~했다", "~였다", "~던"
- 회상, 기억 표현

**미래 (Future):**
- 앞으로 일어날 일
- "~할 것이다", "~겠다", "~려고 한다"
- 계획, 예측, 의도

### 확실성 분류 규칙
**확실 (Certain):**
- 단정적, 명확한 표현
- 확신을 나타내는 어조
- "분명히", "확실히", "틀림없이"

**불확실 (Uncertain):**
- 추측, 의문, 가능성 표현
- "아마도", "혹시", "~인 것 같다"
- 의문문, 조건부 표현

"""
        
        # 규칙 섹션 추가
        if "## 상세 분류 규칙" in prompt:
            # 기존 섹션 교체
            parts = prompt.split("## 상세 분류 규칙")
            if len(parts) > 1:
                next_section = parts[1].find("##")
                if next_section != -1:
                    prompt = parts[0] + rules_section + "##" + parts[1][next_section+2:]
                else:
                    prompt = parts[0] + rules_section
        else:
            # CoT 섹션 뒤에 추가
            if "## 분류 추론 과정" in prompt:
                parts = prompt.split("## 분류 추론 과정")
                cot_end = parts[1].find("##")
                if cot_end != -1:
                    prompt = parts[0] + "## 분류 추론 과정" + parts[1][:cot_end] + rules_section + "##" + parts[1][cot_end+2:]
                else:
                    prompt = parts[0] + "## 분류 추론 과정" + parts[1] + rules_section
            else:
                prompt += rules_section
                
        return prompt
    
    def analyze_error_patterns(self, errors: List[Dict]) -> Dict[str, List[str]]:
        """오류 패턴 분석"""
        error_patterns = {
            'type': [],
            'polarity': [], 
            'tense': [],
            'certainty': []
        }
        
        for error in errors:
            for attr in error_patterns.keys():
                if attr in error:
                    error_patterns[attr].append(error[attr])
        
        return error_patterns
    
    def get_targeted_examples(self, error_patterns: Dict[str, List[str]]) -> List[Sample]:
        """오류 패턴에 맞는 타겟 예시 선택"""
        targeted = []
        
        # 오류가 많은 속성의 정답 예시 우선 선택
        for attr, errors in error_patterns.items():
            if not errors:
                continue
                
            # 해당 속성에서 정확한 예시들 찾기
            for sample in self.samples:
                attr_value = getattr(sample, attr, None)
                if attr_value and len(targeted) < 8:
                    targeted.append(sample)
        
        return targeted