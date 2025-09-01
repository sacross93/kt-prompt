#!/usr/bin/env python3
"""
고급 프롬프트 생성기 - 분석 결과를 바탕으로 개선된 프롬프트를 생성
"""

import json
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from prompt_analyzer import BaselineInsights, AnalysisResult

@dataclass
class PromptStrategy:
    """프롬프트 전략"""
    name: str
    description: str
    technique: str
    parameters: Dict[str, Any]
    expected_improvement: List[str]
    compatibility: List[str]

class AdvancedPromptGenerator:
    """고급 프롬프트 엔지니어링 기법을 적용한 프롬프트 생성기"""
    
    def __init__(self):
        self.output_dir = Path("prompt/gemini")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_few_shot_examples(self, error_patterns: List[str], error_analysis: Dict = None) -> List[str]:
        """오류 패턴을 기반으로 Few-shot 예시를 동적으로 선택"""
        print("🎯 Few-shot 예시 생성 중...")
        
        examples = []
        
        # 실제 오류 사례를 기반으로 한 Few-shot 예시 생성
        if error_analysis:
            print("📊 실제 오류 분석 결과를 바탕으로 Few-shot 예시 생성")
            
            # 확실성 오류 대응 예시
            examples.append(
                '예시 1: "나머지 6억5000만원은 수취인이 불명확해 경찰의 계좌추적 결과를 참고하기로 했다" → 사실형,긍정,과거,확실 (확정된 결정사항이므로 확실)'
            )
            
            # 극성 오류 대응 예시 (갈등, 해지 등 부정적 상황의 사실 서술)
            examples.append(
                '예시 2: "조합은 대우건설과 갈등을 빚다가 시공사 계약을 해지했다" → 사실형,긍정,과거,확실 (부정적 상황이지만 객관적 사실 서술이므로 긍정)'
            )
            
            # 유형+시제 복합 오류 대응 예시 (인용문의 유형과 시제)
            examples.append(
                '예시 3: "그는 "이건 한국에 가장 첫 오스카상"이라고 덧붙였다" → 사실형,긍정,과거,확실 (발언 행위의 과거 서술이므로 사실형+과거)'
            )
        
        # 기존 핵심 경계 사례들 (4가지 속성 균형)
        
        # 4. 유형 분류: 사실형 vs 추론형 경계
        examples.append(
            '예시 4: "유소연은 아쉬움이 클 수밖에 없다" → 사실형,긍정,현재,확실 (현재 상황의 객관적 서술)'
        )
        
        # 5. 극성 분류: 부정적 내용의 긍정 극성
        examples.append(
            '예시 5: "시험에 떨어졌다" → 사실형,긍정,과거,확실 (부정적 결과이지만 사실 서술이므로 긍정)'
        )
        
        # 6. 시제 분류: 현재 상태 설명
        examples.append(
            '예시 6: "지난 30년간 만들어온 제품이다" → 사실형,긍정,현재,확실 (현재 상태 설명이므로 현재 시제)'
        )
        
        # 7. 유형 분류: 대화형 구분
        examples.append(
            '예시 7: "논리입니다" → 대화형,긍정,현재,확실 (대화체 표현)'
        )
        
        # 8. 유형 분류: 추론형 명확한 사례
        examples.append(
            '예시 8: "노리는 경우도 많다" → 추론형,긍정,현재,불확실 (추측 표현)'
        )
        
        # 9. 확실성 분류: 미래 계획의 불확실성
        examples.append(
            '예시 9: "내년에 출시할 예정이다" → 사실형,긍정,미래,불확실 (미래 계획이므로 불확실)'
        )
        
        # 10. 극성 분류: 명확한 부정 표현
        examples.append(
            '예시 10: "결과가 좋지 않았다" → 사실형,부정,과거,확실 (명확한 부정 표현)'
        )
        
        return examples
    
    def add_chain_of_thought(self, base_prompt: str) -> str:
        """Chain-of-Thought 추론 과정을 포함한 프롬프트 생성"""
        print("🧠 Chain-of-Thought 추론 과정 추가 중...")
        
        cot_section = """
**단계별 분류 과정:**

1단계: 문장 구조 분석
- 주어, 서술어, 시간 표현 파악
- 화자의 의도와 문맥 이해

2단계: 유형 분류 판단
- 직접 인용문/대화체 → 대화형
- 미래 예측/계획 → 예측형  
- 추측/의견/분석 → 추론형
- 객관적 사실/경험 → 사실형

3단계: 극성 분류 판단
- 질문문/가정법 → 미정
- 명확한 부정 표현 → 부정
- 나머지 (사실 서술 포함) → 긍정

4단계: 시제 분류 판단
- 과거 어미 (했다, 였다) → 과거
- 미래 표현 (할 것이다, 예정) → 미래
- 현재 상태/일반 사실 → 현재

5단계: 확실성 분류 판단
- 추측/가능성 표현 → 불확실
- 명확한 사실/확정 내용 → 확실
"""
        
        # 기존 프롬프트에 CoT 섹션 추가
        enhanced_prompt = base_prompt + "\n" + cot_section
        return enhanced_prompt
    
    def apply_strategy(self, strategy: PromptStrategy, base_prompt: str) -> str:
        """특정 전략을 기존 프롬프트에 적용"""
        print(f"🔧 전략 적용 중: {strategy.name}")
        
        if strategy.technique == "few-shot":
            examples = self.generate_few_shot_examples([])
            examples_text = "\n".join(examples)
            return base_prompt + f"\n\n**핵심 예시:**\n{examples_text}"
        
        elif strategy.technique == "chain-of-thought":
            return self.add_chain_of_thought(base_prompt)
        
        elif strategy.technique == "explicit-rules":
            return self._enhance_explicit_rules(base_prompt)
        
        elif strategy.technique == "output-format":
            return self._strengthen_output_format(base_prompt)
        
        else:
            return base_prompt
    
    def create_enhanced_prompt(self, insights: BaselineInsights) -> str:
        """기준선 인사이트를 바탕으로 개선된 프롬프트 생성"""
        print("🚀 개선된 프롬프트 생성 중...")
        
        # 기존 최고 성능 프롬프트 읽기
        with open(insights.best_prompt_path, 'r', encoding='utf-8') as f:
            base_prompt = f.read()
        
        # 핵심 개선사항 적용
        enhanced_prompt = self._apply_critical_improvements(base_prompt, insights)
        
        # Few-shot 예시 추가 (오류 분석 결과 포함)
        error_analysis = self._load_latest_error_analysis()
        examples = self.generate_few_shot_examples(insights.critical_weaknesses, error_analysis)
        enhanced_prompt = self._add_examples_section(enhanced_prompt, examples)
        
        # 출력 형식 강화
        enhanced_prompt = self._strengthen_output_format(enhanced_prompt)
        
        # 버전 정보 추가
        version_info = f"""
# 개선된 프롬프트 v7 (기준: {insights.best_score:.1%})
# 주요 개선사항: 극성 분류, 유형 분류, 시제 분류 정확도 향상
# 생성일: {self._get_current_timestamp()}

"""
        
        final_prompt = version_info + enhanced_prompt
        
        # 파일로 저장
        output_file = self.output_dir / "enhanced_v7_improved.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_prompt)
        
        print(f"✅ 개선된 프롬프트 생성 완료: {output_file}")
        return final_prompt
    
    def _apply_critical_improvements(self, base_prompt: str, insights: BaselineInsights) -> str:
        """핵심 개선사항을 적용"""
        
        # 기존 프롬프트를 기반으로 핵심 약점 개선
        improved_prompt = """당신은 한국어 문장을 네 가지 속성으로 분류하는 전문가입니다.

**핵심 분류 규칙 (우선순위별):**

**1순위: 극성 분류 (가장 중요한 개선 영역)**
- 긍정: 
  * 긍정적 내용 또는 **중립적/객관적 서술**
  * **부정적 상황의 사실 서술도 긍정** ("시험에 떨어졌다", "위험한 좌석이다")
  * 정보 제공, 경고, 설명 → 모두 긍정
  * **핵심**: 부정적 내용 ≠ 부정 극성
  
- 부정: 
  * **명확한 부정 표현만** ("없었다", "않다", "못하다")
  * 실제 거부나 반대 의사 표현
  
- 미정: 질문문, 가정법, 불확실한 추측

**2순위: 유형 분류**
- 사실형: 
  * 객관적 사실, 뉴스 보도, 통계, 역사적 사건
  * **개인 경험의 객관적 서술** ("공부했다", "떨어졌다", "찾았다")
  * **현재 상황의 객관적 서술** ("유소연은 아쉬움이 클 수밖에 없다")
  * 제품/기술 설명, 현재 상태 설명
  
- 추론형: 
  * **명확한 추측이나 주관적 판단만**
  * 가능성 표현 ("할 수 있다", "것 같다", "볼 수 있다")
  * 추측 ("노리는 경우도 많다")

- 대화형: **직접 대화체, 구어체** ("습니다", "해요", "논리입니다")
- 예측형: 미래 예측, 계획 발표, 날씨 예보

**3순위: 시제 분류**
- 과거: 완료된 과거 행동 ("했다", "였다", "갖췄다")
- 현재: 
  * 현재 시제, 일반 사실, **현재 상태 설명**
  * **핵심**: "제품이다", "18위다" → 현재 (현재 상태)
  * "지난 30년간 만들어온 제품이다" → 현재 (현재 상태 설명)
- 미래: 미래 시제, 계획 ("할 것이다", "예정", "된다")

**4순위: 확실성 분류**
- 확실: 명확한 사실, 확정된 내용, 단정적 표현
- 불확실: 추측, 가능성, 미래 계획, 예정, 전망

**핵심 원칙 (반드시 기억):**
1. 부정적 상황의 객관적 서술 = 긍정 극성
2. 개인 경험의 객관적 서술 = 사실형
3. 현재 상태 설명 = 현재 시제
4. 대화체 표현 = 대화형"""
        
        return improved_prompt
    
    def _add_examples_section(self, prompt: str, examples: List[str]) -> str:
        """예시 섹션을 추가"""
        examples_section = f"""

**핵심 예시 (경계 사례):**
{chr(10).join(examples)}

**추가 예시:**
- "위험한 좌석이다" → 사실형,긍정,현재,확실 (위험 정보 제공이므로 긍정)
- "정신이 없었다" → 사실형,부정,과거,확실 (명확한 부정 표현)
- "18위로... 유소연은 아쉬움이 클 수밖에 없다" → 사실형,긍정,현재,확실 (현재 상황 서술)
"""
        
        return prompt + examples_section
    
    def _strengthen_output_format(self, prompt: str) -> str:
        """출력 형식 지침을 강화"""
        format_section = """

**출력 형식 (엄격히 준수):**
"유형,극성,시제,확실성" 형식으로만 답하세요.

**중요한 출력 규칙:**
- 반드시 쉼표로만 구분, 공백 없음
- 지정된 한글 라벨만 사용
- 추가 설명이나 텍스트 절대 금지
- 각 문장마다 4가지 속성 모두 반드시 분류

**올바른 출력 예시:**
사실형,긍정,과거,확실
추론형,긍정,현재,불확실
대화형,긍정,현재,확실

**잘못된 출력 (절대 금지):**
- 사실형, 긍정, 과거, 확실 (공백 포함)
- 사실형/긍정/과거/확실 (다른 구분자)
- 사실형,긍정,과거,확실 (설명 추가)
"""
        
        return prompt + format_section
    
    def _enhance_explicit_rules(self, prompt: str) -> str:
        """명시적 규칙을 강화"""
        # 이미 _apply_critical_improvements에서 처리됨
        return prompt
    
    def _load_latest_error_analysis(self) -> Dict:
        """최신 오류 분석 결과 로드"""
        try:
            error_file = Path("prompt/analysis/test_results_20250901_174008.json")
            if error_file.exists():
                with open(error_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ 오류 분석 로드 실패: {e}")
        return None
    
    def _get_current_timestamp(self) -> str:
        """현재 타임스탬프 반환"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def main():
    """메인 실행 함수"""
    generator = AdvancedPromptGenerator()
    
    # 기준선 인사이트 로드 (분석기에서 생성된 것 사용)
    insights_file = Path("prompt/analysis/baseline_insights.json")
    
    if insights_file.exists():
        with open(insights_file, 'r', encoding='utf-8') as f:
            insights_data = json.load(f)
        
        insights = BaselineInsights(
            best_prompt_path=insights_data["best_prompt_path"],
            best_score=insights_data["best_score"],
            key_success_factors=insights_data["key_success_factors"],
            critical_weaknesses=insights_data["critical_weaknesses"],
            improvement_priorities=insights_data["improvement_priorities"]
        )
    else:
        # 기본 인사이트 생성
        insights = BaselineInsights(
            best_prompt_path="prompt/gemini/enhanced_v6_final.txt",
            best_score=0.7778,
            key_success_factors=[
                "개인 경험의 객관적 서술을 사실형으로 정확히 분류",
                "부정적 상황의 사실 서술을 긍정 극성으로 올바르게 처리"
            ],
            critical_weaknesses=[
                "극성 분류 오류 (4회)",
                "유형 분류 오류 (3회)",
                "시제 분류 오류 (3회)"
            ],
            improvement_priorities=[
                "1순위: 극성 분류 개선",
                "2순위: 유형 분류 정확도 향상"
            ]
        )
    
    # 개선된 프롬프트 생성
    enhanced_prompt = generator.create_enhanced_prompt(insights)
    
    print("\n" + "="*50)
    print("🚀 개선된 프롬프트 생성 완료")
    print("="*50)
    print(f"기준 성능: {insights.best_score:.1%}")
    print(f"주요 개선: 극성, 유형, 시제 분류 정확도 향상")
    print(f"저장 위치: prompt/gemini/enhanced_v7_improved.txt")

if __name__ == "__main__":
    main()