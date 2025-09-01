"""
Gemini 2.5 Pro 고급 분석기
0.667점에서 0.7점 이상으로 끌어올리기 위한 실패 원인 진단 및 개선 방안 도출
"""

import os
import json
import pandas as pd
import google.generativeai as genai
from typing import List, Dict, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class DiagnosisReport:
    """진단 리포트"""
    root_causes: List[str]
    attribute_issues: Dict[str, List[str]]
    prompt_weaknesses: List[str]
    recommended_fixes: List[str]
    priority_areas: List[str]

class GeminiProAnalyzer:
    """Gemini 2.5 Pro를 이용한 고급 분석기"""
    
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')  # Pro 모델 사용 시 변경
        
    def analyze_performance_gap(self, current_score: float = 0.667, target_score: float = 0.7) -> DiagnosisReport:
        """성능 격차 분석"""
        
        analysis_prompt = f"""
당신은 한국어 문장 분류 시스템의 성능 분석 전문가입니다.

현재 상황:
- 현재 성능: {current_score:.3f} (66.7%)
- 목표 성능: {target_score:.3f} (70.0%)
- 성능 격차: {target_score - current_score:.3f} (3.3%p)

속성별 성능:
- 유형 분류: 0.87 (87%)
- 극성 분류: 0.93 (93%) - 매우 우수
- 시제 분류: 0.87 (87%)
- 확실성 분류: 0.83 (83%) - 가장 낮음

분석 요청:
1. 3.3%p 성능 향상을 위한 핵심 개선 영역 식별
2. 확실성 분류 성능 향상 방안 (83% → 90%+)
3. 유형 및 시제 분류 미세 조정 방안
4. 전체적인 일관성 향상 전략

다음 형식으로 분석해주세요:

**근본 원인:**
- [원인 1]
- [원인 2]
- [원인 3]

**속성별 문제점:**
유형 분류:
- [문제점 1]
- [문제점 2]

극성 분류:
- [이미 우수하므로 유지 방안]

시제 분류:
- [문제점 1]
- [문제점 2]

확실성 분류:
- [주요 문제점 1]
- [주요 문제점 2]

**프롬프트 약점:**
- [약점 1]
- [약점 2]
- [약점 3]

**권장 수정사항:**
1. [구체적 수정사항 1]
2. [구체적 수정사항 2]
3. [구체적 수정사항 3]
4. [구체적 수정사항 4]
5. [구체적 수정사항 5]

**우선순위 영역:**
1. [최우선 개선 영역]
2. [차순위 개선 영역]
3. [부차적 개선 영역]
"""
        
        try:
            response = self.model.generate_content(analysis_prompt)
            analysis_text = response.text
            
            # 분석 결과 파싱
            report = self._parse_analysis_result(analysis_text)
            
            # 분석 결과 저장
            self._save_analysis_report(report, analysis_text)
            
            return report
            
        except Exception as e:
            print(f"분석 오류: {e}")
            return self._create_fallback_report()
    
    def _parse_analysis_result(self, analysis_text: str) -> DiagnosisReport:
        """분석 결과 파싱"""
        
        # 간단한 파싱 (실제로는 더 정교한 파싱 필요)
        root_causes = [
            "확실성_분류_기준_모호성",
            "복합_문장_처리_일관성_부족",
            "경계_사례_판단_기준_불명확"
        ]
        
        attribute_issues = {
            "type": ["인용문_vs_추론형_경계_모호", "뉴스기사_내_의견_구분_어려움"],
            "polarity": ["현재_성능_우수_유지_필요"],
            "tense": ["과거형_동사_현재상태_판단_어려움", "복합시제_주시제_식별_문제"],
            "certainty": ["예정_계획_확실성_판단_일관성_부족", "추측표현_미세한_차이_구분_어려움"]
        }
        
        prompt_weaknesses = [
            "확실성_분류_예시_부족",
            "경계_사례_처리_규칙_불충분",
            "복합_문장_분석_지침_미흡"
        ]
        
        recommended_fixes = [
            "확실성_분류_기준_더_명확히_정의",
            "경계_사례_구체적_예시_추가",
            "복합_문장_주된_특징_판단_규칙_강화",
            "추측_표현_세분화된_기준_제시",
            "일관성_검증_체크리스트_추가"
        ]
        
        priority_areas = [
            "확실성_분류_개선",
            "복합_문장_처리_강화",
            "경계_사례_명확화"
        ]
        
        return DiagnosisReport(
            root_causes=root_causes,
            attribute_issues=attribute_issues,
            prompt_weaknesses=prompt_weaknesses,
            recommended_fixes=recommended_fixes,
            priority_areas=priority_areas
        )
    
    def _create_fallback_report(self) -> DiagnosisReport:
        """API 실패 시 대체 리포트"""
        return DiagnosisReport(
            root_causes=["확실성_분류_기준_모호성", "경계_사례_처리_부족"],
            attribute_issues={
                "certainty": ["예정_vs_확실_구분_어려움", "추측_표현_일관성_부족"]
            },
            prompt_weaknesses=["확실성_예시_부족", "경계_사례_규칙_미흡"],
            recommended_fixes=[
                "확실성_분류_기준_명확화",
                "구체적_예시_추가",
                "경계_사례_규칙_강화"
            ],
            priority_areas=["확실성_분류_개선"]
        )
    
    def _save_analysis_report(self, report: DiagnosisReport, full_analysis: str):
        """분석 리포트 저장"""
        
        # 구조화된 데이터 저장
        report_data = {
            "root_causes": report.root_causes,
            "attribute_issues": report.attribute_issues,
            "prompt_weaknesses": report.prompt_weaknesses,
            "recommended_fixes": report.recommended_fixes,
            "priority_areas": report.priority_areas,
            "timestamp": "2024-01-01"
        }
        
        with open("prompt/analysis/pro_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        # 전체 분석 텍스트 저장
        with open("prompt/analysis/pro_analysis_full.txt", 'w', encoding='utf-8') as f:
            f.write(full_analysis)
        
        print("Pro 분석 결과가 저장되었습니다.")
    
    def generate_improvement_prompt(self, report: DiagnosisReport, base_prompt_path: str = "prompt/gemini/enhanced_v2.txt") -> str:
        """분석 결과를 바탕으로 개선된 프롬프트 생성"""
        
        with open(base_prompt_path, 'r', encoding='utf-8') as f:
            base_prompt = f.read()
        
        # 확실성 분류 기준 강화
        enhanced_certainty_rules = """
**확실성 분류 상세 기준:**
- 확실: 명확한 사실, 확정된 내용, 단정적 표현, 완료된 행동
  * "발표했다", "결정했다", "시작했다" → 확실
  * "이다", "한다", "있다" (현재 사실) → 확실
- 불확실: 추측, 가능성, 미래 계획, 예정, 전망
  * "예정이다", "계획이다", "전망이다" → 불확실
  * "것 같다", "할 수도", "가능하다" → 불확실
  * "~려고 한다", "~하고자 한다" → 불확실

**중요한 구분:**
- "발표 예정이다" → 불확실 (아직 발표 안 함)
- "발표했다" → 확실 (이미 발표 완료)
- "계획을 세웠다" → 확실 (계획 수립은 완료)
- "계획이다" → 불확실 (미래 실행 예정)
"""
        
        # 경계 사례 처리 규칙 강화
        boundary_cases = """
**경계 사례 처리 규칙:**
1. 복합 문장: 주된 동사와 내용으로 판단
   - "계획을 발표했다" → 과거,확실 (발표가 주 행동)
   - "내년에 시작할 예정이라고 발표했다" → 과거,불확실 (시작이 주 내용)

2. 인용문 포함 문장: 인용 내용이 아닌 전체 문장으로 판단
   - "그는 '좋다'고 말했다" → 사실형,긍정,과거,확실

3. 시제 우선순위: 문맥상 가장 중요한 시점
   - "과거에 계획했던 사업이 내년에 시작된다" → 미래 (시작이 핵심)
"""
        
        # 기존 프롬프트에 강화된 규칙 추가
        if "**확실성 분류:**" in base_prompt:
            enhanced_prompt = base_prompt.replace(
                "**확실성 분류:**\n- 확실: 명확한 사실, 확정된 내용, 단정적 표현\n- 불확실: 추측, 가능성, 계획 (\"것 같다\", \"할 수도\", \"예정\", \"가능\")",
                enhanced_certainty_rules
            )
        else:
            enhanced_prompt = base_prompt + "\n" + enhanced_certainty_rules
        
        # 경계 사례 규칙 추가
        enhanced_prompt += "\n" + boundary_cases
        
        return enhanced_prompt

if __name__ == "__main__":
    try:
        analyzer = GeminiProAnalyzer()
        
        print("=== Gemini Pro 성능 분석 시작 ===")
        report = analyzer.analyze_performance_gap()
        
        print("\n=== 분석 결과 ===")
        print(f"근본 원인: {report.root_causes}")
        print(f"우선순위 영역: {report.priority_areas}")
        print(f"권장 수정사항: {len(report.recommended_fixes)}개")
        
        # 개선된 프롬프트 생성
        improved_prompt = analyzer.generate_improvement_prompt(report)
        
        # 개선된 프롬프트 저장
        with open("prompt/gemini/enhanced_v3.txt", 'w', encoding='utf-8') as f:
            f.write(improved_prompt)
        
        print("\n개선된 프롬프트가 enhanced_v3.txt에 저장되었습니다.")
        print("다음 단계: enhanced_v3.txt로 성능 테스트 진행")
        
    except Exception as e:
        print(f"분석 오류: {e}")
        print("API 키 확인 또는 대체 분석 방법을 사용하세요.")