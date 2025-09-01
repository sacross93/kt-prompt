#!/usr/bin/env python3
"""
결과 분석기 - 테스트 결과를 분석하여 다음 개선 방향을 도출
"""

import json
import glob
from typing import Dict, List, Any
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ImprovementPlan:
    """개선 계획"""
    priority_areas: List[str]
    specific_fixes: List[str]
    next_strategies: List[str]
    expected_improvement: float

class ResultAnalyzer:
    """테스트 결과를 분석하여 개선 방향을 도출하는 클래스"""
    
    def __init__(self):
        self.analysis_dir = Path("prompt/analysis")
    
    def analyze_latest_results(self) -> Dict[str, Any]:
        """최신 테스트 결과를 분석"""
        print("📊 최신 테스트 결과 분석 중...")
        
        # 최신 테스트 결과 파일 찾기
        test_files = list(self.analysis_dir.glob("test_results_*.json"))
        error_files = list(self.analysis_dir.glob("error_analysis_*.json"))
        
        if not test_files or not error_files:
            print("❌ 분석할 테스트 결과가 없습니다.")
            return {}
        
        # 최신 파일 선택
        latest_test = max(test_files, key=lambda x: x.stat().st_mtime)
        latest_error = max(error_files, key=lambda x: x.stat().st_mtime)
        
        # 결과 로드
        with open(latest_test, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        with open(latest_error, 'r', encoding='utf-8') as f:
            error_data = json.load(f)
        
        # 종합 분석
        analysis = self._perform_comprehensive_analysis(test_data, error_data)
        
        # 분석 결과 저장
        self._save_analysis_report(analysis)
        
        return analysis
    
    def generate_improvement_plan(self, analysis: Dict[str, Any]) -> ImprovementPlan:
        """분석 결과를 바탕으로 개선 계획 생성"""
        print("🎯 개선 계획 생성 중...")
        
        current_accuracy = analysis.get("accuracy", 0.0)
        attribute_errors = analysis.get("attribute_errors", {})
        
        # 우선순위 영역 결정 (오류가 많은 순서)
        priority_areas = []
        sorted_errors = sorted(attribute_errors.items(), key=lambda x: x[1], reverse=True)
        
        for attr, count in sorted_errors:
            if count > 0:
                priority_areas.append(f"{attr} 분류 개선 ({count}회 오류)")
        
        # 구체적 수정사항
        specific_fixes = self._generate_specific_fixes(attribute_errors, analysis.get("errors", []))
        
        # 다음 전략
        next_strategies = self._suggest_next_strategies(current_accuracy, attribute_errors)
        
        # 예상 개선도
        expected_improvement = self._estimate_improvement(current_accuracy, attribute_errors)
        
        plan = ImprovementPlan(
            priority_areas=priority_areas,
            specific_fixes=specific_fixes,
            next_strategies=next_strategies,
            expected_improvement=expected_improvement
        )
        
        # 개선 계획 저장
        self._save_improvement_plan(plan)
        
        return plan
    
    def _perform_comprehensive_analysis(self, test_data: Dict, error_data: Dict) -> Dict[str, Any]:
        """종합적인 분석 수행"""
        analysis = {
            "accuracy": test_data.get("accuracy", 0.0),
            "total_samples": test_data.get("total_samples", 0),
            "correct_predictions": test_data.get("correct_predictions", 0),
            "error_count": test_data.get("error_count", 0),
            "parsing_failure_count": test_data.get("parsing_failure_count", 0),
            "attribute_errors": error_data.get("attribute_errors", {}),
            "confidence_scores": error_data.get("confidence_scores", {}),
            "errors": test_data.get("errors", [])
        }
        
        # 성능 평가
        analysis["performance_level"] = self._evaluate_performance_level(analysis["accuracy"])
        
        # 주요 문제점 식별
        analysis["main_issues"] = self._identify_main_issues(analysis)
        
        # 개선 가능성 평가
        analysis["improvement_potential"] = self._assess_improvement_potential(analysis)
        
        return analysis
    
    def _generate_specific_fixes(self, attribute_errors: Dict[str, int], errors: List[Dict]) -> List[str]:
        """구체적인 수정사항 생성"""
        fixes = []
        
        # 속성별 오류에 따른 수정사항
        if attribute_errors.get("극성", 0) > 0:
            fixes.append("부정적 상황의 사실 서술 = 긍정 극성 규칙 강화")
            fixes.append("명확한 부정 표현 예시 추가")
        
        if attribute_errors.get("유형", 0) > 0:
            fixes.append("개인 경험의 객관적 서술 = 사실형 규칙 명확화")
            fixes.append("추론형 vs 사실형 경계 사례 예시 보강")
        
        if attribute_errors.get("시제", 0) > 0:
            fixes.append("현재 상태 설명 = 현재 시제 규칙 강조")
            fixes.append("시제 판단 기준 세분화")
        
        if attribute_errors.get("확실성", 0) > 0:
            fixes.append("확실성 판단 기준 명확화")
            fixes.append("추측 표현 식별 규칙 강화")
        
        # 실제 오류 사례 기반 수정사항
        for error in errors[:3]:  # 처음 3개 오류만 분석
            sentence = error.get("sentence", "")
            if "아쉬움" in sentence:
                fixes.append("감정 표현의 객관적 서술 처리 규칙 추가")
            elif "떨어졌다" in sentence:
                fixes.append("부정적 결과의 사실 서술 예시 강화")
        
        return list(set(fixes))  # 중복 제거
    
    def _suggest_next_strategies(self, accuracy: float, attribute_errors: Dict[str, int]) -> List[str]:
        """다음 전략 제안"""
        strategies = []
        
        if accuracy < 0.75:
            strategies.append("Few-shot learning 예시 확장")
            strategies.append("경계 사례 중심 규칙 세분화")
        
        if accuracy < 0.8:
            strategies.append("Chain-of-Thought 추론 과정 도입")
            strategies.append("속성별 우선순위 재조정")
        
        # 가장 문제가 되는 속성에 따른 전략
        max_error_attr = max(attribute_errors.items(), key=lambda x: x[1])[0] if attribute_errors else None
        
        if max_error_attr == "극성":
            strategies.append("극성 분류 전용 예시 집중 추가")
        elif max_error_attr == "유형":
            strategies.append("유형 분류 결정 트리 방식 도입")
        elif max_error_attr == "시제":
            strategies.append("시제 표현 패턴 매칭 강화")
        
        return strategies
    
    def _estimate_improvement(self, current_accuracy: float, attribute_errors: Dict[str, int]) -> float:
        """예상 개선도 추정"""
        total_errors = sum(attribute_errors.values())
        
        if total_errors == 0:
            return current_accuracy
        
        # 각 오류를 50% 개선한다고 가정
        potential_fixes = total_errors * 0.5
        total_samples = 10  # 현재 테스트 샘플 수
        
        improvement = potential_fixes / total_samples
        estimated_accuracy = min(current_accuracy + improvement, 1.0)
        
        return estimated_accuracy
    
    def _evaluate_performance_level(self, accuracy: float) -> str:
        """성능 수준 평가"""
        if accuracy >= 0.85:
            return "우수"
        elif accuracy >= 0.75:
            return "양호"
        elif accuracy >= 0.65:
            return "보통"
        else:
            return "개선필요"
    
    def _identify_main_issues(self, analysis: Dict) -> List[str]:
        """주요 문제점 식별"""
        issues = []
        
        accuracy = analysis["accuracy"]
        attribute_errors = analysis["attribute_errors"]
        parsing_failures = analysis["parsing_failure_count"]
        
        if accuracy < 0.7:
            issues.append("전체 정확도가 목표치 미달")
        
        if parsing_failures > 0:
            issues.append("출력 형식 파싱 오류 발생")
        
        # 속성별 문제점
        for attr, count in attribute_errors.items():
            if count > 1:
                issues.append(f"{attr} 분류에서 반복적 오류")
        
        return issues
    
    def _assess_improvement_potential(self, analysis: Dict) -> str:
        """개선 가능성 평가"""
        accuracy = analysis["accuracy"]
        total_errors = sum(analysis["attribute_errors"].values())
        
        if accuracy >= 0.8:
            return "미세 조정 단계"
        elif total_errors <= 3:
            return "높은 개선 가능성"
        elif total_errors <= 5:
            return "중간 개선 가능성"
        else:
            return "대폭 개선 필요"
    
    def _save_analysis_report(self, analysis: Dict[str, Any]):
        """분석 리포트 저장"""
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.analysis_dir / f"comprehensive_analysis_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        
        print(f"📄 종합 분석 저장: {output_file}")
    
    def _save_improvement_plan(self, plan: ImprovementPlan):
        """개선 계획 저장"""
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.analysis_dir / f"improvement_plan_{timestamp}.json"
        
        plan_dict = {
            "priority_areas": plan.priority_areas,
            "specific_fixes": plan.specific_fixes,
            "next_strategies": plan.next_strategies,
            "expected_improvement": plan.expected_improvement
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(plan_dict, f, ensure_ascii=False, indent=2)
        
        print(f"🎯 개선 계획 저장: {output_file}")
    
    def generate_summary_report(self, analysis: Dict[str, Any], plan: ImprovementPlan) -> str:
        """요약 리포트 생성"""
        report = f"""
# 프롬프트 성능 분석 및 개선 계획

## 현재 성능
- **정확도**: {analysis['accuracy']:.1%}
- **성능 수준**: {analysis['performance_level']}
- **총 샘플**: {analysis['total_samples']}개
- **정답**: {analysis['correct_predictions']}개
- **오답**: {analysis['error_count']}개

## 속성별 성능
"""
        
        for attr, score in analysis.get('confidence_scores', {}).items():
            error_count = analysis.get('attribute_errors', {}).get(attr, 0)
            report += f"- **{attr}**: {score:.1%} 정확도 ({error_count}회 오류)\n"
        
        report += f"""
## 주요 문제점
"""
        for issue in analysis.get('main_issues', []):
            report += f"- {issue}\n"
        
        report += f"""
## 개선 계획

### 우선순위 영역
"""
        for area in plan.priority_areas:
            report += f"- {area}\n"
        
        report += f"""
### 구체적 수정사항
"""
        for fix in plan.specific_fixes:
            report += f"- {fix}\n"
        
        report += f"""
### 다음 전략
"""
        for strategy in plan.next_strategies:
            report += f"- {strategy}\n"
        
        report += f"""
## 예상 성과
- **목표 정확도**: {plan.expected_improvement:.1%}
- **개선 가능성**: {analysis['improvement_potential']}

## 다음 단계
1. 우선순위 영역부터 프롬프트 수정
2. 소규모 테스트로 개선 효과 확인
3. 점진적으로 샘플 크기 확대
4. 목표 달성까지 반복 개선
"""
        
        return report

def main():
    """메인 실행 함수"""
    analyzer = ResultAnalyzer()
    
    # 최신 결과 분석
    analysis = analyzer.analyze_latest_results()
    
    if not analysis:
        print("❌ 분석할 데이터가 없습니다.")
        return
    
    # 개선 계획 생성
    improvement_plan = analyzer.generate_improvement_plan(analysis)
    
    # 요약 리포트 생성
    summary_report = analyzer.generate_summary_report(analysis, improvement_plan)
    
    # 리포트 저장
    report_file = analyzer.analysis_dir / "latest_analysis_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(summary_report)
    
    print("\n" + "="*60)
    print("📊 결과 분석 완료")
    print("="*60)
    print(f"현재 정확도: {analysis['accuracy']:.1%}")
    print(f"성능 수준: {analysis['performance_level']}")
    print(f"예상 개선도: {improvement_plan.expected_improvement:.1%}")
    print(f"개선 가능성: {analysis['improvement_potential']}")
    print(f"\n📄 상세 리포트: {report_file}")
    
    # 주요 개선사항 출력
    if improvement_plan.priority_areas:
        print(f"\n🎯 우선 개선 영역:")
        for area in improvement_plan.priority_areas[:3]:
            print(f"  - {area}")

if __name__ == "__main__":
    main()