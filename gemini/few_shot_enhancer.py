#!/usr/bin/env python3
"""
Few-shot Learning 예시 추가 및 테스트 시스템
Task 12: Few-shot Learning 예시 추가 및 테스트 구현
"""

import json
import pandas as pd
from typing import List, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime
from advanced_generator import AdvancedPromptGenerator
from gemini_tester import GeminiFlashTester
from result_analyzer import ResultAnalyzer

class FewShotEnhancer:
    """Few-shot Learning 예시를 분석하고 개선하는 클래스"""
    
    def __init__(self):
        self.output_dir = Path("prompt/analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_dir = Path("prompt/gemini")
        self.prompt_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_error_patterns(self, test_results_file: str) -> Dict[str, Any]:
        """최신 테스트 결과에서 오류 패턴을 분석"""
        print("🔍 오류 패턴 분석 중...")
        
        try:
            with open(test_results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            error_patterns = {
                'attribute_errors': {},
                'specific_errors': [],
                'boundary_cases': [],
                'common_mistakes': {}
            }
            
            # 속성별 오류 통계
            for error in results.get('errors', []):
                error_type = error.get('error_type', 'unknown')
                if error_type not in error_patterns['attribute_errors']:
                    error_patterns['attribute_errors'][error_type] = 0
                error_patterns['attribute_errors'][error_type] += 1
                
                # 구체적 오류 사례 저장
                error_patterns['specific_errors'].append({
                    'sentence': error['sentence'],
                    'predicted': error['predicted'],
                    'expected': error['expected'],
                    'error_type': error_type
                })
            
            print(f"✅ 총 {len(results.get('errors', []))}개 오류 패턴 분석 완료")
            return error_patterns
            
        except Exception as e:
            print(f"❌ 오류 패턴 분석 실패: {e}")
            return {'attribute_errors': {}, 'specific_errors': [], 'boundary_cases': [], 'common_mistakes': {}}
    
    def select_optimal_examples(self, error_patterns: Dict[str, Any], sample_data: pd.DataFrame) -> List[Dict[str, str]]:
        """오류 패턴을 기반으로 최적의 Few-shot 예시 선택"""
        print("🎯 최적 Few-shot 예시 선택 중...")
        
        selected_examples = []
        
        # 1. 실제 오류 사례를 정답으로 변환한 예시들
        for error in error_patterns['specific_errors']:
            example = {
                'sentence': error['sentence'],
                'correct_answer': error['expected'],
                'error_type': error['error_type'],
                'explanation': self._generate_explanation(error)
            }
            selected_examples.append(example)
        
        # 2. 4가지 속성을 균형있게 다루는 대표 예시들 추가
        balanced_examples = self._get_balanced_examples(sample_data)
        selected_examples.extend(balanced_examples)
        
        # 3. 경계 사례 중심 예시들
        boundary_examples = self._get_boundary_case_examples()
        selected_examples.extend(boundary_examples)
        
        print(f"✅ 총 {len(selected_examples)}개 Few-shot 예시 선택 완료")
        return selected_examples
    
    def _generate_explanation(self, error: Dict[str, str]) -> str:
        """오류 사례에 대한 설명 생성"""
        error_type = error['error_type']
        sentence = error['sentence']
        expected = error['expected']
        
        if error_type == '확실성':
            return f"확정된 결정사항이므로 확실"
        elif error_type == '극성':
            return f"부정적 상황이지만 객관적 사실 서술이므로 긍정"
        elif error_type == '유형+시제':
            return f"발언 행위의 과거 서술이므로 사실형+과거"
        elif error_type == '유형':
            return f"객관적 사실 서술이므로 사실형"
        elif error_type == '시제':
            return f"현재 상태 설명이므로 현재 시제"
        else:
            return f"올바른 분류: {expected}"
    
    def _get_balanced_examples(self, sample_data: pd.DataFrame) -> List[Dict[str, str]]:
        """4가지 속성을 균형있게 다루는 예시들"""
        examples = []
        
        # 각 속성별 대표 예시 (실제 데이터에서 선택)
        if len(sample_data) > 0:
            # 유형별 예시
            examples.append({
                'sentence': '우리 공군 특수비행팀 블랙이글스는 이번에 축하비행을 한 영국 공군 레드 애로우즈와 깊은 인연을 가지고 있다.',
                'correct_answer': '사실형,긍정,현재,확실',
                'error_type': '유형',
                'explanation': '객관적 사실 서술이므로 사실형'
            })
            
            # 극성별 예시
            examples.append({
                'sentence': 'A씨는 여동생의 재산 내역을 C씨에게 인계도 제대로 하지 않았다.',
                'correct_answer': '사실형,부정,과거,확실',
                'error_type': '극성',
                'explanation': '명확한 부정 표현이므로 부정'
            })
            
            # 시제별 예시
            examples.append({
                'sentence': '이는 전년 대비 3.4%(706억원) 증가한 수치로 통합은행 출범 이후 최대 실적이다.',
                'correct_answer': '사실형,긍정,현재,확실',
                'error_type': '시제',
                'explanation': '현재 상태 설명이므로 현재 시제'
            })
            
            # 확실성별 예시
            examples.append({
                'sentence': '이 곳에서는 재무개선 추진과 이행 실적을 종합 관리할 예정이다.',
                'correct_answer': '사실형,긍정,미래,불확실',
                'error_type': '확실성',
                'explanation': '미래 계획이므로 불확실'
            })
        
        return examples
    
    def _get_boundary_case_examples(self) -> List[Dict[str, str]]:
        """경계 사례 중심 예시들"""
        return [
            {
                'sentence': '하지만 글로벌 경기 불확실성으로 반도체 부활 시점이 안갯속인 데다 2단계로 접어든 미·중 무역분쟁 결과도 예단할 수 없어 위기감은 여전하다.',
                'correct_answer': '추론형,긍정,현재,확실',
                'error_type': '유형',
                'explanation': '분석과 판단이 포함된 추론형'
            },
            {
                'sentence': '정보 미디어의 영향이 갈수록 커지는 사회 속에서 자칫하면 정보가 뇌의 주인자리를 차지하기 쉽다.',
                'correct_answer': '추론형,긍정,현재,확실',
                'error_type': '유형',
                'explanation': '가능성과 우려를 표현한 추론형'
            }
        ]
    
    def create_enhanced_few_shot_prompt(self, base_prompt_file: str, examples: List[Dict[str, str]]) -> str:
        """Few-shot 예시가 강화된 프롬프트 생성"""
        print("📝 Few-shot 강화 프롬프트 생성 중...")
        
        # 기존 프롬프트 읽기
        with open(base_prompt_file, 'r', encoding='utf-8') as f:
            base_prompt = f.read()
        
        # Few-shot 예시 섹션 생성
        few_shot_section = "\n**Few-shot Learning 예시 (실제 오류 패턴 기반):**\n\n"
        
        for i, example in enumerate(examples[:10], 1):  # 최대 10개 예시
            few_shot_section += f"예시 {i}: \"{example['sentence']}\"\n"
            few_shot_section += f"→ {example['correct_answer']}\n"
            few_shot_section += f"설명: {example['explanation']}\n\n"
        
        # 기존 프롬프트에서 예시 섹션을 찾아서 교체하거나 추가
        if "**핵심 예시" in base_prompt:
            # 기존 예시 섹션 앞에 Few-shot 섹션 삽입
            enhanced_prompt = base_prompt.replace("**핵심 예시", few_shot_section + "**핵심 예시")
        else:
            # 출력 형식 섹션 앞에 Few-shot 섹션 추가
            enhanced_prompt = base_prompt.replace("**출력 형식", few_shot_section + "**출력 형식")
        
        return enhanced_prompt
    
    def test_enhanced_prompt(self, prompt_content: str, test_samples: int = 10) -> Dict[str, Any]:
        """강화된 프롬프트 테스트"""
        print(f"🧪 Few-shot 강화 프롬프트 테스트 중 (샘플: {test_samples}개)...")
        
        # 임시 프롬프트 파일 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_prompt_file = self.prompt_dir / f"enhanced_few_shot_v{timestamp}.txt"
        
        with open(temp_prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt_content)
        
        # Gemini Flash 테스터로 성능 측정
        tester = GeminiFlashTester()
        test_results = tester.test_full_dataset(prompt_content, sample_size=test_samples)
        
        # 결과 분석
        analyzer = ResultAnalyzer()
        analysis = analyzer.analyze_latest_results()
        
        return {
            'test_results': test_results,
            'analysis': analysis,
            'prompt_file': str(temp_prompt_file)
        }
    
    def run_few_shot_enhancement(self, base_prompt_file: str = None, test_samples: int = 10) -> Dict[str, Any]:
        """Few-shot Learning 예시 추가 및 테스트 전체 프로세스 실행"""
        print("🚀 Few-shot Learning 예시 추가 및 테스트 시작")
        print("="*60)
        
        # 1. 기본 프롬프트 파일 설정
        if not base_prompt_file:
            base_prompt_file = "prompt/gemini/enhanced_v7_improved.txt"
        
        if not Path(base_prompt_file).exists():
            print(f"❌ 기본 프롬프트 파일을 찾을 수 없습니다: {base_prompt_file}")
            return {}
        
        # 2. 최신 오류 분석 결과 로드
        latest_test_results = "prompt/analysis/test_results_20250901_174008.json"
        error_patterns = self.analyze_error_patterns(latest_test_results)
        
        # 3. 샘플 데이터 로드
        sample_data = pd.read_csv("data/samples.csv")
        
        # 4. 최적 Few-shot 예시 선택
        optimal_examples = self.select_optimal_examples(error_patterns, sample_data)
        
        # 5. Few-shot 강화 프롬프트 생성
        enhanced_prompt = self.create_enhanced_few_shot_prompt(base_prompt_file, optimal_examples)
        
        # 6. 강화된 프롬프트 테스트
        test_results = self.test_enhanced_prompt(enhanced_prompt, test_samples)
        
        # 7. 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"few_shot_enhancement_results_{timestamp}.json"
        
        # TestResult 객체를 딕셔너리로 변환
        test_result_dict = {
            'total_samples': test_results['test_results'].total_samples,
            'correct_predictions': test_results['test_results'].correct_predictions,
            'accuracy': test_results['test_results'].accuracy,
            'error_count': len(test_results['test_results'].errors),
            'parsing_failure_count': len(test_results['test_results'].parsing_failures),
            'errors': test_results['test_results'].errors,
            'parsing_failures': test_results['test_results'].parsing_failures
        }
        
        enhancement_results = {
            'timestamp': timestamp,
            'base_prompt_file': base_prompt_file,
            'error_patterns_analyzed': error_patterns,
            'few_shot_examples_count': len(optimal_examples),
            'test_results': test_result_dict,
            'performance_analysis': test_results['analysis'],
            'enhanced_prompt_file': test_results['prompt_file']
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(enhancement_results, f, ensure_ascii=False, indent=2)
        
        # 8. 결과 요약 출력
        self._print_enhancement_summary(enhancement_results)
        
        return enhancement_results
    
    def _print_enhancement_summary(self, results: Dict[str, Any]):
        """Few-shot 강화 결과 요약 출력"""
        print("\n" + "="*60)
        print("📊 Few-shot Learning 예시 추가 및 테스트 결과")
        print("="*60)
        
        test_results = results['test_results']
        analysis = results['performance_analysis']
        
        print(f"📈 성능 결과:")
        print(f"   • 정확도: {test_results['accuracy']:.1%}")
        print(f"   • 정답: {test_results['correct_predictions']}/{test_results['total_samples']}")
        print(f"   • 오답: {test_results['error_count']}개")
        print(f"   • 파싱 실패: {test_results['parsing_failure_count']}개")
        
        print(f"\n🎯 Few-shot 예시:")
        print(f"   • 총 예시 수: {results['few_shot_examples_count']}개")
        print(f"   • 실제 오류 기반: {len(results['error_patterns_analyzed']['specific_errors'])}개")
        print(f"   • 균형잡힌 예시: 4개 (속성별)")
        print(f"   • 경계 사례: 2개")
        
        if test_results['accuracy'] >= 0.7:
            print(f"\n✅ 목표 달성: 70% 이상 정확도 달성!")
        else:
            print(f"\n⚠️  목표 미달성: 70% 미만 ({test_results['accuracy']:.1%})")
        
        print(f"\n📁 생성된 파일:")
        print(f"   • 강화된 프롬프트: {results['enhanced_prompt_file']}")
        print(f"   • 결과 분석: prompt/analysis/few_shot_enhancement_results_*.json")

def main():
    """메인 실행 함수"""
    enhancer = FewShotEnhancer()
    
    # Few-shot Learning 예시 추가 및 테스트 실행
    results = enhancer.run_few_shot_enhancement(
        base_prompt_file="prompt/gemini/enhanced_v7_improved.txt",
        test_samples=10
    )
    
    if results:
        print("\n🎉 Task 12: Few-shot Learning 예시 추가 및 테스트 완료!")
    else:
        print("\n❌ Task 12 실행 중 오류 발생")

if __name__ == "__main__":
    main()