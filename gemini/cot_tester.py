#!/usr/bin/env python3
"""
Chain-of-Thought 프롬프트 테스터
CoT 적용 전후 성능 비교 및 효과 측정
"""

import os
import csv
import json
import time
from datetime import datetime
import google.generativeai as genai
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

@dataclass
class TestResult:
    """테스트 결과"""
    total_samples: int
    correct_predictions: int
    accuracy: float
    predictions: List[str]
    actual_answers: List[str]
    errors: List[Dict[str, Any]]
    parsing_failures: List[str]

class CoTTester:
    """Chain-of-Thought 프롬프트 성능 테스터"""
    
    def __init__(self):
        self.setup_gemini()
        self.results_dir = Path("prompt/analysis")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_gemini(self):
        """Gemini API 설정"""
        load_dotenv()
        
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY가 .env 파일에 설정되지 않았습니다.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        print("✅ Gemini 2.0 Flash 모델 설정 완료")
        
    def load_prompt(self, prompt_path: str) -> str:
        """프롬프트 파일 로드"""
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            self.logger.error(f"프롬프트 로드 실패: {e}")
            return ""
    
    def test_prompt_performance(self, prompt_path: str, sample_size: int = 10) -> TestResult:
        """프롬프트 성능 테스트"""
        print(f"🧪 프롬프트 테스트 시작: {prompt_path}")
        
        # 프롬프트 로드
        prompt = self.load_prompt(prompt_path)
        if not prompt:
            print("❌ 프롬프트 로드 실패")
            return None
        
        # 테스트 데이터 로드
        samples = self._load_samples('data/samples.csv')
        test_samples = samples[:sample_size]
        print(f"📊 테스트 샘플 수: {len(test_samples)}")
        
        predictions = []
        actual_answers = []
        errors = []
        parsing_failures = []
        
        # 개별 문장 테스트
        for i, (sentence, expected) in enumerate(test_samples):
            try:
                # Gemini API 호출
                full_prompt = prompt + f"\n\n다음 문장을 분류하세요:\n{sentence}"
                response = self.model.generate_content(full_prompt)
                response_text = response.text.strip()
                
                # 응답 파싱
                predicted = self._parse_response(response_text)
                if not predicted:
                    parsing_failures.append(sentence)
                    predictions.append("파싱실패")
                    actual_answers.append(expected)
                    print(f"❌ 샘플 {i+1}: 파싱 실패")
                    continue
                
                predictions.append(predicted)
                actual_answers.append(expected)
                
                # 정답과 비교
                is_correct = predicted == expected
                if not is_correct:
                    errors.append({
                        "sentence": sentence,
                        "predicted": predicted,
                        "expected": expected,
                        "response": response_text
                    })
                
                print(f"{'✅' if is_correct else '❌'} 샘플 {i+1}/{len(test_samples)}: {'정답' if is_correct else '오답'}")
                
                # API 제한 고려 대기
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ 샘플 {i+1} 처리 중 오류: {e}")
                parsing_failures.append(sentence)
                predictions.append("파싱실패")
                actual_answers.append(expected)
        
        # 정확도 계산
        correct_count = sum(1 for p, a in zip(predictions, actual_answers) if p == a and p != "파싱실패")
        valid_count = len([p for p in predictions if p != "파싱실패"])
        accuracy = correct_count / valid_count if valid_count > 0 else 0.0
        
        result = TestResult(
            total_samples=len(test_samples),
            correct_predictions=correct_count,
            accuracy=accuracy,
            predictions=predictions,
            actual_answers=actual_answers,
            errors=errors,
            parsing_failures=parsing_failures
        )
        
        print(f"✅ 테스트 완료 - 정확도: {accuracy:.3f}")
        return result
    
    def _load_samples(self, csv_path: str) -> List[Tuple[str, str]]:
        """CSV 파일에서 샘플 데이터 로드"""
        samples = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'user_prompt' in row:
                    sentence = row['user_prompt']
                    answer = row['output']
                else:
                    keys = list(row.keys())
                    sentence = row[keys[0]]
                    answer = row[keys[1]]
                
                samples.append((sentence, answer))
        return samples
    
    def _parse_response(self, response: str) -> str:
        """응답 파싱"""
        try:
            response = response.strip()
            
            # 여러 줄인 경우 마지막 줄 사용
            lines = response.split('\n')
            for line in reversed(lines):
                line = line.strip()
                if ',' in line and len(line.split(',')) == 4:
                    parts = [part.strip() for part in line.split(',')]
                    if all(parts) and self._is_valid_classification(','.join(parts)):
                        return ','.join(parts)
            
            # 단일 줄 처리
            if ',' in response and len(response.split(',')) == 4:
                parts = [part.strip() for part in response.split(',')]
                if all(parts) and self._is_valid_classification(','.join(parts)):
                    return ','.join(parts)
            
            return None
            
        except Exception as e:
            print(f"❌ 응답 파싱 오류: {e}")
            return None
    
    def _is_valid_classification(self, text: str) -> bool:
        """유효한 분류 결과인지 확인"""
        if not text or "," not in text:
            return False
        
        parts = text.split(",")
        if len(parts) != 4:
            return False
        
        # 각 부분이 유효한 라벨인지 확인
        valid_labels = {
            0: ["사실형", "추론형", "대화형", "예측형"],
            1: ["긍정", "부정", "미정"],
            2: ["과거", "현재", "미래"],
            3: ["확실", "불확실"]
        }
        
        for i, part in enumerate(parts):
            if part.strip() not in valid_labels[i]:
                return False
        
        return True
    
    def compare_prompts(self, baseline_path: str, cot_path: str, sample_size: int = 10) -> Dict[str, Any]:
        """CoT 적용 전후 성능 비교"""
        print("🔄 CoT 적용 전후 성능 비교 시작")
        
        # 기준 프롬프트 테스트
        print("\n📊 기준 프롬프트 테스트:")
        baseline_results = self.test_prompt_performance(baseline_path, sample_size)
        
        # CoT 프롬프트 테스트
        print("\n🧠 CoT 프롬프트 테스트:")
        cot_results = self.test_prompt_performance(cot_path, sample_size)
        
        if not baseline_results or not cot_results:
            return {"error": "테스트 실패"}
        
        # 비교 분석
        comparison = {
            "baseline": {
                "path": baseline_path,
                "accuracy": baseline_results.accuracy,
                "error_count": len(baseline_results.errors),
                "parsing_failures": len(baseline_results.parsing_failures)
            },
            "cot": {
                "path": cot_path,
                "accuracy": cot_results.accuracy,
                "error_count": len(cot_results.errors),
                "parsing_failures": len(cot_results.parsing_failures)
            },
            "improvement": {
                "accuracy_delta": cot_results.accuracy - baseline_results.accuracy,
                "error_reduction": len(baseline_results.errors) - len(cot_results.errors),
                "parsing_improvement": len(baseline_results.parsing_failures) - len(cot_results.parsing_failures)
            },
            "detailed_baseline": {
                "total_samples": baseline_results.total_samples,
                "correct_predictions": baseline_results.correct_predictions,
                "accuracy": baseline_results.accuracy,
                "errors": baseline_results.errors[:5],  # 처음 5개만
                "parsing_failures": baseline_results.parsing_failures[:3]  # 처음 3개만
            },
            "detailed_cot": {
                "total_samples": cot_results.total_samples,
                "correct_predictions": cot_results.correct_predictions,
                "accuracy": cot_results.accuracy,
                "errors": cot_results.errors[:5],  # 처음 5개만
                "parsing_failures": cot_results.parsing_failures[:3]  # 처음 3개만
            }
        }
        
        # 개선 효과 분석
        accuracy_improvement = comparison["improvement"]["accuracy_delta"]
        if accuracy_improvement > 0:
            comparison["conclusion"] = f"CoT 적용으로 {accuracy_improvement:.3f} 정확도 향상"
        elif accuracy_improvement < 0:
            comparison["conclusion"] = f"CoT 적용으로 {abs(accuracy_improvement):.3f} 정확도 하락"
        else:
            comparison["conclusion"] = "CoT 적용 전후 정확도 동일"
        
        print(f"✅ 비교 완료: {comparison['conclusion']}")
        return comparison
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """결과 저장"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"📄 결과 저장 완료: {output_path}")
        except Exception as e:
            print(f"❌ 결과 저장 실패: {e}")

def main():
    """메인 실행 함수"""
    tester = CoTTester()
    
    # 프롬프트 경로 설정
    baseline_prompt = "prompt/gemini/enhanced_few_shot_v20250901_175308.txt"
    cot_prompt = "prompt/gemini/enhanced_cot_v1.txt"
    
    # CoT 적용 전후 성능 비교
    comparison_results = tester.compare_prompts(
        baseline_path=baseline_prompt,
        cot_path=cot_prompt,
        sample_size=10
    )
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"prompt/analysis/cot_comparison_{timestamp}.json"
    tester.save_results(comparison_results, output_path)
    
    # 결과 요약 출력
    print("\n=== Chain-of-Thought 적용 결과 ===")
    print(f"기준 프롬프트 정확도: {comparison_results['baseline']['accuracy']:.3f}")
    print(f"CoT 프롬프트 정확도: {comparison_results['cot']['accuracy']:.3f}")
    print(f"정확도 변화: {comparison_results['improvement']['accuracy_delta']:+.3f}")
    print(f"결론: {comparison_results['conclusion']}")
    print(f"\n상세 결과: {output_path}")

if __name__ == "__main__":
    main()