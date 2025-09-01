#!/usr/bin/env python3
"""
Gemini 2.5 Flash 테스터 - 개선된 프롬프트의 성능을 측정
"""

import os
import csv
import json
import time
import google.generativeai as genai
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path

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

@dataclass
class ErrorAnalysis:
    """오류 분석 결과"""
    total_errors: int
    attribute_errors: Dict[str, int]
    error_patterns: Dict[str, List[str]]
    boundary_cases: List[Dict[str, Any]]
    parsing_failures: List[str]
    confidence_scores: Dict[str, float]

class GeminiFlashTester:
    """Gemini 2.5 Flash를 이용한 프롬프트 성능 테스트"""
    
    def __init__(self):
        self.setup_gemini()
        self.results_dir = Path("prompt/analysis")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_gemini(self):
        """Gemini API 설정"""
        # .env 파일에서 API 키 로드
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY가 .env 파일에 설정되지 않았습니다.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        print("✅ Gemini 2.5 Flash 모델 설정 완료")
    
    def test_full_dataset(self, prompt: str, sample_size: int = None) -> TestResult:
        """전체 데이터셋 또는 샘플로 테스트 수행"""
        print(f"🧪 Gemini 2.5 Flash 테스트 시작...")
        
        # 데이터 로드
        samples = self._load_samples("data/samples.csv")
        
        if sample_size:
            samples = samples[:sample_size]
            print(f"📊 샘플 크기: {len(samples)}개")
        else:
            print(f"📊 전체 데이터셋: {len(samples)}개")
        
        predictions = []
        actual_answers = []
        errors = []
        parsing_failures = []
        
        # 배치 처리로 테스트
        batch_size = 5  # API 제한 고려
        total_batches = (len(samples) + batch_size - 1) // batch_size
        
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            print(f"🔄 배치 {batch_num}/{total_batches} 처리 중...")
            
            try:
                batch_predictions = self._process_batch(prompt, batch)
                
                for j, (sentence, expected) in enumerate(batch):
                    if j < len(batch_predictions):
                        prediction = batch_predictions[j]
                        predictions.append(prediction)
                        actual_answers.append(expected)
                        
                        # 오답 분석
                        if prediction != expected:
                            errors.append({
                                "sentence": sentence,
                                "predicted": prediction,
                                "expected": expected,
                                "error_type": self._analyze_error_type(prediction, expected)
                            })
                    else:
                        # 파싱 실패
                        parsing_failures.append(sentence)
                        predictions.append("파싱실패")
                        actual_answers.append(expected)
                
                # API 제한 고려 대기
                if batch_num < total_batches:
                    time.sleep(2)
                    
            except Exception as e:
                print(f"❌ 배치 {batch_num} 처리 실패: {e}")
                # 실패한 배치는 파싱 실패로 처리
                for sentence, expected in batch:
                    parsing_failures.append(sentence)
                    predictions.append("파싱실패")
                    actual_answers.append(expected)
        
        # 정확도 계산
        accuracy = self.calculate_accuracy(predictions, actual_answers)
        
        result = TestResult(
            total_samples=len(samples),
            correct_predictions=sum(1 for p, a in zip(predictions, actual_answers) if p == a),
            accuracy=accuracy,
            predictions=predictions,
            actual_answers=actual_answers,
            errors=errors,
            parsing_failures=parsing_failures
        )
        
        # 결과 저장
        self._save_test_result(result)
        
        print(f"✅ 테스트 완료: 정확도 {accuracy:.1%}")
        return result
    
    def calculate_accuracy(self, predictions: List[str], answers: List[str]) -> float:
        """정확도 계산"""
        if not predictions or not answers:
            return 0.0
        
        correct = sum(1 for p, a in zip(predictions, answers) if p == a and p != "파싱실패")
        total = len([p for p in predictions if p != "파싱실패"])
        
        return correct / total if total > 0 else 0.0
    
    def analyze_errors(self, test_result: TestResult) -> ErrorAnalysis:
        """상세한 오류 분석 수행"""
        print("🔍 오류 패턴 분석 중...")
        
        # 속성별 오류 집계
        attribute_errors = {"유형": 0, "극성": 0, "시제": 0, "확실성": 0}
        error_patterns = {"유형": [], "극성": [], "시제": [], "확실성": []}
        boundary_cases = []
        
        for error in test_result.errors:
            predicted = error["predicted"].split(",") if "," in error["predicted"] else []
            expected = error["expected"].split(",") if "," in error["expected"] else []
            
            if len(predicted) == 4 and len(expected) == 4:
                attributes = ["유형", "극성", "시제", "확실성"]
                for i, attr in enumerate(attributes):
                    if predicted[i] != expected[i]:
                        attribute_errors[attr] += 1
                        error_patterns[attr].append({
                            "sentence": error["sentence"],
                            "predicted": predicted[i],
                            "expected": expected[i]
                        })
                        
                        # 경계 사례 식별
                        if self._is_boundary_case(error["sentence"], predicted[i], expected[i]):
                            boundary_cases.append(error)
        
        # 신뢰도 점수 계산
        confidence_scores = {}
        for attr in attribute_errors:
            total_samples = test_result.total_samples - len(test_result.parsing_failures)
            if total_samples > 0:
                confidence_scores[attr] = 1 - (attribute_errors[attr] / total_samples)
            else:
                confidence_scores[attr] = 0.0
        
        analysis = ErrorAnalysis(
            total_errors=len(test_result.errors),
            attribute_errors=attribute_errors,
            error_patterns=error_patterns,
            boundary_cases=boundary_cases,
            parsing_failures=test_result.parsing_failures,
            confidence_scores=confidence_scores
        )
        
        # 분석 결과 저장
        self._save_error_analysis(analysis)
        
        return analysis
    
    def check_target_achievement(self, accuracy: float, target: float = 0.7) -> bool:
        """목표 성능 달성 여부 확인"""
        achieved = accuracy >= target
        print(f"🎯 목표 달성 여부: {achieved} (현재: {accuracy:.1%}, 목표: {target:.1%})")
        return achieved
    
    def _load_samples(self, csv_path: str) -> List[Tuple[str, str]]:
        """CSV 파일에서 샘플 데이터 로드"""
        samples = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i == 0:  # 첫 번째 행에서 컬럼명 확인
                    print(f"📋 CSV 컬럼: {list(row.keys())}")
                
                # 컬럼명 확인 후 적절한 키 사용
                if 'user_prompt' in row:
                    sentence = row['user_prompt']
                    answer = row['output']
                elif len(row) >= 2:
                    # 첫 번째와 두 번째 컬럼 사용
                    keys = list(row.keys())
                    sentence = row[keys[0]]
                    answer = row[keys[1]]
                else:
                    continue
                
                samples.append((sentence, answer))
        return samples
    
    def _process_batch(self, prompt: str, batch: List[Tuple[str, str]]) -> List[str]:
        """배치 단위로 문장들을 처리"""
        sentences = [sentence for sentence, _ in batch]
        
        # 배치 프롬프트 구성
        batch_prompt = prompt + "\n\n다음 문장들을 분류하세요:\n"
        for i, sentence in enumerate(sentences, 1):
            batch_prompt += f"{i}. {sentence}\n"
        
        try:
            response = self.model.generate_content(batch_prompt)
            response_text = response.text.strip()
            
            # 응답 파싱
            predictions = self._parse_batch_response(response_text, len(sentences))
            return predictions
            
        except Exception as e:
            print(f"❌ 배치 처리 오류: {e}")
            return ["파싱실패"] * len(sentences)
    
    def _parse_batch_response(self, response_text: str, expected_count: int) -> List[str]:
        """배치 응답을 파싱하여 개별 예측 결과 추출"""
        lines = response_text.strip().split('\n')
        predictions = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 번호 제거 (1. 2. 등)
            if line and line[0].isdigit() and '.' in line:
                line = line.split('.', 1)[1].strip()
            
            # 유효한 분류 결과인지 확인
            if self._is_valid_classification(line):
                predictions.append(line)
        
        # 예상 개수와 맞지 않으면 파싱 실패로 처리
        while len(predictions) < expected_count:
            predictions.append("파싱실패")
        
        return predictions[:expected_count]
    
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
    
    def _analyze_error_type(self, predicted: str, expected: str) -> str:
        """오류 유형 분석"""
        if predicted == "파싱실패":
            return "파싱실패"
        
        if "," not in predicted or "," not in expected:
            return "형식오류"
        
        pred_parts = predicted.split(",")
        exp_parts = expected.split(",")
        
        if len(pred_parts) != 4 or len(exp_parts) != 4:
            return "형식오류"
        
        error_attrs = []
        attributes = ["유형", "극성", "시제", "확실성"]
        
        for i, attr in enumerate(attributes):
            if pred_parts[i] != exp_parts[i]:
                error_attrs.append(attr)
        
        return "+".join(error_attrs) if error_attrs else "기타"
    
    def _is_boundary_case(self, sentence: str, predicted: str, expected: str) -> bool:
        """경계 사례인지 판단"""
        # 특정 키워드가 포함된 경우 경계 사례로 판단
        boundary_keywords = [
            "아쉬움", "떨어졌다", "위험", "논리입니다", 
            "노리는", "것 같다", "제품이다", "18위"
        ]
        
        return any(keyword in sentence for keyword in boundary_keywords)
    
    def _save_test_result(self, result: TestResult):
        """테스트 결과를 파일로 저장"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = self.results_dir / f"test_results_{timestamp}.json"
        
        result_dict = {
            "total_samples": result.total_samples,
            "correct_predictions": result.correct_predictions,
            "accuracy": result.accuracy,
            "error_count": len(result.errors),
            "parsing_failure_count": len(result.parsing_failures),
            "errors": result.errors[:10],  # 처음 10개 오류만 저장
            "parsing_failures": result.parsing_failures[:5]  # 처음 5개만 저장
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
        
        print(f"📄 테스트 결과 저장: {output_file}")
    
    def _save_error_analysis(self, analysis: ErrorAnalysis):
        """오류 분석 결과를 파일로 저장"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = self.results_dir / f"error_analysis_{timestamp}.json"
        
        analysis_dict = {
            "total_errors": analysis.total_errors,
            "attribute_errors": analysis.attribute_errors,
            "confidence_scores": analysis.confidence_scores,
            "boundary_cases_count": len(analysis.boundary_cases),
            "parsing_failures_count": len(analysis.parsing_failures),
            "error_patterns_summary": {
                attr: len(patterns) for attr, patterns in analysis.error_patterns.items()
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_dict, f, ensure_ascii=False, indent=2)
        
        print(f"📊 오류 분석 저장: {output_file}")

def main():
    """메인 실행 함수"""
    tester = GeminiFlashTester()
    
    # 개선된 프롬프트 로드
    prompt_file = Path("prompt/gemini/enhanced_v7_improved.txt")
    
    if not prompt_file.exists():
        print("❌ 개선된 프롬프트 파일을 찾을 수 없습니다.")
        return
    
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt = f.read()
    
    # 소규모 테스트 (10개 샘플)
    print("🧪 소규모 테스트 시작 (10개 샘플)")
    test_result = tester.test_full_dataset(prompt, sample_size=10)
    
    # 오류 분석
    error_analysis = tester.analyze_errors(test_result)
    
    # 목표 달성 여부 확인
    target_achieved = tester.check_target_achievement(test_result.accuracy, 0.7)
    
    print("\n" + "="*50)
    print("📊 테스트 결과 요약")
    print("="*50)
    print(f"정확도: {test_result.accuracy:.1%}")
    print(f"총 샘플: {test_result.total_samples}개")
    print(f"정답: {test_result.correct_predictions}개")
    print(f"오답: {len(test_result.errors)}개")
    print(f"파싱 실패: {len(test_result.parsing_failures)}개")
    print(f"목표 달성: {'✅' if target_achieved else '❌'}")
    
    if error_analysis.attribute_errors:
        print("\n📈 속성별 오류:")
        for attr, count in error_analysis.attribute_errors.items():
            print(f"  {attr}: {count}회 ({error_analysis.confidence_scores[attr]:.1%} 정확도)")

if __name__ == "__main__":
    main()